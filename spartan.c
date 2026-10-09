/* Online SPARTAN - the maths. Knows nothing about files, datasets or printing.
 *
 * Everything is a plain array of doubles. A table with k rows and m columns
 * is one flat block; row j starts at index j*m. Same idea everywhere.
 */

#include "spartan.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
/* ---------------- basic maths ---------------- */

double dot_product(const double* v1, const double* v2, size_t n) {
  double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
  size_t i = 0;
  double t_sum = 0.0;
  if (n >= 4) {
    for (; i <= n - 4; i += 4) {
      sum0 += v1[i] * v2[i];
      sum1 += v1[i + 1] * v2[i + 1];
      sum2 += v1[i + 2] * v2[i + 2];
      sum3 += v1[i + 3] * v2[i + 3];
    }
  }
  t_sum = sum0 + sum1 + sum2 + sum3;
  for (; i < n; i++) {
    t_sum += v1[i] * v2[i];
  }
  return t_sum;
}

void znorm_row(double* x, int m) {
  double sum = 0.0;
  double variance = 0.0;
  double mean, std_dev;

  for (int i = 0; i < m; i++) {
    sum += x[i];
  }
  mean = sum / m;

  for (int i = 0; i < m; i++) {
    variance += pow(x[i] - mean, 2);
  }

  /* Divide by m, not m-1. The reference does the same, and matching it
     matters more than the textbook choice. */
  std_dev = sqrt(variance / m) + SPARTAN_EPS;

  double inv_std_dev = 1.0 / std_dev;
  for (int i = 0; i < m; i++) {
    x[i] = (x[i] - mean) * inv_std_dev;
  }
}

/* ---------------- running average of the stream ---------------- */

int rmean_init(RunningMean* rm, int m) {
  if (m <= 0) return -1;
  rm->mean = calloc((size_t)m, sizeof(double));
  if (!rm->mean) return -1;
  rm->n = 0;
  rm->m = m;
  return 0;
}

void rmean_free(RunningMean* rm) {
  free(rm->mean);
  rm->mean = NULL;
  rm->n = 0;
  rm->m = 0;
}

void rmean_update(RunningMean* rm, const double* x) {
  int m = rm->m;
  rm->n++;
  for (int i = 0; i < m; i++) rm->mean[i] += (x[i] - rm->mean[i]) / rm->n;
}

void pring_free(ProjRing* r) {
  if (!r) return;
  free(r->buf);
  free(r->s1);
  free(r->s2);
  memset(r, 0, sizeof(*r));
}

int pring_init(ProjRing* r, int cap, int k) {
  if (!r || cap < 2 || k < 1) return -1;
  memset(r, 0, sizeof(*r));
  r->buf = calloc((size_t)cap * k, sizeof(double));
  r->s1 = calloc((size_t)k, sizeof(double));
  r->s2 = calloc((size_t)k, sizeof(double));
  if (!r->buf || !r->s1 || !r->s2) {
    pring_free(r);
    return -1;
  }
  r->cap = cap;
  r->k = k;
  return 0;
}

void pring_push(ProjRing* r, const double* y) {
  double* row = &r->buf[(size_t)r->cursor * r->k];

  if (r->count == r->cap) {
    for (int j = 0; j < r->k; j++) {
      double old_val = row[j];
      r->s1[j] -= old_val;
      r->s2[j] -= old_val * old_val;
    }
  } else {
    r->count++;
  }

  for (int j = 0; j < r->k; j++) {
    row[j] = y[j];
    r->s1[j] += y[j];
    r->s2[j] += y[j] * y[j];
  }

  if (++r->cursor == r->cap) {
    r->cursor = 0;
  }
}

void pring_importances(const ProjRing* r, double* ev) {
  int n = r->count;
  if (n < 1) {
    for (int j = 0; j < r->k; j++) ev[j] = 1.0 / r->k;
    return;
  }
  double total = 0.0;
  for (int j = 0; j < r->k; j++) {
    double mean = r->s1[j] / n;
    double var = r->s2[j] / n - mean * mean;
    if (var < 0.0) var = 0.0;

    ev[j] = var;
    total += var;
  }
  if (total < SPARTAN_EPS) {
    for (int j = 0; j < r->k; j++) ev[j] = 1.0 / r->k;
    return;
  }
  for (int j = 0; j < r->k; j++) ev[j] /= total;
}

void pring_refresh(ProjRing* r) {
  if (r == NULL) {
    return;
  }

  memset(r->s1, 0, (size_t)r->k * sizeof(double));
  memset(r->s2, 0, (size_t)r->k * sizeof(double));

  for (int i = 0; i < r->count; i++) {
    const double* row = &r->buf[(size_t)i * r->k];
    for (int j = 0; j < r->k; j++) {
      double val = row[j];
      r->s1[j] += val;
      r->s2[j] += val * val;
    }
  }
}

void pring_reset(ProjRing* r) {
  if (r == NULL || r->buf == NULL) {
    return;
  }
  r->cursor = 0;
  r->count = 0;
  memset(r->s1, 0, (size_t)r->k * sizeof(double));

  memset(r->s2, 0, (size_t)r->k * sizeof(double));

  memset(r->buf, 0, (size_t)r->cap * (size_t)r->k * sizeof(double));
}

/* ---------------- the PCA directions ---------------- */

int oja_init(OjaPCA* p, int k, int m) {
  if (k > m || k <= 0 || m <= 0) return -1;

  p->k = k;
  p->m = m;

  p->v = calloc((size_t)k * m, sizeof(double));
  if (!p->v) return -1;

  p->u = calloc((size_t)m, sizeof(double));
  if (!p->u) {
    free(p->v);
    p->v = NULL;
    return -1;
  }

  /* Start each direction on a different axis. All-zero would never move. */
  for (int j = 0; j < k; j++) p->v[j * m + j] = 1.0;

  return 0;
}

void oja_free(OjaPCA* p) {
  free(p->v);
  p->v = NULL;
  free(p->u);
  p->u = NULL;
  p->k = 0;
  p->m = 0;
}

void oja_update(OjaPCA* p, const double* xc, double lr) {
  int k = p->k, m = p->m;
  double* u = p->u;

  for (int i = 0; i < m; i++) u[i] = xc[i];

  for (int j = 0; j < k; j++) {
    double* vj = &p->v[j * m];
    double y = dot_product(u, vj, m);

    for (int i = 0; i < m; i++) {
      vj[i] += lr * y * (u[i] - y * vj[i]);
    }

    /* Measure again AFTER the move, then take that part out of the copy, so
       the next direction only sees what is left. Reusing the old measurement
       here is the classic way to break this. */
    double d = dot_product(u, vj, m);

    for (int i = 0; i < m; i++) {
      u[i] -= d * vj[i];
    }
  }
}

void gram_schmidt(OjaPCA* p) {
  int k = p->k, m = p->m;

  for (int j = 0; j < k; j++) {
    double* vj = &p->v[j * m];

    for (int p_idx = 0; p_idx < j; p_idx++) {
      double* vp = &p->v[p_idx * m];
      double d = dot_product(vj, vp, m);
      for (int i = 0; i < m; i++) {
        vj[i] -= d * vp[i];
      }
    }

    double nrm = sqrt(dot_product(vj, vj, m));

    /* A direction that collapsed to nothing is skipped, not treated as an
       error: the remaining ones are still usable. */
    if (nrm < SPARTAN_EPS) {
      continue;
    }

    for (int i = 0; i < m; i++) {
      vj[i] = vj[i] / nrm;
    }
  }
}

void oja_orthonormality_error(const OjaPCA* p, double* max_dot,
                              double* max_len) {
  int k = p->k, m = p->m;
  double worst_dot = 0.0;
  double worst_len = 0.0;

  for (int j = 0; j < k; j++) {
    const double* vj = &p->v[j * m];

    double len = sqrt(dot_product(vj, vj, m));
    double len_err = fabs(len - 1.0);
    if (len_err > worst_len) worst_len = len_err;

    for (int i = j + 1; i < k; i++) {
      double d = fabs(dot_product(vj, &p->v[i * m], m));
      if (d > worst_dot) worst_dot = d;
    }
  }

  if (max_dot) *max_dot = worst_dot;
  if (max_len) *max_len = worst_len;
}

int oja_copy_from(OjaPCA* dst, const OjaPCA* src) {
  if (!dst || !src) {
    return -1;
  }
  if (!dst->v || !src->v) {
    return -1;
  }

  if (dst->k != src->k || dst->m != src->m) {
    return -2;
  }
  size_t total_elements = (size_t)src->k * (size_t)src->m;
  memcpy(dst->v, src->v, total_elements * sizeof(double));
  return 0;
}
/* ---------------- from numbers to letters ---------------- */

static int cmp_double(const void* a, const void* b) {
  double x = *(const double*)a;
  double y = *(const double*)b;
  if (x < y) return -1;
  if (x > y) return 1;
  return 0;
}

void project(const OjaPCA* p, const double* xc, double* y) {
  for (int j = 0; j < p->k; j++) {
    y[j] = dot_product(xc, &p->v[j * p->m], p->m);
  }
}

void compute_breakpoints(const double* proj, int n, int k, const int* alphabet,
                         double* bkpt) {
  double* col = calloc((size_t)n, sizeof(double));
  if (!col) return;
  for (int j = 0; j < k; j++) {
    double target_depth = (double)n / alphabet[j];
    for (int r = 0; r < n; r++) col[r] = proj[r * k + j];
    qsort(col, n, sizeof(double), cmp_double);
    double bin_index = 0.0;
    for (int bp = 0; bp < alphabet[j] - 1; bp++) {
      bin_index += target_depth;
      bkpt[MAX_ALPHABET * j + bp] = col[(int)bin_index];
    }
    /* Last slot is a wall the search always stops at. */
    bkpt[MAX_ALPHABET * j + (alphabet[j] - 1)] = DBL_MAX;
  }
  free(col);
}

/* The PCA directions all have length one, so their own size tells us nothing
   about how important they are. The spread of the projected values does, and
   we already collect those during warm-up, so this costs nothing extra. */
void estimate_eigenvalues(const double* proj, int n, int k, double* ev) {
  double sum = 0.0;
  for (int j = 0; j < k; j++) {
    double col_sum = 0.0;
    for (int r = 0; r < n; r++) {
      col_sum += proj[r * k + j];
    }
    double mean = col_sum / n;
    double diff_sum = 0.0;
    for (int r = 0; r < n; r++) {
      double diff = proj[r * k + j] - mean;
      diff_sum += diff * diff;
    }
    double variance = diff_sum / n;
    ev[j] = variance;
    sum += variance;
  }
  if (sum < SPARTAN_EPS) {
    for (int j = 0; j < k; j++) {
      ev[j] = 1.0 / k;
    }
    return;
  }
  for (int j = 0; j < k; j++) {
    ev[j] /= sum;
  }
}

void digitize(const double* y, const double* bkpt, int k, const int* alphabet,
              int* word) {
  for (int j = 0; j < k; j++) {
    int s = 0;
    while (s < alphabet[j] - 1 && y[j] > bkpt[j * MAX_ALPHABET + s]) {
      s++;
    }
    word[j] = s;
  }
}

/* ---------------- splitting the bit budget ---------------- */

/* Cost of moving away from the reference split. Returns a NEGATIVE number;
   whoever calls it adds it on. Same shape as regularization_term in
   spartan.py, except the reference point can be chosen per position. */
static double regularization_term(int x, double ev_value, int ref_bit,
                                  double lamda) {
  int dev = x - ref_bit;
  return -lamda * (double)(dev * dev) * ev_value;
}

double daa_score(const double* ev, int k, int total_bits, double lamda,
                 const int* ref_bits, const int* bits) {
  int avg_bit = total_bits / k;
  double s = 0.0;
  for (int i = 0; i < k; i++) {
    int ref = ref_bits ? ref_bits[i] : avg_bit;
    s += bits[i] * ev[i] + regularization_term(bits[i], ev[i], ref, lamda);
  }
  return s;
}

/* The unreachable marker. It has to be a value that cannot be produced by
   adding a gain to it, which -1e9 is not: -1e9 + 0.125 beats -1e9 and the cell
   gets filled from a path that does not exist. Compared with ==, never <. */
#define DAA_UNREACHABLE (-DBL_MAX)

int daa_allocate(const double* ev, int k, int total_bits, double lamda,
                 const int* ref_bits, int* bits, double* score) {
  int N = total_bits;
  int cols = N + 1;
  int min_bit = 0;
  if (k <= 0 || N < 1) return -1;

  /* Whole division on purpose. spartan.py uses int(N/K), so a budget that
     does not divide evenly puts the reference point slightly low, and we
     copy that rather than "fix" it. */
  int avg_bit = N / k;

  /* Two real ceilings and nothing else: what is left if everyone else takes
     the minimum, and what one position can spend before its alphabet hits
     MAX_ALPHABET.

     The reference caps at ev_max * N instead, which compares a share of the
     BIT budget against a share of the VARIANCE. Those are not the same
     currency: bits are logarithmic, variance is not. In batch, with a steep
     spectrum, the cap never binds. In a stream the spectrum flattens, the cap
     tightens, and it starts excluding the optimum exactly when the decision
     gets hard. */

  int max_bit = N - (k - 1) * min_bit;
  if (max_bit > MAX_BITS) max_bit = MAX_BITS;

  /* No shortcut for N == k: with a zero floor the even split is no longer
     automatically best. ev (0.80, 0.15, 0.05) with N = k = 3 prefers 2 1 0. */
  double* DP = malloc((size_t)(k + 1) * cols * sizeof(double));
  int* alloc = malloc((size_t)(k + 1) * cols * sizeof(int));
  if (!DP || !alloc) {
    free(DP);
    free(alloc);
    return -1;
  }

  /* Every cell starts unreachable. The one reachable start is "no positions
     filled, no bits spent". The companion table starts at the full budget,
     which acts as "no limit yet" for the never-increase test below. */
  for (int i = 0; i <= k; i++) {
    for (int j = 0; j < cols; j++) {
      DP[i * cols + j] = DAA_UNREACHABLE;
      alloc[i * cols + j] = N;
    }
  }
  DP[0] = 0.0;

  for (int i = 1; i <= k; i++) {
    /* This is the addition: with no reference given we fall back to the even
       split and behave exactly like spartan.py. With one given, the split
       prefers to stay where it was and only moves when the gain is real. */
    int ref = ref_bits ? ref_bits[i - 1] : avg_bit;

    for (int j = 0; j < cols; j++) {
      double best = DAA_UNREACHABLE;

      for (int x = min_bit; x <= max_bit; x++) {
        if (j - x < 0) continue;

        /* No position may get more bits than the one before it. Allowed only
           because the directions arrive sorted from most to least important.
         */
        if (x > alloc[(i - 1) * cols + (j - x)]) continue;

        /* == and not <, because adding a gain to the sentinel would beat it */
        if (DP[(i - 1) * cols + (j - x)] == DAA_UNREACHABLE) continue;

        double cr = DP[(i - 1) * cols + (j - x)] + x * ev[i - 1] +
                    regularization_term(x, ev[i - 1], ref, lamda);

        if (cr > best) {
          best = cr;
          alloc[i * cols + j] = x;
          DP[i * cols + j] = cr;
        }
      }
    }
  }
  /* No feasible split reaches the full budget. Say so instead of handing back
     the sentinel as if it were a score. */
  if (DP[k * cols + N] == DAA_UNREACHABLE) {
    free(DP);
    free(alloc);
    return -1;
  }
  if (score) *score = DP[k * cols + N];

  /* Walk back from the last position. Whatever is left over goes to the
     first one. Same result as the reference, which builds the list backwards
     and then flips it. */
  int unused_bit = N;
  for (int i = k; i >= 2; i--) {
    bits[i - 1] = alloc[i * cols + unused_bit];
    unused_bit -= bits[i - 1];
  }
  bits[0] = unused_bit;

  free(DP);
  free(alloc);

  /* The reference asserts here. A library returns an error instead. */
  int sum = 0;
  for (int i = 0; i < k; i++) {
    if (bits[i] < min_bit || bits[i] > max_bit) return -1;
    sum += bits[i];
  }
  if (sum != N) return -1;

  return 0;
}

static double position_gain(int x, double ev_value, int ref_bit, double lamda) {
  return x * ev_value + regularization_term(x, ev_value, ref_bit, lamda);
}

double daa_gap(const double* ev, int k, int total_bits, double lamda,
               const int* bits) {
  int ref_bit = total_bits / k;
  int min_bit = 0;
  int max_bit = total_bits - (k - 1) * min_bit;
  if (max_bit > MAX_BITS) max_bit = MAX_BITS;
  double best_bid = -DBL_MAX;
  double best_ask = DBL_MAX;
  for (int j = 0; j < k; j++) {
    double cur_gain = position_gain(bits[j], ev[j], ref_bit, lamda);
    if (bits[j] < max_bit) {
      double up = position_gain(bits[j] + 1, ev[j], ref_bit, lamda) - cur_gain;
      if (up > best_bid) best_bid = up;
    }
    if (bits[j] > min_bit) {
      double down =
          cur_gain - position_gain(bits[j] - 1, ev[j], ref_bit, lamda);

      if (down < best_ask) best_ask = down;
    }
  }

  if (best_bid == -DBL_MAX || best_ask == DBL_MAX) {
    return 0.0;
  }

  double gap = best_bid - best_ask;
  if (gap <= 0) {
    return 0;
  }
  return gap;
}
/* ---------------- settings ---------------- */

Config config_default(int k, int m) { return config_default_mv(k, 1, m); }

Config config_default_mv(int k, int d, int m_time) {
  Config c = {
      .k = k,
      .d = d,
      .m_time = m_time,
      .m = d * m_time,
      .stride = m_time,
      .lr = 0.01,
      .gs_period = 5,
      .warmup = 10,
      .total_bits = 2 * k,
      .lambda = 0.5,
  };
  c.ring_w = c.warmup;
  return c;
}
