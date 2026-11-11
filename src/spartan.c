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
    for (; i <= n - 4;
         i += 4) {  // Unroll 4x into independent accumulators to break FMA
                    // latency chains and enable SIMD vectorization
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

/*
 * The two numbers z-norm throws away: where the window sits and how big it
 * is. Same formulas as znorm_row, so the window can be rebuilt exactly as
 *
 *   x = mu + sd * znorm(x)
 *
 *   mu = (1/m) * sum_i x_i                  [same units as x]
 *   sd = sqrt( (1/m) * sum_i (x_i - mu)^2 ) [same units as x]
 *
 * x      one channel of one window, NOT yet z-normed
 * m      its length, m_time
 * mu, sd outputs
 *
 * Edge cases:
 *   Call BEFORE znorm_row, which overwrites x: afterwards these are 0 and 1.
 *   No SPARTAN_EPS here, unlike znorm_row. A flat channel has spread 0,
 *   which is the true value and a legal letter.
 */
void window_level(const double* x, int m, double* mu, double* sd) {
  double sum = 0.0;
  for (int i = 0; i < m; i++) sum += x[i];
  double mean = sum / m;
  double sq = 0.0;
  for (int i = 0; i < m; i++) sq += (x[i] - mean) * (x[i] - mean);
  *mu = mean;
  *sd = sqrt(sq / m);
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
  double* row =
      &r->buf[(size_t)r->cursor *
              r->k];  // Alias the oldest k-element slot at the circular write
                      // cursor for O(k) moment eviction and overwrite

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
    double var =
        r->s2[j] / n -
        mean * mean;  // Compute variance via E[Y^2] - (E[Y])^2 and clamp below
                      // at 0.0 to absorb floating-point cancellation
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
    double* vj = &p->v[j * m];  // Slice row j from the flat k-by-m basis matrix
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
      double* vp =
          &p->v[p_idx * m];  // Modified Gram-Schmidt: deflate vj sequentially
                             // against already-orthogonalized rows 0 .. j-1
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

/*
 * Reorders the rows of a basis in place.
 *
 * Row i of the result is row perm[i] of the input: a "gather", not a
 * "scatter". Pass the inverse by mistake and you get the inverse
 * permutation, which looks plausible and is wrong.
 *
 * p     the basis to reorder; k and m are unchanged
 * perm  k entries, a permutation of 0..k-1
 *
 * Returns 0 on success, -1 on a bad argument or a failed allocation.
 *
 * Edge cases:
 *   Reordering rows is multiplication by a permutation matrix, which is
 *   orthogonal, so an orthonormal basis stays orthonormal, the subspace it
 *   spans does not move, and principal angles against any other basis are
 *   unchanged. theta_max, theta_noise and staleness all read the same.
 *   NEVER call this on the live Oja basis. There, row order is not a label:
 *   gram_schmidt orthogonalises row j against rows 0..j-1, so the order
 *   decides the dynamics. Snapshots only.
 */
int oja_permute_rows(OjaPCA* p, const int* perm) {
  if (!p || !p->v || !perm || p->k < 1 || p->m < 1) return -1;

  double* temporary = malloc((size_t)p->k * (size_t)p->m * sizeof(double));
  if (!temporary) return -1;
  for (int i = 0; i < p->k; i++) {
    int src_row = perm[i];
    if (src_row < 0 || src_row >= p->k) {
      free(temporary);
      return -1;
    }
    memcpy(
        &temporary[i * p->m], &p->v[src_row * p->m],
        (size_t)p->m *
            sizeof(double));  // Gather old row perm[i] into new row i so basis
                              // directions match descending eigenvalue order
  }
  memcpy(p->v, temporary, (size_t)p->k * (size_t)p->m * sizeof(double));
  free(temporary);
  return 0;
}

int oja_blend(OjaPCA* dst, const OjaPCA* src, double beta) {
  if (!dst || !src || !dst->v || !src->v) {
    return -1;
  }
  if (dst->k != src->k || dst->m != src->m) {
    return -1;
  }
  if (!(beta >= 0.0) || !(beta <= 1.0)) {
    return -1;
  }

  size_t total = (size_t)src->k * (size_t)src->m;
  for (size_t i = 0; i < total; i++) {
    dst->v[i] = (1.0 - beta) * dst->v[i] + beta * src->v[i];
  }
  return 0;
}

/* ---------------- singular values of a small matrix ---------------- */

#define SVD_MAX_SWEEPS 30

#define SVD_TOL 1e-14

int svd_values(double* a, int n, double* sv, double* jacc) {
  if (!a || !sv || n < 1) {
    return -1;
  }
  if (jacc) {
    memset(jacc, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
      jacc[i * n + i] = 1.0;
    }
  }

  for (int sweep = 0; sweep < SVD_MAX_SWEEPS; sweep++) {
    int rotated = 0;

    for (int p = 0; p < n - 1; p++) {
      for (int q = p + 1; q < n; q++) {
        /* The three Gram numbers for this pair of rows. */
        double alpha = dot_product(
            &a[p * n], &a[p * n],
            n);  // Form 2x2 Gram entries for rows p and q of the n-by-n matrix
        double beta = dot_product(&a[q * n], &a[q * n], n);
        double gamma = dot_product(&a[p * n], &a[q * n], n);

        /* Already orthogonal to tolerance: skip. This test MUST come before
           the rotation is worked out, because that divides by gamma. */
        if (fabs(gamma) <= SVD_TOL * sqrt(alpha * beta)) {
          continue;
        }

        /* The rotation that kills gamma. Use the form that adds in the
           denominator; the algebraically equal one subtracts and loses every
           digit when the rows are nearly orthogonal, which is most of the
           time near convergence. */
        double zeta = (beta - alpha) / (2.0 * gamma);
        double t =
            (zeta >= 0.0 ? 1.0 : -1.0) /
            (fabs(zeta) +
             sqrt(
                 1 +
                 (zeta *
                  zeta)));  // Solve quadratic Jacobi tangent using sign-matched
                            // denominator to avoid catastrophic cancellation
        double c = 1 / sqrt(1 + (t * t));
        double s = c * t;

        for (int j = 0; j < n; j++) {
          double pj = a[p * n + j];
          double qj = a[q * n + j];
          a[p * n + j] = c * pj - s * qj;
          a[q * n + j] = s * pj + c * qj;
        }
        if (jacc) {
          for (int j = 0; j < n; j++) {
            double jjp = jacc[p * n + j];
            double jjq = jacc[q * n + j];
            jacc[p * n + j] = c * jjp - s * jjq;
            jacc[q * n + j] = s * jjp + c * jjq;
          }
        }

        rotated = 1;
      }
    }

    if (!rotated) {
      for (int i = 0; i < n; i++) {
        sv[i] = sqrt(dot_product(&a[i * n], &a[i * n], n));
      }
      return sweep + 1;
    }
  }

  return -1;
}

int subspace_angles(const OjaPCA* frozen, const OjaPCA* live, double* scratch,
                    double* angles, double* max_angle) {
  /* guard: both bases present and initialised, same k and same m, and a
     scratch to work in. */
  if (!scratch || !frozen || !live || !frozen->v || !live->v) {
    return -1;
  }
  if (frozen->k <= 0 || frozen->m <= 0 || frozen->k != live->k ||
      frozen->m != live->m) {
    return -1;
  }

  int k = frozen->k;
  int m = frozen->m;

  /* fill the k by k with the inner products between every frozen row and
     every live row. row major, so entry (i, j) sits at scratch[i * k + j]. */
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      double result = dot_product(&frozen->v[i * m], &live->v[j * m], m);
      scratch[i * k + j] = result;
    }
  }

  /* svd_values destroys what it is given, which is why scratch exists and
     why it has to be refilled on every call rather than kept. */
  double* sv =
      &scratch[(size_t)k *
               k];  // Partition caller scratch buffer: first k*k doubles hold
                    // Gram matrix, tail k doubles hold singular values
  int sweeps = svd_values(scratch, k, sv, NULL);

  if (sweeps < 0) {
    return -2;
  }

  /* each singular value is a cosine. clamp before the arc cosine: a value
     one rounding step above one gives nan, and every comparison with nan is
     false, so a threshold test downstream would silently never fire. */
  double max_th = 0.0;
  for (int j = 0; j < k; j++) {
    double co_val = sv[j];
    if (co_val > 1.0) co_val = 1.0;
    if (co_val < 0.0) co_val = 0.0;
    double angle_theta = acos(co_val);
    if (angles) {
      angles[j] = angle_theta;
    }
    if (angle_theta > max_th) max_th = angle_theta;
  }
  if (max_angle) {
    *max_angle = max_th;
  }

  /* the largest angle comes from the smallest cosine, because the cosine
     decreases as the angle grows. */

  return 0;
}

/*
 * Builds the map that carries coordinates from one epoch's basis into
 * another's, and measures how much of the old subspace the new one
 * cannot see.
 *
 * The k by k table of inner products between the two bases is not a
 * rotation: it contracts, and what it contracts is exactly the part the
 * old basis could not represent. A contraction cannot appear in a
 * distance bound, so this returns the nearest rotation to it, plus the
 * number the slack is built from.
 *
 *   sin_theta = sqrt(1 - s_min^2),  s_min = smallest singular value
 *
 * to         the basis we translate INTO    (k rows of m, orthonormal)
 * from       the basis the coordinates came FROM (same k and m)
 * scratch    2*k*k + k doubles, all overwritten
 * rot        output, k*k row major: entry (i, j) at rot[i*k + j]
 * sin_theta  output, dimensionless, in [0, 1]
 *
 * Returns 0 on success, -1 on a bad argument, -2 if the Jacobi sweep did
 * not converge, -3 if a direction pair has collapsed (a singular value at
 * zero): the two epochs are not translatable and the caller must skip the
 * pair rather than use a meaningless rotation.
 *
 * Edge cases:
 *   Both bases must be orthonormal or none of this is angles or lengths.
 *   rot may have determinant -1, a reflection. Harmless here: a
 *   reflection preserves lengths and the bound needs only lengths. It
 *   matters only for the Lie layer, which needs a logarithm.
 */
int epoch_map(const OjaPCA* to, const OjaPCA* from, double* scratch,
              double* map, double* rot, double* sin_theta) {
  if (!scratch || !to || !from || !to->v || !from->v || !rot || !sin_theta) {
    return -1;
  }
  if (to->k <= 0 || to->m <= 0 || to->k != from->k || to->m != from->m) {
    return -1;
  }

  int k = to->k;
  int m = to->m;

  /* Three regions in one block. m_buf is destroyed by the sweep, jacc
     collects the rotations it applied, sv ends up holding the row
     lengths, which are the cosines. */
  double* m_buf = scratch;
  double* jacc = &scratch[(size_t)k * k];
  double* sv = &scratch[2 * (size_t)k * k];

  /* Step 1: the table. Row i is a row of to, column j a row of from. */
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      m_buf[i * k + j] = dot_product(&to->v[i * m], &from->v[j * m], m);
    }
  }
  /* The bound translates with THIS, not with rot. It contracts, and for a
     lower bound a contraction is safe and in fact tighter: rot would add
     a second error term of 1 - cos(theta_max) on top. rot exists for the
     Lie layer, which needs a rotation. NULL when only rot is wanted. */
  if (map) {
    memcpy(map, m_buf, (size_t)k * (size_t)k * sizeof(double));
  }

  /* Step 2: the sweep. The only place in the code that asks for the
     accumulator, every other call passes NULL. */
  int sweeps = svd_values(m_buf, k, sv, jacc);
  if (sweeps < 0) return -2;

  /* Step 3: the largest angle, through the smallest cosine. Note this
     looks for a MINIMUM, the opposite of the loop at line 498. */
  double s_min = 1.0;
  for (int t = 0; t < k; t++) {
    double val = sv[t];
    if (val > 1.0) val = 1.0;
    if (val < 0.0) val = 0.0;
    if (val < s_min) {
      s_min = val;
    }
  }
  if (s_min < SPARTAN_EPS) return -3;
  *sin_theta = sqrt(1 - s_min * s_min);

  /* Step 4: the rotation, J transposed times Q.
     Q row t is m_buf row t divided by sv[t].
     Entry (i, j) is the sum over t of J[t][i] * Q[t][j].
     The transpose lives in that index order on jacc: writing
     jacc[i * k + t] instead gives a plausible matrix that is wrong. */
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      double acc = 0.0;
      for (int t = 0; t < k; t++) {
        acc += jacc[t * k + i] * (m_buf[t * k + j] / sv[t]);
      }
      rot[i * k + j] = acc;
    }
  }

  return 0;
}

/*
 * The sign of the determinant of the map from one basis to another.
 *
 * +1 the two frames turn the same way, -1 they do not, 0 the two subspaces
 * are degenerate against each other and the question has no answer.
 *
 * A sign needs no SVD. The map is the nearest rotation to the k by k
 * overlap table, and the two differ only by positive stretching factors,
 * which cannot change the sign of a determinant. So the sign of the
 * overlap table's own determinant is the whole answer.
 *
 * The sign is arbitrary rather than meaningful: a PCA direction and its
 * negative are equally valid, Gram-Schmidt picks one at random, and every
 * pick flips this. A reflection costs the bound nothing, since it
 * preserves lengths. It is fatal to the Lie layer, which needs a
 * logarithm, and a reflection has none.
 *
 * to, from   two orthonormal bases, same k and same m, neither modified
 * scratch    k*k doubles, overwritten
 *
 * Returns +1, -1 or 0. A bad argument also returns 0: the caller cannot
 * tell that from degenerate and does not need to, because both mean do
 * nothing.
 *
 * Edge cases:
 *   The singularity threshold is SPARTAN_EPS, the same one epoch_map uses,
 *   so the two agree on exactly which pairs are unusable.
 *   Fixing each incoming basis against the one it replaces holds the
 *   determinant at +1 for EVERY pair of epochs, not only neighbouring
 *   ones, because the sign is transitive while no pair has reached 90
 *   degrees. Past that it stops being transitive, which is the same
 *   boundary at which epoch_map refuses the pair.
 *   A per-basis rule does NOT work in place of this one: making the
 *   largest entry of each row positive leaves about half of all pairs
 *   reflected, which is what doing nothing leaves.
 */
int basis_det_sign(const OjaPCA* to, const OjaPCA* from, double* scratch) {
  if (!scratch || !to || !from || !to->v || !from->v) {
    return 0;
  }
  if (to->k <= 0 || to->m <= 0 || to->k != from->k || to->m != from->m) {
    return 0;
  }

  int k = to->k;
  int m = to->m;

  /* The overlap table, exactly as epoch_map builds it at line 568. */
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      scratch[i * k + j] = dot_product(&to->v[i * m], &from->v[j * m], m);
    }
  }

  /* Gaussian elimination with partial pivoting. The determinant itself is
     never formed, only its sign: each row swap flips it, each pivot
     contributes its own. Multiplying the pivots together would risk
     overflow for a magnitude that is thrown away anyway. */
  int sign = 1;
  for (int c = 0; c < k; c++) {
    /* The row at or below c with the largest magnitude in column c.
       Pivoting here is not about accuracy, it is about not dividing by
       something that is zero or nearly so. */
    int piv = c;
    for (int r = c + 1; r < k; r++) {
      if (fabs(scratch[r * k + c]) > fabs(scratch[piv * k + c])) piv = r;
    }

    if (fabs(scratch[piv * k + c]) < SPARTAN_EPS) return 0;

    if (piv != c) {
      for (int j = 0; j < k; j++) {
        double tmp = scratch[c * k + j];
        scratch[c * k + j] = scratch[piv * k + j];
        scratch[piv * k + j] = tmp;
      }
      sign = -sign;
    }

    if (scratch[c * k + c] < 0.0) sign = -sign;

    /* Clear below the pivot so the next column's pivot is honest. */
    for (int r = c + 1; r < k; r++) {
      double f = scratch[r * k + c] / scratch[c * k + c];
      for (int j = c; j < k; j++) {
        scratch[r * k + j] -= f * scratch[c * k + j];
      }
    }
  }

  return sign;
}
/*
 * Turns one word into the box it stands for: k intervals, one per
 * position, in the coordinates of the epoch that wrote it.
 *
 * Letter s at position j means the projection landed in cell s, so
 *
 *   lo = bkpt[s-1]   (or -X for the first cell)
 *   hi = bkpt[s]     (or +X for the last cell)
 *
 * matching digitize, which walks up while y is STRICTLY greater than a
 * cut point, so a value sitting exactly on one belongs to the lower cell.
 *
 * word       k letters, as written
 * bkpt       the epoch's cut points, k rows of MAX_ALPHABET
 * alphabet   k entries, the cell count per position
 * k, x_max   positions, and the clip described below [same units as y]
 * lo, hi     outputs, k doubles each
 *
 * Edge cases:
 *   The outer cells are unbounded, and a position with 0 bits is a single
 *   unbounded cell. Pushed through the map an unbounded interval makes
 *   EVERY output coordinate unbounded and the bound collapses to zero, so
 *   they are clipped to [-x_max, x_max]. The clip is not a heuristic: a
 *   basis row has length one, so a projection can never exceed the length
 *   of the window. Words are projections of the z-normed window itself,
 *   nothing subtracted, and each channel slice z-normed to population
 *   std has length sqrt(m_time) (less by the eps in the divisor, 0 for a
 *   constant slice), so the whole window has length at most sqrt(m).
 *   Pass x_max = sqrt(m).
 */
void word_box(const int* word, const double* bkpt, const int* alphabet, int k,
              double x_max, double* lo, double* hi) {
  for (int j = 0; j < k; j++) {
    int s = word[j];
    if (s == 0) {
      lo[j] = -x_max;
    } else {
      double v = bkpt[j * MAX_ALPHABET + s - 1];
      lo[j] = (v < -x_max) ? -x_max : v;
    }
    if (s == alphabet[j] - 1) {
      hi[j] = x_max;
    } else {
      double v = bkpt[j * MAX_ALPHABET + s];
      hi[j] = (v > x_max) ? x_max : v;
    }
  }
}
/*
 * Pushes a box through the epoch map and returns the smallest
 * axis-aligned box that contains the image.
 *
 * The image of a box under a linear map is a tilted box, and distances to
 * a tilted box are awkward. The smallest upright box around it is looser
 * but valid, and it is pure interval arithmetic: for output coordinate i,
 * each input interval contributes its low end where the coefficient is
 * positive and its high end where it is negative.
 *
 *   out_lo[i] = sum_j ( map[i][j] >= 0 ? map[i][j]*lo[j] : map[i][j]*hi[j] )
 *   out_hi[i] = sum_j ( map[i][j] >= 0 ? map[i][j]*hi[j] : map[i][j]*lo[j] )
 *
 * map        k*k, row major, from epoch_map; entry (i, j) at map[i*k + j]
 * lo, hi     the input box, k doubles each, in the FROM epoch
 * k          positions
 * out_lo, out_hi  outputs, k doubles each, in the TO epoch
 *
 * Edge cases:
 *   Every input must be finite, which word_box guarantees with its clip.
 *   An infinity here meets a coefficient that can be zero and the product
 *   is NaN, and every comparison with NaN is false, so the bound would
 *   silently stop pruning rather than fail.
 *   Aliasing: out_lo must not be lo and out_hi must not be hi. Each
 *   output coordinate reads every input coordinate, so writing in place
 *   feeds half-updated values back into the sum.
 */
void box_through(const double* map, const double* lo, const double* hi, int k,
                 double* out_lo, double* out_hi) {
  for (int i = 0; i < k; i++) {
    double acc_lo = 0.0;
    double acc_hi = 0.0;
    for (int j = 0; j < k; j++) {
      double c = map[i * k + j];
      if (c >= 0.0) {
        acc_lo += c * lo[j];
        acc_hi += c * hi[j];
      } else {
        acc_lo += c * hi[j];
        acc_hi += c * lo[j];
      }
    }
    out_lo[i] = acc_lo;
    out_hi[i] = acc_hi;
  }
}
/*
 * How much of the window the writing basis could not see.
 *
 * The window has length x_norm and the part the basis kept has length
 * |y|, and the two are perpendicular, so
 *
 *   r = sqrt( x_norm^2 - |y|^2 )          [same units as the window]
 *
 * This is what the slack is built from: sin(theta_max) * r.
 *
 * y        the k kept coordinates
 * k        positions
 * x_norm   length of the centred window the coordinates came from
 *
 * Returns r, never negative.
 *
 * Edge cases:
 *   Rounding can push the difference a hair below zero when the basis
 *   captured nearly everything; floor it at zero before the square root
 *   rather than after, or it returns NaN.
 *   For a live query x_norm is exact and free. For a stored word the
 *   window is gone, so the caller passes the same upper bound word_box
 *   clips with, sqrt(m), which keeps the result an over-estimate and
 *   therefore the bound valid.
 */
double residual_len(const double* y, int k, double x_norm) {
  double y_sq = dot_product(y, y, (size_t)k);
  double diff = x_norm * x_norm - y_sq;
  if (diff < 0.0) diff = 0.0;
  return sqrt(diff);
}
/*
 * The lower bound between a query box and a candidate word that were
 * written in DIFFERENT epochs.
 *
 * The query box has already been carried into the candidate's epoch by
 * box_through, so both boxes are in the same coordinates. Per coordinate
 * the two intervals either overlap, contributing nothing, or they do not,
 * contributing the gap between them. Then
 *
 *   dist = sqrt( sum_i gap_i^2 ) - sin_theta * r,   floored at 0
 *
 * q_lo, q_hi    the translated query box, k doubles each
 * c_lo, c_hi    the candidate's box, k doubles each
 * k             positions
 * sin_theta     from epoch_map, dimensionless
 * r             from residual_len, same units as the coordinates
 *
 * Returns a distance in the same units as the coordinates, never
 * negative.
 *
 * Edge cases:
 *   The slack comes off ONCE, after the square root, because that is the
 *   triangle inequality: the true position is within sin_theta * r of
 *   somewhere in the box. Subtracting it inside the sum would pay for it
 *   k times and loosen the bound for nothing.
 *   The floor at zero is not cosmetic. A negative lower bound is still
 *   valid but meaningless, and leaving it negative would let a sort put
 *   the least informative candidates first.
 *   A position with 0 bits contributes exactly zero and always has: its
 *   single cell is the whole line, so the two intervals always overlap.
 */
double cross_epoch_mindist(const double* q_lo, const double* q_hi,
                           const double* c_lo, const double* c_hi, int k,
                           double sin_theta, double r) {
  double sum = 0.0;
  for (int i = 0; i < k; i++) {
    double gap = 0.0;
    if (c_lo[i] - q_hi[i] > gap) gap = c_lo[i] - q_hi[i];
    if (q_lo[i] - c_hi[i] > gap) gap = q_lo[i] - c_hi[i];
    sum += gap * gap;
  }
  double dist = sqrt(sum) - sin_theta * r;
  return dist < 0.0 ? 0.0 : dist;
}
/* ---------------- where the subspace is going ---------------- */

/*
 * Moves a subspace along the motion that carried `from` onto `to`, scaled.
 *
 * The motion between two k-dimensional subspaces is k plane rotations
 * happening at once, and their angles are EXACTLY the principal angles, the
 * same numbers subspace_angles already reports. What this adds is the
 * planes: the directions the subspace is travelling in. The angles say how
 * far it has gone, the planes say where it is heading, and a prediction
 * needs both.
 *
 * Each plane is spanned by a principal direction a of `from` and the unit
 * vector w perpendicular to it inside the plane of its partner b in `to`:
 *
 *   w = (b - cos(theta) a) / sin(theta)
 *   advance(f) sends a to  cos(f*theta) a + sin(f*theta) w
 *
 * The k planes are mutually orthogonal, which is the whole reason this is a
 * formula and not an iteration: orthogonal rotations do not interfere, so
 * the result comes out orthonormal with no Gram-Schmidt.
 *
 * Measured: f = 0.5 lands at exactly half the angle, f = 2 at exactly
 * double, and f = 1 reproduces the span of `to` to 1e-16.
 *
 * from     where the motion started, k rows of m, orthonormal
 * to       where it has reached, same k and m, orthonormal
 * f        the scale [dimensionless]
 * out      the result, pre-allocated, same k and m, must not alias either input
 * scratch  2*k*k + k + 2*m doubles, overwritten
 *
 * Returns 0 on success, -1 on a bad argument, -2 if the sweep failed, -3 if
 * a principal pair is degenerate, where the motion has no defined direction.
 *
 * Edge cases:
 *   The rows of `out` are the MOVED PRINCIPAL DIRECTIONS, not moved copies
 *   of the rows of `from`. They span the right subspace, which is what
 *   matters, but they are in principal order and not in eigenvalue order.
 *   Whatever uses this must sort, which build_dictionary does anyway.
 *   A plane with zero angle has nowhere to move and its perpendicular is
 *   0/0. Those directions are copied through unmoved.
 *   The plane's ORIENTATION is the easy thing to get backwards. Reversed,
 *   the subspace moves AWAY from the data: the symptom is staleness getting
 *   worse instead of better, not a crash. Checked here by the fact that
 *   f = 1 must reproduce `to`.
 *   The result is orthonormal by construction, to machine precision.
 *   oja_orthonormality_error on it is the test that the construction holds,
 *   so do not straighten it first or the test proves nothing.
 */
int subspace_advance(const OjaPCA* from, const OjaPCA* to, double f,
                     OjaPCA* out, double* scratch) {
  if (!from || !to || !out || !scratch) {
    return -1;
  }
  if (!from->v || !to->v || !out->v) {
    return -1;
  }
  if (from->k <= 0 || from->m <= 0) {
    return -1;
  }
  if (from->k != to->k || from->m != to->m) {
    return -1;
  }
  if (out->k != from->k || out->m != from->m) {
    return -1;
  }
  if (out->v == from->v || out->v == to->v) {
    return -1;
  }

  int k = from->k;
  int m = from->m;

  double* m_buf = scratch;                     /* k*k, the overlap table */
  double* jacc = &scratch[(size_t)k * k];      /* k*k, accumulated rotations */
  double* sv = &scratch[2 * (size_t)k * k];    /* k, the cosines */
  double* a = &scratch[2 * (size_t)k * k + k]; /* m, a principal direction */
  double* b = &a[m];                           /* m, its partner */

  /* The overlap table, exactly as epoch_map builds it. */
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      m_buf[i * k + j] = dot_product(&from->v[i * m], &to->v[j * m], m);
    }
  }

  int sweeps = svd_values(m_buf, k, sv, jacc);
  if (sweeps < 0) {
    return -2;
  }

  /* After the sweep m_buf holds J*M, whose rows are orthogonal with lengths
     sv, and jacc holds J. So the principal directions are
         a_j = sum_t jacc[j*k + t] * row t of from
         b_j = sum_t (m_buf[j*k + t] / sv[j]) * row t of to
     Both come out unit length, and a_j is perpendicular to every b_i with
     i != j, which is what makes the planes orthogonal. */
  for (int j = 0; j < k; j++) {
    if (sv[j] < SPARTAN_EPS) {
      return -3;
    }
  }

  for (int j = 0; j < k; j++) {
    double cos_t = sv[j];
    if (cos_t > 1.0) cos_t = 1.0;
    if (cos_t < 0.0) cos_t = 0.0;

    for (int i = 0; i < m; i++) {
      a[i] = 0.0;
      b[i] = 0.0;
    }
    for (int t = 0; t < k; t++) {
      double ca = jacc[j * k + t];
      double cb = m_buf[j * k + t] / sv[j];
      const double* fr = &from->v[t * m];
      const double* tr = &to->v[t * m];
      for (int i = 0; i < m; i++) {
        a[i] += ca * fr[i];
        b[i] += cb * tr[i];
      }
    }

    double sin_t = sqrt(1.0 - cos_t * cos_t);
    double* dst = &out->v[j * m];

    /* Nothing to rotate toward, and w would be 0/0. */
    if (sin_t < SPARTAN_EPS) {
      for (int i = 0; i < m; i++) {
        dst[i] = a[i];
      }
      continue;
    }

    double theta = acos(cos_t);
    double ca = cos(f * theta);
    double cw = sin(f * theta);
    for (int i = 0; i < m; i++) {
      double w = (b[i] - cos_t * a[i]) / sin_t;
      dst[i] = ca * a[i] + cw * w;
    }
  }

  return 0;
}
/* ---------------- the epoch-0 basis ---------------- */

/*
 * The k leading directions of a block of windows, by orthogonal iteration:
 *
 *   V <- orth( (V X^T) X + mu V )      repeated until span(V) stops moving
 *
 * Power iteration on the covariance done k directions at once, without ever
 * forming the m x m covariance: a pass costs about 2*k*n*m multiply-adds and
 * k*n + k*m extra doubles, so it stays cheap when m is large (multivariate).
 * The span converges by a factor lambda_{k+1} / lambda_k per pass.
 *
 * X is used CENTRED by its own row mean c, as PCA is defined, without
 * writing a centred copy: with X~ = X - 1 c^T,
 *
 *   V X~^T = V X^T - (V c) 1^T       and       A X~ = A X - (A 1) c^T
 *
 * so a pass costs k*m more multiply-adds for V c, nothing per row. The
 * caller passes the windows exactly as the words see them (z-normed,
 * nothing subtracted) and gets the PCA of their spread around their mean.
 *
 * mu is a tiny shift, 1e-9 times the total energy of X~. It changes no
 * eigenvector, and it keeps a row from collapsing to zero when the windows
 * span fewer than k directions (a clean signal with too few parts): the
 * extra rows then settle on arbitrary directions instead of vanishing.
 *
 * X      n rows of m, the windows as written, row r at X[r*m]; read only
 * n      rows, must be at least k
 * p      in: the starting basis, k orthonormal rows (the warm-up Oja basis);
 *        out: the k leading directions, orthonormal
 *
 * Returns the passes taken, -1 on a bad argument or failed allocation, -2 if
 * the span had not settled after WARM_PCA_MAX_PASSES. On -2 the result is
 * still orthonormal and usable, only not fully converged.
 *
 * Edge cases:
 *   Rows come out in no particular order, and inside a pair of equal
 *   eigenvalues at any rotation within the pair. The caller sorts by variance
 *   (build_dictionary does); only the span matters.
 *   So convergence is tested on the SPAN, k - ||V_new V_old^T||_F^2 [no unit],
 *   which is 0 when the two spans are equal. A per-row test would never pass
 *   inside an equal pair, whose rows keep rotating and never need to stop.
 */
int warm_pca(const double* X, int n, int m, OjaPCA* p) {
  if (!X || !p || !p->v || m != p->m || n < p->k) return -1;
  int k = p->k;
  double* A = malloc((size_t)k * n * sizeof(double));   /* V X~^T, k x n */
  double* old = malloc((size_t)k * m * sizeof(double)); /* V before the pass */
  double* c = calloc((size_t)m, sizeof(double));        /* row mean of X */
  double* vc = malloc((size_t)k * sizeof(double));      /* V c, k */
  if (!A || !old || !c || !vc) {
    free(A);
    free(old);
    free(c);
    free(vc);
    return -1;
  }

  // c = (1/n) sum_r x_r
  for (int r = 0; r < n; r++)
    for (int i = 0; i < m; i++) c[i] += X[(size_t)r * m + i];
  for (int i = 0; i < m; i++) c[i] /= n;

  // energy = sum_r ||x_r - c||^2 [squared units of X]
  double energy = 0.0;
  for (int r = 0; r < n; r++)
    for (int i = 0; i < m; i++) {
      double d = X[(size_t)r * m + i] - c[i];
      energy += d * d;
    }
  double mu = 1e-9 * energy;

  int pass;
  for (pass = 1; pass <= WARM_PCA_MAX_PASSES; pass++) {
    memcpy(old, p->v, (size_t)k * m * sizeof(double));

    // A = V X~^T = V X^T - (V c) 1^T: each centred window along each row
    for (int a = 0; a < k; a++) {
      vc[a] = dot_product(&old[a * m], c, m);
      for (int r = 0; r < n; r++)
        A[a * n + r] = dot_product(&old[a * m], &X[(size_t)r * m], m) - vc[a];
    }

    // V = A X~ + mu V = A X - (A 1) c^T + mu V
    for (int a = 0; a < k; a++) {
      double* va = &p->v[a * m];
      double a_sum = 0.0; /* A 1, the row sum of A */
      for (int i = 0; i < m; i++) va[i] = mu * old[a * m + i];
      for (int r = 0; r < n; r++) {
        double w = A[a * n + r];
        const double* xr = &X[(size_t)r * m];
        a_sum += w;
        for (int i = 0; i < m; i++) va[i] += w * xr[i];
      }
      for (int i = 0; i < m; i++) va[i] -= a_sum * c[i];
    }
    gram_schmidt(p);

    // How much of the old span the new one still holds
    double kept = 0.0;
    for (int a = 0; a < k; a++)
      for (int b = 0; b < k; b++) {
        double d = dot_product(&p->v[a * m], &old[b * m], m);
        kept += d * d;
      }
    if (k - kept < WARM_PCA_TOL) break;
  }

  free(A);
  free(old);
  free(c);
  free(vc);
  return pass > WARM_PCA_MAX_PASSES ? -2 : pass;
}

/* ---------------- from numbers to letters ---------------- */

static int cmp_double(const void* a, const void* b) {
  double x = *(const double*)a;
  double y = *(const double*)b;
  if (x < y) return -1;
  if (x > y) return 1;
  return 0;
}

void project(const OjaPCA* p, const double* x, double* y) {
  for (int j = 0; j < p->k; j++) {
    y[j] = dot_product(x, &p->v[j * p->m], p->m);
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
      bkpt[MAX_ALPHABET * j + bp] = col[(
          int)bin_index];  // Stride rows by MAX_ALPHABET regardless of
                           // alphabet[j] so digitize can index rows uniformly
    }
    /* Last slot is a wall the search always stops at. */
    bkpt[MAX_ALPHABET * j + (alphabet[j] - 1)] = DBL_MAX;
  }
  free(col);
}

/* The PCA directions all have length one, so their own size tells us
   nothing about how important they are. The spread of the projected values
   does, and we already collect those during warm-up, so this costs nothing
   extra. */
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

/* The bid for the bit taking a position from x to x+1, from the closed form
 *   ev * dg(x),   dg(x) = g(x+1) - g(x) = 1 - lamda * (2 * (x - ref) + 1)
 * and NOT as the difference of two daa_score terms. The difference of two
 * scores carries rounding of ~1e-17, so a bid that is exactly 0 (importance
 * 0, or lamda = 1 at ref) came out as +-1e-17 and the ties were decided by
 * noise: measured, 3890 of 237833 random splits not non-increasing. The
 * closed form gives an exact 0 there. */
static double bit_bid(int x, double ev_value, int ref_bit, double lamda) {
  return ev_value * (1.0 - lamda * (2.0 * (x - ref_bit) + 1.0));
}

/*
 * Whether the greedy allocation is the proven optimum for these settings.
 *
 * Greedy is exact for a separable concave objective under one budget (the
 * appendix of the paper). The non-increasing order across positions is NOT
 * part of that structure; greedy keeps it by itself only when every bit of
 * the budget is bought at a non-negative marginal gain. The marginal gain of
 * the bit taking a position from x to x+1 is
 *
 *   ev_j * dg(x),   dg(x) = 1 - lamda * (2 * (x - ref) + 1)      [no unit]
 *
 * and ev_j >= 0, so it is non-negative exactly where dg(x) >= 0. Counting
 * those levels over every position, up to MAX_BITS, gives how many bits can
 * be bought that way; the budget must not exceed it.
 *
 *   lamda <= 1                    always holds (dg(ref) = 1 - lamda >= 0)
 *   lamda  > 1, k divides budget  holds (every position needs only ref)
 *   lamda  > 1 otherwise          fails: the leftover bits go to the LEAST
 *                                 important positions. Measured: on random
 *                                 sorted importances greedy then beats the
 *                                 non-increasing optimum in 2 of 3 cases
 *
 * k, total_bits, lamda, ref_bits as in daa_allocate.
 * Returns 1 if the condition holds, 0 if not.
 */
int daa_exact(int k, int total_bits, double lamda, const int* ref_bits) {
  if (k <= 0) return 0;
  int avg_bit = total_bits / k;
  long cap = 0; /* bits buyable at a non-negative marginal gain */
  for (int j = 0; j < k; j++) {
    int ref = ref_bits ? ref_bits[j] : avg_bit;
    for (int x = 0; x < MAX_BITS; x++)
      if (1.0 - lamda * (2.0 * (x - ref) + 1.0) >= 0.0) cap++;
  }
  return total_bits <= cap;
}

/*
 * The bit split, by the auction: the budget is sold one bit at a time to
 * the position that bids most for it, its importance times the gain of one
 * more bit,
 *
 *   bid_j = ev_j * dg(b_j),   dg(x) = 1 - lamda * (2 * (x - ref) + 1)
 *
 * Every bid falls as its owner accumulates (g is concave), so this is the
 * classical greedy allocation, exact for a separable concave objective
 * under one budget. Refused outside daa_exact, where it is not.
 *
 * Ties: equal bids happen exactly, at importance 0 (every bid is 0) and at
 * dg = 0 (lamda = 1 at ref). They go to the position holding FEWER bits,
 * then to the lower index. With importances in descending order this keeps
 * the split non-increasing; a plain lowest-index rule would pile the
 * leftover bits on the first zero-importance position, same score, wrong
 * order.
 *
 * Cost: total_bits passes over k positions, O(k * total_bits), no memory.
 *
 * ev          k importances, descending (build_dictionary sorts them)
 * k           positions
 * total_bits  the budget, spent exactly
 * lamda       pull toward ref [no unit]
 * ref_bits    k reference bits, or NULL for total_bits / k everywhere
 *             (whole division, as spartan.py does)
 * bits        output, k entries
 * score       output, daa_score of the result; may be NULL
 *
 * Returns 0, or -1 on a bad argument, a budget no split can spend
 * (above k * MAX_BITS), settings outside daa_exact, or importances not in
 * descending order.
 *
 * Edge cases:
 *   Descending order is a PRECONDITION, checked rather than assumed: it is
 *   what makes greedy keep the split non-increasing. On unsorted input
 *   greedy returns the true optimum for that order, which is not
 *   non-increasing, and the old dynamic program returned the best
 *   non-increasing split instead, a different answer. build_dictionary
 *   sorts first, so the pipeline never passes unsorted importances; equal
 *   neighbours are fine.
 */
int daa_allocate(const double* ev, int k, int total_bits, double lamda,
                 const int* ref_bits, int* bits, double* score) {
  if (!ev || !bits || k <= 0 || total_bits < 1 || total_bits > k * MAX_BITS ||
      !daa_exact(k, total_bits, lamda, ref_bits))
    return -1;
  for (int j = 1; j < k; j++)
    if (ev[j] > ev[j - 1])
      return -1;  // not descending, see the edge cases above
  int avg_bit = total_bits / k;

  for (int j = 0; j < k; j++) bits[j] = 0;

  for (int sold = 0; sold < total_bits; sold++) {
    int win = -1;
    double best = 0.0;
    for (int j = 0; j < k; j++) {
      if (bits[j] >= MAX_BITS) continue;
      int ref = ref_bits ? ref_bits[j] : avg_bit;
      double bid = bit_bid(bits[j], ev[j], ref, lamda);
      // higher bid wins; on an exact tie, fewer bits, then lower index
      // (j only grows, so a later j never wins a full tie)
      if (win < 0 || bid > best || (bid == best && bits[j] < bits[win])) {
        win = j;
        best = bid;
      }
    }
    bits[win]++;  // win >= 0: total_bits <= k * MAX_BITS leaves a free slot
  }

  if (score) *score = daa_score(ev, k, total_bits, lamda, ref_bits, bits);
  return 0;
}

double daa_gap(const double* ev, int k, int total_bits, double lamda,
               const int* ref_bits, const int* bits) {
  int avg_bit = total_bits / k;
  int min_bit = 0;
  int max_bit = total_bits - (k - 1) * min_bit;
  if (max_bit > MAX_BITS) max_bit = MAX_BITS;
  double best_bid = -DBL_MAX;
  double best_ask = DBL_MAX;
  for (int j = 0; j < k; j++) {
    // Per position, inside the loop. NULL falls back to the even split, which
    // is exactly what daa_allocate does
    int ref = ref_bits ? ref_bits[j] : avg_bit;
    if (bits[j] < max_bit) {
      double up = bit_bid(bits[j], ev[j], ref, lamda);
      if (up > best_bid) best_bid = up;
    }
    if (bits[j] > min_bit) {
      double down = bit_bid(bits[j] - 1, ev[j], ref, lamda);

      if (down < best_ask) best_ask = down;
    }
  }

  if (best_bid == -DBL_MAX || best_ask == DBL_MAX) {
    return 0.0;
  }

  // Above SPARTAN_EPS, not above 0. The bids come from the closed form, so
  // a tie is an exact 0 here; the margin is a second guard, for importances
  // that differ only by rounding. Before both, a gap of ~1e-17 at lamda = 1
  // opened type 1 epochs that re-solved to the same split (measured: 4 on
  // tsb_drift, total_bits 17)
  double gap = best_bid - best_ask;
  if (gap <= SPARTAN_EPS) {
    return 0;
  }
  return gap;
}
/* ---------------- settings ---------------- */

Config config_default(int k, int m) { return config_default_mv(k, 1, m); }

/* The defaults. This is the ONLY place a default exists: the settings file
 * overrides any of them, and the tools read what a run used from the
 * settings.cfg the C writes, never from a copy of these.
 *
 *   ring_w = 512   a cut point is a quantile, and 10 points per cell of the
 *                  largest alphabet is 10 * MAX_ALPHABET = 160; 512 is the
 *                  next power of two past three times that
 *   warmup = ring_w  the epoch-0 basis is exact PCA on the warm-up windows
 *                  (warm_pca), so the warm-up needs only enough rows for the
 *                  cut points, the same number a re-cut uses. Measured on the
 *                  three tsb streams: 256 already gave no false epochs
 *   cross_epoch = search   tight at every angle; on tsb_drift_strong TLB
 *                  0.87 of batch all against 0.55 for box with stored r
 */
Config config_default_mv(int k, int d, int m_time) {
  Config c = {
      .k = k,
      .d = d,
      .m_time = m_time,
      .m = d * m_time,
      .stride = m_time,
      .step = 0.24,
      .beta = 0.1,
      .gs_period = 5,
      .total_bits = 2 * k,
      .lambda = 0.5,
      .theta_threshold_deg = 45.0,
  };
  c.ring_w = 512;
  c.warmup = c.ring_w;
  c.angle_period = c.gs_period;
  c.lr = c.step / c.m;
  c.theta_threshold = c.theta_threshold_deg * SPARTAN_DEG_TO_RAD;
  c.freeze_ahead = 0.0;
  c.cross_epoch = CROSS_EPOCH_SEARCH;
  c.level_letters = 0;
  return c;
}
