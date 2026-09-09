#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 20
#define COLS 8
#define K 3

// Helper function i will use loop unrolling just to have it ready for rwds
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

  // in case reminder
  for (; i < n; i++) {
    t_sum += v1[i] * v2[i];
  }

  return t_sum;
}

// z-normalization: one vector, one length
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

  std_dev = sqrt(variance / m) + 1e-12;

  // reverse division opt
  double inv_std_dev = 1.0 / std_dev;
  for (int i = 0; i < m; i++) {
    x[i] = (x[i] - mean) * inv_std_dev;
  }
}

/* ---------------- RunningMean: heap-allocated ---------------- */

typedef struct {
  double* mean; /* m doubles */
  long n;
  int m;
} RunningMean;

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

// running mean
void rmean_update(RunningMean* rm, const double* x) {
  int m = rm->m;
  rm->n++;
  for (int i = 0; i < m; i++) rm->mean[i] += (x[i] - rm->mean[i]) / rm->n;
}

/* ---------------- OjaPCA: heap-allocated ---------------- */

typedef struct {
  double* v; /* k*m doubles: the basis, row j starts at v[j*m] */
  double* u; /* m doubles: scratch buffer for the deflation   */
  int k, m;
} OjaPCA;

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

  /* seeds e1, e2, ... : one on the diagonal of each row */
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

    // Oja rule
    for (int i = 0; i < m; i++) {
      vj[i] += lr * y * (u[i] - y * vj[i]);
    }

    double d = dot_product(u, vj, m);

    // Sanger deflation
    for (int i = 0; i < m; i++) {
      u[i] -= d * vj[i];
    }
  }
}

void gram_schmidt(OjaPCA* p) {
  int k = p->k, m = p->m;

  // 1. Εξωτερικό loop: Διατρέχει κάθε διάνυσμα που θέλουμε να
  // ορθοκανονικοποιήσουμε
  for (int j = 0; j < k; j++) {
    double* vj = &p->v[j * m];

    // 2. Εσωτερικό loop: "Καθαρίζει" το τρέχον v[j] από ΟΛΑ τα προηγούμενα
    // v[p_idx]
    for (int p_idx = 0; p_idx < j; p_idx++) {
      double* vp = &p->v[p_idx * m];
      // Υπολογισμός της προβολής (dot) - ΠΡΕΠΕΙ να είναι μέσα στο loop
      double d = dot_product(vj, vp, m);

      // Αφαίρεση της προβολής
      for (int i = 0; i < m; i++) {
        vj[i] -= d * vp[i];
      }
    }

    // 3. Κανονικοποίηση (Μέτρο, Guard, Διαίρεση)
    double nrm = sqrt(dot_product(vj, vj, m));

    // Το return έγινε continue: αν μηδενιστεί, προχωράμε στο επόμενο j
    if (nrm < 1e-12) {
      continue;
    }

    for (int i = 0; i < m; i++) {
      vj[i] = vj[i] / nrm;
    }
  }
}

int cmp_double(const void* a, const void* b) {
  double x = *(const double*)a;  // diref as double
  double y = *(const double*)b;
  if (x < y) return -1;
  if (x > y) return 1;
  return 0;
}

void compute_breakpoints(const double* proj, int n, int k, int alphabet,
                         double* bkpt) {
  double* col = calloc((size_t)n, sizeof(double));
  if (!col) return;
  double target_depth = (double)n / alphabet;
  for (int j = 0; j < k; j++) {
    for (int r = 0; r < n; r++) col[r] = proj[r * k + j];
    qsort(col, n, sizeof(double), cmp_double);
    double bin_index = 0.0;
    for (int bp = 0; bp < alphabet - 1; bp++) {
      bin_index += target_depth;
      bkpt[j * alphabet + bp] = col[(int)bin_index];
      printf("β%d=%9.6f\n", bp, bkpt[j * alphabet + bp]);
    }
    bkpt[j * alphabet + (alphabet - 1)] = DBL_MAX;
  }
  free(col);
}

void digitize(const double* y, const double* bkpt, int k, int alphabet,
              int* word) {
  for (int j = 0; j < k; j++) {
    int s = 0;
    while (y[j] > bkpt[j * alphabet + s]) s++;
    word[j] = s;
  }
}

int main() {
  double data[ROWS][COLS] = {{1.0, 2.1, 2.9, 4.2, 5.0, 6.1, 6.9, 8.0},
                             {0.8, 1.9, 3.1, 3.9, 5.2, 5.9, 7.1, 7.9},
                             {1.2, 2.0, 3.2, 4.1, 4.8, 6.2, 7.0, 8.1},
                             {0.9, 2.2, 2.8, 4.0, 5.1, 5.8, 7.2, 8.2},
                             {1.1, 1.8, 3.0, 4.3, 4.9, 6.0, 7.1, 7.8},
                             {1.0, 2.0, 3.1, 3.8, 5.0, 6.2, 6.8, 8.0},
                             {0.7, 2.1, 3.0, 4.1, 5.2, 6.1, 7.0, 8.3},
                             {8.0, 6.9, 6.1, 5.0, 4.2, 2.9, 2.1, 1.0},
                             {7.9, 7.1, 5.9, 5.2, 3.9, 3.1, 1.9, 0.8},
                             {8.1, 7.0, 6.2, 4.8, 4.1, 3.2, 2.0, 1.2},
                             {8.2, 7.2, 5.8, 5.1, 4.0, 2.8, 2.2, 0.9},
                             {7.8, 7.1, 6.0, 4.9, 4.3, 3.0, 1.8, 1.1},
                             {8.0, 6.8, 6.2, 5.0, 3.8, 3.1, 2.0, 1.0},
                             {8.3, 7.0, 6.1, 5.2, 4.1, 3.0, 2.1, 0.7},
                             {8.0, 6.0, 4.1, 2.0, 2.1, 3.9, 6.1, 7.9},
                             {7.8, 5.9, 3.8, 1.9, 2.2, 4.1, 5.8, 8.1},
                             {8.2, 6.1, 4.0, 2.2, 1.8, 4.0, 6.2, 7.8},
                             {7.9, 6.2, 3.9, 1.8, 2.0, 4.2, 6.0, 8.0},
                             {8.1, 5.8, 4.2, 2.1, 1.9, 3.8, 5.9, 8.2},
                             {8.0, 6.0, 4.0, 2.0, 2.0, 4.0, 6.0, 8.0}};

  RunningMean my_stats;
  if (rmean_init(&my_stats, COLS) != 0) {
    fprintf(stderr, "rmean_init failed\n");
    return 1;
  }

  for (int i = 0; i < ROWS; i++) {
    znorm_row(data[i], COLS);
    rmean_update(&my_stats, data[i]);
  }

  OjaPCA oja;
  if (oja_init(&oja, K, COLS) != 0) {
    fprintf(stderr, "oja_init failed\n");
    rmean_free(&my_stats);
    return 1;
  }

  double xc[COLS];
  for (int epoch = 0; epoch < 200; epoch++) {
    for (int r = 0; r < ROWS; r++) {
      for (int i = 0; i < COLS; i++) xc[i] = data[r][i] - my_stats.mean[i];
      oja_update(&oja, xc, 0.01);
    }
    gram_schmidt(&oja);
  }

  double* proj = calloc((size_t)ROWS * K, sizeof(double));

  for (int r = 0; r < ROWS; r++) {
    for (int i = 0; i < COLS; i++) {
      xc[i] = data[r][i] - my_stats.mean[i];
    }
    proj[r * K + 0] = dot_product(xc, &oja.v[0 * COLS], COLS);
    proj[r * K + 1] = dot_product(xc, &oja.v[1 * COLS], COLS);
    proj[r * K + 2] = dot_product(xc, &oja.v[2 * COLS], COLS);

    printf("%2d %7.3f %7.3f %7.3f\n", r, proj[r * K + 0], proj[r * K + 1],
           proj[r * K + 2]);
  }

  double bkpt[K * 4];
  compute_breakpoints(proj, ROWS, K, 4, bkpt);

  printf("\nWords:\n");
  for (int r = 0; r < ROWS; r++) {
    printf("%2d  ", r);
    int word[K];
    digitize(&proj[r * K], bkpt, K, 4, word);
    for (int j = 0; j < K; j++) {
      printf("%c", 'a' + word[j]);  // 97-100 ASCII
    }
    printf("\n");
  }

  // v Magnitudes
  printf("Vector Magnitudes\n");
  for (int j = 0; j < K; j++) {
    double nrm_sq = dot_product(&oja.v[j * COLS], &oja.v[j * COLS], COLS);
    printf("||v%d|| = %.4f\n", j, sqrt(nrm_sq));
  }

  // Orthogonality check
  printf("\n Pairwise Orthogonality Check (Dot Products)\n");
  for (int j = 0; j < K; j++) {
    for (int b = j + 1; b < K; b++) {
      double dot_prod = dot_product(&oja.v[j * COLS], &oja.v[b * COLS], COLS);
      printf("v%d . v%d = %11.4e\n", j, b, dot_prod);
    }
  }
  printf("\n");

  printf("Components Matrix:\n");
  for (int j = 0; j < K; j++) {
    for (int i = 0; i < COLS; i++) printf("%7.3f ", oja.v[j * COLS + i]);
    printf("\n");
  }

  printf("\nNumber of samples processed: %ld\n", my_stats.n);

  oja_free(&oja);
  rmean_free(&my_stats);
  free(proj);
  return 0;
}
