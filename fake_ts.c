#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define ROWS 20
#define COLS 8
#define K 3

// Helper function i will use loop unrolling just to have it ready for rwds
double dot_product(const double *v1, const double *v2, size_t n) {
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

// z-normalization
void znorm_row(double *x, int n) {
  double sum = 0.0;
  double variance = 0.0;
  double mean, std_dev;

  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  mean = sum / n;

  for (int i = 0; i < n; i++) {
    variance += pow(x[i] - mean, 2);
  }

  std_dev = sqrt(variance / n) + 1e-12;

  // reverse division opt
  double inv_std_dev = 1.0 / std_dev;
  for (int i = 0; i < n; i++) {
    x[i] = (x[i] - mean) * inv_std_dev;
  }
}

typedef struct {
  double mean[COLS];
  long n;
} RunningMean;

// running mean
void rmean_update(RunningMean *rm, const double *x) {
  rm->n++;
  for (int i = 0; i < COLS; i++)
    rm->mean[i] += (x[i] - rm->mean[i]) / rm->n;
}

typedef struct {
  double v[3][COLS];
} Oja3;

void oja3_update(Oja3 *p, const double *xc, double lr) {
  double u[COLS];
  for (int i = 0; i < COLS; i++)
    u[i] = xc[i];

  for (int j = 0; j < K; j++) {
    double y = dot_product(u, p->v[j], COLS);

    // Oja rule
    for (int i = 0; i < COLS; i++) {
      p->v[j][i] += lr * y * (u[i] - y * p->v[j][i]);
    }

    double d = dot_product(u, p->v[j], COLS);

    // Sanger deflation
    for (int i = 0; i < COLS; i++) {
      u[i] -= d * p->v[j][i];
    }
  }
}

void gram_schmidt(Oja3 *p) {
  double nrm = sqrt(dot_product(p->v[0], p->v[0], COLS));
  if (nrm < 1e-12)
    return;
  for (int i = 0; i < COLS; i++) {
    p->v[0][i] = p->v[0][i] / nrm;
  }

  double d01 = dot_product(p->v[1], p->v[0], COLS);
  for (int i = 0; i < COLS; i++) {
    p->v[1][i] -= d01 * p->v[0][i];
  }
  double nrm1 = sqrt(dot_product(p->v[1], p->v[1], COLS));
  if (nrm1 < 1e-12)
    return;
  for (int i = 0; i < COLS; i++) {
    p->v[1][i] = p->v[1][i] / nrm1;
  }
  double d02 = dot_product(p->v[2], p->v[0], COLS);
  for (int i = 0; i < COLS; i++) {
    p->v[2][i] -= d02 * p->v[0][i];
  }
  double d12 = dot_product(p->v[2], p->v[1], COLS);
  for (int i = 0; i < COLS; i++) {
    p->v[2][i] -= d12 * p->v[1][i];
  }
  double nrm2 = sqrt(dot_product(p->v[2], p->v[2], COLS));
  if (nrm2 < 1e-12)
    return;
  for (int i = 0; i < COLS; i++) {
    p->v[2][i] = p->v[2][i] / nrm2;
  }
}

int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a; // diref as double
  double y = *(const double *)b;
  if (x < y)
    return -1;
  if (x > y)
    return 1;
  return 0;
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

  RunningMean my_stats = {{0.0}, 0};

  for (int i = 0; i < ROWS; i++) {
    znorm_row(data[i], COLS);
    rmean_update(&my_stats, data[i]);
  }

  Oja3 oja = {.v[0] = {1, 0, 0, 0, 0, 0, 0, 0},
              .v[1] = {0, 1, 0, 0, 0, 0, 0, 0},
              .v[2] = {0, 0, 1, 0, 0, 0, 0, 0}};
  double xc[COLS];
  for (int epoch = 0; epoch < 200; epoch++) {
    for (int r = 0; r < ROWS; r++) {
      for (int i = 0; i < COLS; i++)
        xc[i] = data[r][i] - my_stats.mean[i];
      oja3_update(&oja, xc, 0.01);
    }
    gram_schmidt(&oja);
  }
  double proj[ROWS][K];

  for (int r = 0; r < ROWS; r++) {
    for (int i = 0; i < COLS; i++) {
      xc[i] = data[r][i] - my_stats.mean[i];
    }
    proj[r][0] = dot_product(xc, oja.v[0], COLS);
    proj[r][1] = dot_product(xc, oja.v[1], COLS);
    proj[r][2] = dot_product(xc, oja.v[2], COLS);

    printf("%2d %7.3f %7.3f %7.3f\n", r, proj[r][0], proj[r][1], proj[r][2]);
  }

  double col[ROWS];
  double target_depth = (double)ROWS / 4;
  double bkpt[K][4];
  for (int j = 0; j < K; j++) {
    for (int r = 0; r < ROWS; r++)
      col[r] = proj[r][j];
    qsort(col, ROWS, sizeof(double), cmp_double);
    double bin_index = 0.0;
    for (int bp = 0; bp < 3; bp++) {
      bin_index += target_depth;
      bkpt[j][bp] = col[(int)bin_index];
      printf("β%d=%9.6f\n", bp, bkpt[j][bp]);
    }
    bkpt[j][3] = DBL_MAX;
  }

  printf("\nWords:\n");
  for (int r = 0; r < ROWS; r++) {
    printf("%2d  ", r);
    for (int j = 0; j < K; j++) {
      int s = 0;
      while (proj[r][j] > bkpt[j][s])
        s++;
      printf("%c", 'a' + s); // 97-100 ASCII
    }
    printf("\n");
  }

  // v Magnitudes
  printf("Vector Magnitudes\n");
  for (int j = 0; j < K; j++) {
    double nrm_sq = dot_product(oja.v[j], oja.v[j], COLS);
    printf("||v%d|| = %.4f\n", j, sqrt(nrm_sq));
  }

  // Orthogonality check
  printf("\n Pairwise Orthogonality Check (Dot Products)\n");
  for (int j = 0; j < K; j++) {
    for (int b = j + 1; b < K; b++) {
      double dot_prod = dot_product(oja.v[j], oja.v[b], COLS);
      printf("v%d . v%d = %11.4e\n", j, b, dot_prod);
    }
  }
  printf("\n");

  printf("Components Matrix:\n");
  for (int j = 0; j < K; j++) {
    for (int i = 0; i < COLS; i++)
      printf("%7.3f ", oja.v[j][i]);
    printf("\n");
  }

  printf("\nNumber of samples processed: %ld\n", my_stats.n);
  return 0;
}
