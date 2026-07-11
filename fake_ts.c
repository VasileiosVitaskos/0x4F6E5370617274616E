#include <math.h>
#include <stdio.h>

#define ROWS 20
#define COLS 8

// in main i have to call znorm_row in a loop
void znorm_row(double *x, int n) {
  double sum = 0.0;
  double variance = 0.0;
  double mean, std_dev;

  // 1. mean
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  mean = sum / n;

  // 2. variance & std_dev
  for (int i = 0; i < n; i++) {
    variance += pow(x[i] - mean, 2);
  }

  std_dev = sqrt(variance / n) + 1e-12;

  // 3. apply
  for (int i = 0; i < n; i++) {
    x[i] = (x[i] - mean) / std_dev;
  }
}

typedef struct {
  double mean[8];
  long n;
} RunningMean;

void rmean_update(RunningMean *rm, const double *x) {
  rm->n++; // this represents the row not a single value in main i will have to
           // call it in a loop
  for (int i = 0; i < 8; i++)
    rm->mean[i] +=
        (x[i] - rm->mean[i]) /
        rm->n; // new val - prev mean / n and then add that to update prev mean
}

typedef struct {
  double v[COLS];
} Oja1;

void oja1_update(Oja1 *p, const double *xc, double lr) {
  double y = 0.0;
  for (int i = 0; i < COLS; i++) {
    y += xc[i] * p->v[i];
  }
  for (int i = 0; i < COLS; i++) {
    p->v[i] += lr * y * (xc[i] - y * p->v[i]);
  }
}
int main() {
  /* 0-6: ανοδικές, 7-13: καθοδικές, 14-19: V-shape */
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

  // It is crucial to set 'n' to 0 and optionally zero out the means
  RunningMean my_stats = {{0.0}, 0};

  // for every row
  for (int i = 0; i < ROWS; i++) {
    // data[i] is like (double*) pointing at line start(or after arrays
    // metadata)
    znorm_row(data[i], COLS);
    rmean_update(&my_stats, data[i]);
  }

  // 4. Print the resulting means
  printf("Number of samples processed: %ld\n", my_stats.n);
  printf("Updated means:\n");
  for (int i = 0; i < 8; i++) {
    printf("  Dimension %d: %.4f\n", i, my_stats.mean[i]);
  }

  printf("\nafter znorm_row (per line):\n");
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      printf("%7.2f ", data[i][j]);
    }
    printf("\n");
  }

  return 0;
}
