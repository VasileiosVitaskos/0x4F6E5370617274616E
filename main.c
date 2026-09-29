#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spartan.h"
#include "stream.h"

// Fixed dimensions for the demo synthetic stream
#define ROWS 20
#define COLS 8
#define K 3
#define TOTAL_ELEMENTS (ROWS * COLS)
#define NUM_REPEATS 10
#define STREAM_LEN (TOTAL_ELEMENTS * NUM_REPEATS)

int main(void) {
  // 20 short multivariate/univariate series used as source patterns
  const double data[ROWS][COLS] = {{1.0, 2.1, 2.9, 4.2, 5.0, 6.1, 6.9, 8.0},
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

  // Flatten source data into a single continuous stream
  double flat_data[STREAM_LEN];
  const size_t block_bytes = TOTAL_ELEMENTS * sizeof(double);
  for (int rep = 0; rep < NUM_REPEATS; rep++) {
    memcpy(flat_data + (rep * TOTAL_ELEMENTS), data, block_bytes);
  }

  // Algorithm configuration: k directions, d=1 channel, m_time=COLS frame size
  Config cfg = config_default_mv(K, 1, COLS);

  // Default exit code and safe initialization of pointers and state
  int rc = 1;
  double* win = NULL;
  double* xc = NULL;
  double* y = NULL;
  int* word = NULL;
  double* bkpt = NULL;
  double* warmup_proj = NULL;
  double* ev = NULL;
  int* bits = NULL;
  int* alphabet_arr = NULL;

  RunningMean my_stats = {0};
  OjaPCA oja = {0};
  WindowStream stream = {0};

  // Allocate algorithm buffers based on runtime configuration
  win = calloc((size_t)cfg.m, sizeof(double));
  xc = calloc((size_t)cfg.m, sizeof(double));
  y = calloc((size_t)cfg.k, sizeof(double));
  word = calloc((size_t)cfg.k, sizeof(int));
  bkpt = calloc((size_t)cfg.k * MAX_ALPHABET, sizeof(double));
  warmup_proj = calloc((size_t)cfg.warmup * cfg.k, sizeof(double));
  ev = calloc((size_t)cfg.k, sizeof(double));
  bits = calloc((size_t)cfg.k, sizeof(int));
  alphabet_arr = calloc((size_t)cfg.k, sizeof(int));

  if (!win || !xc || !y || !word || !bkpt || !warmup_proj || !ev || !bits ||
      !alphabet_arr) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    goto cleanup;
  }

  // Initialize computational components
  if (rmean_init(&my_stats, cfg.m) != 0) {
    fprintf(stderr, "Error: rmean_init failed\n");
    goto cleanup;
  }

  if (oja_init(&oja, cfg.k, cfg.m) != 0) {
    fprintf(stderr, "Error: oja_init failed\n");
    goto cleanup;
  }

  if (stream_init_array(&stream, flat_data, STREAM_LEN, cfg.d, cfg.m_time,
                        cfg.stride) != 0) {
    fprintf(stderr, "Error: stream_init_array failed\n");
    goto cleanup;
  }

  int warm_count = 0;
  int have_bkpt = 0;
  long n_seen = 0;
  double worst_dot_seen = 0.0;
  double worst_len_seen = 0.0;

  // Main stream processing loop
  while (stream_next(&stream, win) == 1) {
    n_seen++;

    // Channel-wise z-score normalization
    for (int c = 0; c < cfg.d; c++) {
      znorm_row(&win[c * cfg.m_time], cfg.m_time);
    }

    // Center window by removing the running streaming mean
    rmean_update(&my_stats, win);
    for (int i = 0; i < cfg.m; i++) {
      xc[i] = win[i] - my_stats.mean[i];
    }

    // Online PCA update via Oja's rule
    oja_update(&oja, xc, cfg.lr);

    // Periodically monitor subspace drift and restore orthonormality
    if (n_seen % cfg.gs_period == 0) {
      double d_err, l_err;
      oja_orthonormality_error(&oja, &d_err, &l_err);
      if (d_err > worst_dot_seen) worst_dot_seen = d_err;
      if (l_err > worst_len_seen) worst_len_seen = l_err;

      gram_schmidt(&oja);
    }

    // Project centered window onto the learned PCA subspace
    project(&oja, xc, y);

    // Warmup phase: collect projections to establish DAA bit allocation and
    // breakpoints
    if (!have_bkpt) {
      for (int j = 0; j < cfg.k; j++) {
        warmup_proj[warm_count * cfg.k + j] = y[j];
      }
      warm_count++;

      if (warm_count == cfg.warmup) {
        estimate_eigenvalues(warmup_proj, cfg.warmup, cfg.k, ev);

        double score = 0.0;
        if (daa_allocate(ev, cfg.k, cfg.total_bits, cfg.lambda, NULL, bits,
                         &score) != 0) {
          fprintf(stderr, "Error: daa_allocate failed\n");
          goto cleanup;
        }

        // Compute alphabet sizes from allocated bit budget
        for (int j = 0; j < cfg.k; j++) {
          alphabet_arr[j] = 1 << bits[j];
          if (alphabet_arr[j] > MAX_ALPHABET) {
            bits[j] = 31 - __builtin_clz(MAX_ALPHABET);
            fprintf(stderr, "Warning: position %d clamped to max alphabet %d\n",
                    j, MAX_ALPHABET);
            alphabet_arr[j] = MAX_ALPHABET;
          }
        }

        compute_breakpoints(warmup_proj, cfg.warmup, cfg.k, alphabet_arr, bkpt);

        // Print warmup configuration diagnostics
        fprintf(stderr, "ev       =");
        for (int j = 0; j < cfg.k; j++) fprintf(stderr, " %.4f", ev[j]);
        fprintf(stderr, "\nbits     =");
        for (int j = 0; j < cfg.k; j++) fprintf(stderr, " %d", bits[j]);
        fprintf(stderr, "\nalphabet =");
        for (int j = 0; j < cfg.k; j++) fprintf(stderr, " %d", alphabet_arr[j]);
        fprintf(stderr, "\nscore    = %.6f\n", score);

        have_bkpt = 1;
        free(warmup_proj);
        warmup_proj = NULL;
      }
      continue;
    }

    // Quantization: convert subspace coordinates into discrete symbols
    digitize(y, bkpt, cfg.k, alphabet_arr, word);

    // Emit symbolic representation for the current window
    printf("%2ld  ", n_seen - 1);
    for (int j = 0; j < cfg.k; j++) {
      printf("%c", 'a' + word[j]);
    }
    printf("\n");
  }

  fprintf(stderr, "worst drift before cleanup: overlap %.3e, length %.3e\n",
          worst_dot_seen, worst_len_seen);

  rc = 0;

cleanup:
  free(win);
  free(xc);
  free(y);
  free(word);
  free(bkpt);
  free(warmup_proj);
  free(ev);
  free(bits);
  free(alphabet_arr);
  oja_free(&oja);
  rmean_free(&my_stats);
  stream_free(&stream);
  return rc;
}
