/* Demo driver. Twenty short series fed in one at a time, as if they arrived
 * from a stream. Nothing is read from disk yet.
 *
 * Names used below:
 *   win / xc     = the current window, before and after the average is removed
 *   y            = where that window lands on each kept direction
 *   word         = the letters produced for that window
 *   bkpt         = the cut points, one row per letter position
 *   ev           = how important each direction turned out to be
 *   bits         = how many bits each letter position gets
 */

#include "spartan.h"

#include <stdio.h>
#include <stdlib.h>

#define ROWS 20
#define COLS 8
#define K 3

int main(void) {
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

  Config cfg = config_default(K, COLS);

  /* One way out of this function. Every pointer starts empty so the tidy-up
     at the bottom is correct no matter how far we got. */
  int rc = 1;
  double *xc = NULL;
  double *y = NULL;
  int *word = NULL;
  double *bkpt = NULL;
  double *warmup_proj = NULL;
  double *ev = NULL;
  int *bits = NULL;
  int *alphabet_arr = NULL;

  RunningMean my_stats;
  if (rmean_init(&my_stats, cfg.m) != 0) {
    fprintf(stderr, "rmean_init failed\n");
    return 1;
  }

  OjaPCA oja;
  if (oja_init(&oja, cfg.k, cfg.m) != 0) {
    fprintf(stderr, "oja_init failed\n");
    rmean_free(&my_stats);
    return 1;
  }

  xc = calloc((size_t)cfg.m, sizeof(double));
  y = calloc((size_t)cfg.k, sizeof(double));
  word = calloc((size_t)cfg.k, sizeof(int));
  bkpt = calloc((size_t)cfg.k * MAX_ALPHABET, sizeof(double));
  warmup_proj = calloc((size_t)cfg.warmup * cfg.k, sizeof(double));
  ev = calloc((size_t)cfg.k, sizeof(double));
  bits = calloc((size_t)cfg.k, sizeof(int));
  alphabet_arr = calloc((size_t)cfg.k, sizeof(int));
  if (!xc || !y || !word || !bkpt || !warmup_proj || !ev || !bits ||
      !alphabet_arr) {
    fprintf(stderr, "allocation failed\n");
    goto cleanup;
  }

  int warm_count = 0;
  int have_bkpt = 0;
  long n_seen = 0;

  /* Worst drift seen across the whole run, measured just before each cleanup.
   */
  double worst_dot_seen = 0.0;
  double worst_len_seen = 0.0;

  /* Ten passes over twenty rows, because twenty is not enough for the PCA
     directions to settle. This is a stand-in for a real stream and it breaks
     the one-pass rule, so it goes away once the stream layer lands. */
  for (int tt = 0; tt < 10; tt++) {
    for (int r = 0; r < ROWS; r++) {
      n_seen++;

      znorm_row(data[r], cfg.m);
      rmean_update(&my_stats, data[r]);

      for (int i = 0; i < cfg.m; i++) {
        xc[i] = data[r][i] - my_stats.mean[i];
      }

      oja_update(&oja, xc, cfg.lr);

      if (n_seen % cfg.gs_period == 0) {
        /* Measure the drift at its worst, just before it is cleaned up.
           This number is what the distance guarantee has to live with. */
        double d, l;
        oja_orthonormality_error(&oja, &d, &l);
        if (d > worst_dot_seen)
          worst_dot_seen = d;
        if (l > worst_len_seen)
          worst_len_seen = l;

        gram_schmidt(&oja);
      }

      project(&oja, xc, y);

      if (!have_bkpt) {
        for (int j = 0; j < cfg.k; j++) {
          warmup_proj[warm_count * cfg.k + j] = y[j];
        }
        warm_count++;

        if (warm_count == cfg.warmup) {

          /* End of warm-up: this is the one moment the bit split is decided.
             No reference split exists yet, so pass none and match spartan.py.
             From the second epoch on, the split just made is handed in here
             so the new one prefers to stay put. */
          estimate_eigenvalues(warmup_proj, cfg.warmup, cfg.k, ev);

          double score = 0.0;
          if (daa_allocate(ev, cfg.k, cfg.total_bits, cfg.lambda, NULL, bits,
                           &score) != 0) {
            fprintf(stderr, "daa_allocate failed\n");
            goto cleanup;
          }

          for (int j = 0; j < cfg.k; j++) {
            alphabet_arr[j] = 1 << bits[j];
            if (alphabet_arr[j] > MAX_ALPHABET) {
              bits[j] = 31 - __builtin_clz(MAX_ALPHABET);
              fprintf(
                  stderr,
                  "position %d wants alphabet %d, table holds %d - clamped\n",
                  j, alphabet_arr[j], MAX_ALPHABET);
              alphabet_arr[j] = MAX_ALPHABET;
            }
          }
          compute_breakpoints(warmup_proj, cfg.warmup, cfg.k, alphabet_arr,
                              bkpt);
          int max_a = 0;
          for (int j = 0; j < cfg.k; j++) {
            if (alphabet_arr[j] > max_a)
              max_a = alphabet_arr[j];
          }
          if (cfg.warmup < 10 * max_a) {
            fprintf(stderr, "Warning: warmup (%d) < 10 * max_alphabet (%d)\n",
                    cfg.warmup, 10 * max_a);
          }
          fprintf(stderr, "ev       =");
          for (int j = 0; j < cfg.k; j++)
            fprintf(stderr, " %.4f", ev[j]);
          fprintf(stderr, "\nbits     =");
          for (int j = 0; j < cfg.k; j++)
            fprintf(stderr, " %d", bits[j]);
          fprintf(stderr, "\nalphabet =");
          for (int j = 0; j < cfg.k; j++)
            fprintf(stderr, " %d", alphabet_arr[j]);
          fprintf(stderr, "\nscore    = %.6f\n", score);

          have_bkpt = 1;
          free(warmup_proj);
          warmup_proj = NULL;
        }
        continue;
      }

      digitize(y, bkpt, cfg.k, alphabet_arr, word);

      for (int j = 0; j < cfg.k; j++) {
        if (word[j] >= alphabet_arr[j]) {
          fprintf(stderr, "BUG: word[%d] = %d >= alphabet %d\n", j, word[j],
                  alphabet_arr[j]);
        }
      }

      printf("%2d  ", r);
      for (int j = 0; j < cfg.k; j++) {
        printf("%c", 'a' + word[j]);
      }
      printf("\n");
    }
  }

  fprintf(stderr, "worst drift before cleanup: overlap %.3e, length %.3e\n",
          worst_dot_seen, worst_len_seen);

  rc = 0;

cleanup:
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
  return rc;
}
