/* Online SPARTAN - the driver.
 *
 * Reads settings from a file, pulls windows from a stream (a text file or the
 * built-in demo array), and writes one row of symbols per window.
 *
 *   ./spartan_toy                 defaults, built-in demo data
 *   ./spartan_toy run.cfg         everything comes from the settings file
 */

#define _POSIX_C_SOURCE 200809L

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "spartan.h"
#include "stream.h"

// Fixed dimensions for the built-in demo stream; nothing else may use these
#define DEMO_ROWS 20
#define DEMO_COLS 8
#define DEMO_REPEATS 10
#define DEMO_LEN (DEMO_ROWS * DEMO_COLS * DEMO_REPEATS)

// 20 short univariate series used as source patterns
static const double demo_rows[DEMO_ROWS][DEMO_COLS] = {
    {1.0, 2.1, 2.9, 4.2, 5.0, 6.1, 6.9, 8.0},
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

// Monotonic wall clock: immune to system time changes, and it measures elapsed
// time rather than CPU time, which is what we want when part of the work waits
// on a disk
static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Open a destination file, or hand back stdout for "-"; the flag tells the
// caller whether it owns the handle and must close it
static FILE* open_sink(const char* path, int* owned) {
  *owned = 0;
  if (strcmp(path, "-") == 0) return stdout;
  FILE* f = fopen(path, "w");
  if (f) *owned = 1;
  return f;
}

// Persist the frozen dictionary as data; an analysis script needs exactly this
// to recompute distances without re-running the C
static int write_dictionary(const char* path, const Config* cfg,
                            const double* ev, const int* bits,
                            const int* alphabet, const double* bkpt) {
  FILE* f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error: Cannot write dictionary to '%s'\n", path);
    return -1;
  }

  fprintf(f, "pos,alphabet,bits,ev");
  for (int b = 0; b < MAX_ALPHABET; b++) fprintf(f, ",bkpt%d", b);
  fprintf(f, "\n");

  for (int j = 0; j < cfg->k; j++) {
    fprintf(f, "%d,%d,%d,%.17g", j, alphabet[j], bits[j], ev[j]);
    for (int b = 0; b < MAX_ALPHABET; b++) {
      double v = bkpt[j * MAX_ALPHABET + b];
      if (b >= alphabet[j])
        fprintf(f, ",");  // slot unused at this position
      else if (v == DBL_MAX)
        fprintf(f, ",inf");  // the wall the symbol search always stops at
      else
        fprintf(f, ",%.17g", v);
    }
    fprintf(f, "\n");
  }

  fclose(f);
  return 0;
}

int main(int argc, char* argv[]) {
  Config cfg;
  RunOptions run;

  // Settings come from a file when given, otherwise the built-in defaults
  if (argc > 2) {
    fprintf(stderr, "Usage: %s [settings-file]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (argc == 2) {
    if (config_load(argv[1], &cfg, &run) != 0) return EXIT_FAILURE;
  } else {
    config_defaults(&cfg, &run);
  }

  // Default exit code and safe initialization of pointers and state
  int rc = EXIT_FAILURE;
  double* demo_flat = NULL;
  double* win = NULL;
  double* xc = NULL;
  double* y = NULL;
  int* word = NULL;
  double* bkpt = NULL;
  double* warmup_proj = NULL;
  double* ev = NULL;
  int* bits = NULL;
  int* alphabet_arr = NULL;

  FILE* out = NULL;
  int out_owned = 0;

  RunningMean my_stats = {0};
  OjaPCA oja = {0};
  WindowStream stream = {0};

  int warm_count = 0;
  int have_bkpt = 0;
  long n_seen = 0;
  long n_words = 0;
  double worst_dot_seen = 0.0;
  double worst_len_seen = 0.0;
  double t_io = 0.0;
  double t_algo = 0.0;

  int use_demo = (strcmp(run.input, "builtin") == 0);

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
  if (use_demo) demo_flat = calloc(DEMO_LEN, sizeof(double));

  if (!win || !xc || !y || !word || !bkpt || !warmup_proj || !ev || !bits ||
      !alphabet_arr || (use_demo && !demo_flat)) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    goto cleanup;
  }

  // Initialize computational components
  if (rmean_init(&my_stats, cfg.m) != 0) {
    fprintf(stderr, "Error: rmean_init failed\n");
    goto cleanup;
  }

  if (oja_init(&oja, cfg.k, cfg.m) != 0) {
    fprintf(stderr,
            "Error: oja_init failed, components must not exceed "
            "channels * window\n");
    goto cleanup;
  }

  // Initialize input stream from the settings file or the built-in array
  if (use_demo) {
    const size_t block = DEMO_ROWS * DEMO_COLS;
    for (int rep = 0; rep < DEMO_REPEATS; rep++) {
      memcpy(demo_flat + rep * block, demo_rows, block * sizeof(double));
    }
    if (stream_init_array(&stream, demo_flat, DEMO_LEN, cfg.d, cfg.m_time,
                          cfg.stride) != 0) {
      fprintf(stderr, "Error: Failed to initialize built-in stream source\n");
      goto cleanup;
    }
  } else {
    if (stream_init_file(&stream, run.input, cfg.d, cfg.m_time, cfg.stride) !=
        0) {
      fprintf(stderr, "Error: Failed to open input '%s'\n", run.input);
      goto cleanup;
    }
  }

  // Open the symbol sink and write the CSV header
  out = open_sink(run.words_out, &out_owned);
  if (!out) {
    fprintf(stderr, "Error: Cannot write words to '%s'\n", run.words_out);
    goto cleanup;
  }

  fprintf(out, "window");
  for (int j = 0; j < cfg.k; j++) fprintf(out, ",sym%d", j);
  fprintf(out, "\n");

  // Main stream processing loop; reading and computing are timed separately so
  // a slow text parser can never be mistaken for a slow algorithm
  for (;;) {
    double t0 = now_seconds();
    int got = stream_next(&stream, win);
    t_io += now_seconds() - t0;
    if (!got) break;

    t0 = now_seconds();
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
          t_algo += now_seconds() - t0;
          goto cleanup;
        }

        // Compute alphabet sizes from allocated bit budget
        int max_a = 0;
        for (int j = 0; j < cfg.k; j++) {
          alphabet_arr[j] = 1 << bits[j];
          if (alphabet_arr[j] > MAX_ALPHABET) {
            fprintf(stderr,
                    "Warning: Position %d wanted alphabet %d, clamped to %d\n",
                    j, alphabet_arr[j], MAX_ALPHABET);
            alphabet_arr[j] = MAX_ALPHABET;
            bits[j] = 31 - __builtin_clz((unsigned)MAX_ALPHABET);
          }
          if (alphabet_arr[j] > max_a) max_a = alphabet_arr[j];
        }

        // Equi-depth cut points are noise when a bin holds only a sample or two
        if (cfg.warmup < 10 * max_a) {
          fprintf(stderr,
                  "Warning: Warmup %d is small for the largest alphabet %d, "
                  "cut points will be noisy, want at least %d\n",
                  cfg.warmup, max_a, 10 * max_a);
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

        // Persist the frozen dictionary for downstream analysis
        if (run.dict_out[0] && write_dictionary(run.dict_out, &cfg, ev, bits,
                                                alphabet_arr, bkpt) != 0) {
          t_algo += now_seconds() - t0;
          goto cleanup;
        }

        have_bkpt = 1;
        free(warmup_proj);
        warmup_proj = NULL;
      }
      t_algo += now_seconds() - t0;
      continue;
    }

    // Quantization: convert subspace coordinates into discrete symbols
    digitize(y, bkpt, cfg.k, alphabet_arr, word);
    n_words++;

    // Emit symbolic representation for the current window
    fprintf(out, "%ld", n_seen - 1);
    for (int j = 0; j < cfg.k; j++) {
      if (run.pretty) {
        fprintf(out, ",%c", 'a' + word[j]);
      } else {
        fprintf(out, ",%d", word[j]);
      }
    }
    fprintf(out, "\n");

    t_algo += now_seconds() - t0;
  }

  // Check if the stream was aborted due to data corruption or read failure
  if (stream.err) {
    fprintf(stderr,
            "Error: Stream aborted due to invalid formatting or I/O failure\n");
    goto cleanup;
  }

  // A run that produced nothing must say so rather than look like a success
  if (n_seen == 0) {
    fprintf(stderr,
            "Warning: No windows produced, input is shorter than one window of "
            "%d frames\n",
            cfg.m_time);
  } else if (n_words == 0) {
    fprintf(stderr,
            "Warning: %ld windows seen but none symbolised, warmup is %d\n",
            n_seen, cfg.warmup);
  }

  // Run summary: volume, guarantee slack, and where the time actually went
  fprintf(stderr, "windows  = %ld seen, %ld symbolised\n", n_seen, n_words);
  fprintf(stderr,
          "drift    = overlap %.3e, length %.3e (worst before cleanup)\n",
          worst_dot_seen, worst_len_seen);
  fprintf(stderr, "time     = %.3f s reading, %.3f s algorithm\n", t_io,
          t_algo);

  rc = EXIT_SUCCESS;

cleanup:
  if (out && out_owned) fclose(out);
  free(demo_flat);
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
