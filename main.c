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
// How fast the smoothed angle rate forgets. Memory is about 1/this readings,
// so at 0.3 it reflects roughly the last three. NOT a setting: it is internal
// to the detector, and a knob here would only invite tuning it on one dataset
#define RATE_BLEND 0.3

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
static FILE *open_sink(const char *path, int *owned) {
  *owned = 0;
  if (strcmp(path, "-") == 0)
    return stdout;
  FILE *f = fopen(path, "w");
  if (f)
    *owned = 1;
  return f;
}

// Persist the frozen dictionary as data; an analysis script needs exactly this
// to recompute distances without re-running the C
static int write_dictionary(const char *path, const Config *cfg,
                            const double *ev, const int *bits,
                            const int *alphabet, const double *bkpt) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error: Cannot write dictionary to '%s'\n", path);
    return -1;
  }

  fprintf(f, "pos,alphabet,bits,ev");
  for (int b = 0; b < MAX_ALPHABET; b++)
    fprintf(f, ",bkpt%d", b);
  fprintf(f, "\n");

  for (int j = 0; j < cfg->k; j++) {
    fprintf(f, "%d,%d,%d,%.17g", j, alphabet[j], bits[j], ev[j]);
    for (int b = 0; b < MAX_ALPHABET; b++) {
      double v = bkpt[j * MAX_ALPHABET + b];
      if (b >= alphabet[j])
        fprintf(f, ","); // slot unused at this position
      else if (v == DBL_MAX)
        fprintf(f, ",inf"); // the wall the symbol search always stops at
      else
        fprintf(f, ",%.17g", v);
    }
    fprintf(f, "\n");
  }

  fclose(f);
  return 0;
}

/*
 * Builds a dictionary from a block of projections: importances, bit split,
 * alphabet sizes and cut points.
 *
 * Called twice: once when the warm-up ends, and once at every re-basing. The
 * two paths must produce a dictionary the same way or they will drift apart,
 * which is the whole reason this is a function and not two copies.
 *
 * cfg       settings in force
 * proj      n_proj rows of k numbers, flat; row r position j at proj[r*k + j]
 * n_proj    how many rows are valid                          [windows]
 * why       short label for the log line, e.g. "warmup" or "rebasis"
 * ev        out, k importances as shares, they sum to 1
 * bits      out, k bit counts, they sum to cfg->total_bits
 * alphabet  out, k alphabet sizes, each 1 << bits[j]
 * bkpt      out, k rows of MAX_ALPHABET cut points, flat
 * score     out, the objective the chosen split reaches, or NULL. Describes
 *           the split BEFORE the MAX_BITS clamp, so if a clamp fired the real
 *           allocation is worse than this number says
 *
 * Returns 0 on success, -1 if the split could not be solved.
 *
 * Edge cases:
 *   n_proj < 10 * largest alphabet  -> cut points are noise, warned not refused
 *   a position may receive zero bits -> alphabet 1, one cut point, letter
 *                                       always 0, which is intended
 *   ROW ORDER DOES NOT MATTER. Both consumers compute per column statistics,
 *   a variance and a sorted quantile, so a ring buffer may be passed straight
 *   in even though its rows are not in time order.
 */
static int build_dictionary(const Config *cfg, const double *proj, int n_proj,
                            const char *why, double *ev, int *bits,
                            int *alphabet, double *bkpt, double *score) {
  // Step 1: how much spread each position carries, as shares summing to 1
  estimate_eigenvalues(proj, n_proj, cfg->k, ev);
  // Step 2: split total_bits across the k positions. DP, so this is the best
  // split for this lambda, not a greedy approximation
  double local_score = 0.0;
  if (daa_allocate(ev, cfg->k, cfg->total_bits, cfg->lambda, NULL, bits,
                   &local_score) != 0) {
    return -1;
  }
  if (score)
    *score = local_score;
  // Step 3: clamp to what digitize can represent, and note the widest
  // alphabet, since that is the one that needs the most rows to resolve
  int max_a = 0;
  for (int j = 0; j < cfg->k; j++) {
    if (bits[j] > MAX_BITS) {
      fprintf(stderr, "Warning: Position %d wanted %d bits, clamped to %d\n", j,
              bits[j], MAX_BITS);
      bits[j] = MAX_BITS;
    }
    alphabet[j] = 1 << bits[j];
    if (alphabet[j] > max_a) {
      max_a = alphabet[j];
    }
  }
  // Step 4: a cut point is a quantile. With alphabet A the thinnest cell holds
  // about n_proj/A rows, and a quantile drawn from a handful of rows is noise
  if (n_proj < 10 * max_a) {
    fprintf(stderr,
            "Warning: %s %d is small for largest alphabet %d, cut points will "
            "be noisy, want at least %d\n",
            why, n_proj, max_a, 10 * max_a);
  }
  // Step 5: the cut points, one row of MAX_ALPHABET per position
  compute_breakpoints(proj, n_proj, cfg->k, alphabet, bkpt);
  // Step 6: one log line, so both paths report identically
  fprintf(stderr, "ev       =");
  for (int j = 0; j < cfg->k; j++)
    fprintf(stderr, " %.4f", ev[j]);
  fprintf(stderr, "\nbits     =");
  for (int j = 0; j < cfg->k; j++)
    fprintf(stderr, " %d", bits[j]);
  fprintf(stderr, "\nalphabet =");
  for (int j = 0; j < cfg->k; j++)
    fprintf(stderr, " %d", alphabet[j]);
  fprintf(stderr, "\nscore    = %.6f\n", local_score);
  return 0;
}

int main(int argc, char *argv[]) {
  Config cfg;
  RunOptions run;

  // Settings come from a file when given, otherwise the built-in defaults
  if (argc > 2) {
    fprintf(stderr, "Usage: %s [settings-file]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (argc == 2) {
    if (config_load(argv[1], &cfg, &run) != 0)
      return EXIT_FAILURE;
  } else {
    config_defaults(&cfg, &run);
  }
  // Every run describes itself. Written to stderr, which is already the
  // diagnostics channel, and in a shape that can be fed straight back in as a
  // settings file
  config_dump(stderr, &cfg, &run);

  // Default exit code and safe initialization of pointers and state
  int rc = EXIT_FAILURE;
  double *demo_flat = NULL;
  double *win = NULL;
  double *xc = NULL;
  double *y = NULL;
  double *y_cand = NULL;
  int *word = NULL;
  double *bkpt = NULL;
  double *warmup_proj = NULL;
  double *ev = NULL;
  int *bits = NULL;
  int *alphabet_arr = NULL;
  double *ev_live = NULL;
  double *angles_scratch = NULL;
  double *angles = NULL;
  FILE *out = NULL;
  int out_owned = 0;

  RunningMean my_stats = {0};
  OjaPCA oja_live = {0};
  OjaPCA oja_avg = {0};
  OjaPCA oja_frozen = {0};
  OjaPCA oja_cand = {0};

  const OjaPCA *symbol_basis = &oja_avg;
  WindowStream stream = {0};
  ProjRing ring = {0};
  ProjRing cand_ring = {0};

  int warm_count = 0;
  int have_bkpt = 0;
  long n_seen = 0;
  long n_words = 0;
  double worst_dot_seen = 0.0;
  double worst_len_seen = 0.0;
  double worst_dot_avg = 0.0;
  double worst_len_avg = 0.0;
  double worst_noise = 0.0;
  double theta_prev = 0.0;
  long theta_prev_at = -1;
  double theta_rate = 0.0; /* radians per window, smoothed */
  int have_rate = 0;
  double worst_rate = 0.0;
  int staging = 0;
  int cand_ready = 0;
  long n_epochs = 0;    /* re-basings so far; epoch 0 is the warm-up one */
  long late_since = -1; /* window we first went past the policy with no ready
                           ring, or -1 when we are not late */
  long worst_late = 0;  /* worst run of windows spent waiting [windows] */
  double worst_stale =
      0.0; /* worst staleness at a promotion        [radians] */
  long quiet_until = 0;
  double worst_over = 0.0;
  int was_below = 1;
  int warned_infeasible = 0;
  double t_io = 0.0;
  double t_algo = 0.0;

  int use_demo = (strcmp(run.input, "builtin") == 0);

  // Allocate algorithm buffers based on runtime configuration
  win = calloc((size_t)cfg.m, sizeof(double));
  xc = calloc((size_t)cfg.m, sizeof(double));
  y = calloc((size_t)cfg.k, sizeof(double));
  y_cand = calloc((size_t)cfg.k, sizeof(double));
  word = calloc((size_t)cfg.k, sizeof(int));
  bkpt = calloc((size_t)cfg.k * MAX_ALPHABET, sizeof(double));
  warmup_proj = calloc((size_t)cfg.warmup * cfg.k, sizeof(double));
  ev = calloc((size_t)cfg.k, sizeof(double));
  bits = calloc((size_t)cfg.k, sizeof(int));
  angles_scratch = calloc((size_t)cfg.k * (size_t)(cfg.k + 1), sizeof(double));
  angles = calloc((size_t)cfg.k, sizeof(double));
  alphabet_arr = calloc((size_t)cfg.k, sizeof(int));
  ev_live = calloc((size_t)cfg.k, sizeof(double));
  if (use_demo)
    demo_flat = calloc(DEMO_LEN, sizeof(double));

  if (!win || !xc || !y || !y_cand || !word || !bkpt || !warmup_proj || !ev ||
      !bits || !ev_live || !alphabet_arr || !angles_scratch || !angles ||
      (use_demo && !demo_flat)) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    goto cleanup;
  }

  // Initialize computational components
  if (rmean_init(&my_stats, cfg.m) != 0) {
    fprintf(stderr, "Error: rmean_init failed\n");
    goto cleanup;
  }

  if (oja_init(&oja_live, cfg.k, cfg.m) != 0) {
    fprintf(stderr, "Error: oja_init failed, components must not exceed "
                    "channels * window\n");
    goto cleanup;
  }

  if (oja_init(&oja_avg, cfg.k, cfg.m) != 0) {
    fprintf(stderr, "Error: oja_init failed for the averaged basis\n");
    goto cleanup;
  }

  if (oja_init(&oja_frozen, cfg.k, cfg.m) != 0) {
    fprintf(stderr, "Error: oja_init failed for the frozen basis\n");
    goto cleanup;
  }

  if (oja_init(&oja_cand, cfg.k, cfg.m) != 0) {
    fprintf(stderr, "Error: oja_init failed for the candidate basis\n");
    goto cleanup;
  }

  if (pring_init(&ring, cfg.ring_w, cfg.k) != 0) {
    fprintf(stderr, "Error: pring_init failed\n");
    goto cleanup;
  }
  if (pring_init(&cand_ring, cfg.ring_w, cfg.k) != 0) {
    fprintf(stderr, "Error: pring_init failed for the candidate ring\n");
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
  for (int j = 0; j < cfg.k; j++)
    fprintf(out, ",sym%d", j);
  fprintf(out, "\n");

  // Main stream processing loop; reading and computing are timed separately so
  // a slow text parser can never be mistaken for a slow algorithm
  for (;;) {
    double t0 = now_seconds();
    int got = stream_next(&stream, win);
    t_io += now_seconds() - t0;
    if (!got)
      break;

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
    oja_update(&oja_live, xc, cfg.lr);

    // And the smoothed copy follows it. Oja never settles, so freezing one
    // point of its wandering freezes the whole wander with it; the average
    // cancels what the wandering adds.
    if (oja_blend(&oja_avg, &oja_live, cfg.beta) != 0) {
      fprintf(stderr, "Error: oja_blend rejected its arguments\n");
      t_algo += now_seconds() - t0;
      goto cleanup;
    }

    // Periodically monitor subspace drift and restore orthonormality
    if (n_seen % cfg.gs_period == 0) {
      double d_err, l_err;
      oja_orthonormality_error(&oja_live, &d_err, &l_err);
      if (d_err > worst_dot_seen)
        worst_dot_seen = d_err;
      if (l_err > worst_len_seen)
        worst_len_seen = l_err;

      /* The guarantee is stated about the basis that WRITES, and that is now
         the smoothed one, so its drift is the number with meaning. The raw
         one is kept because it is the early warning that the step is too big.
       */
      double da_err, la_err;
      oja_orthonormality_error(&oja_avg, &da_err, &la_err);
      if (da_err > worst_dot_avg)
        worst_dot_avg = da_err;
      if (la_err > worst_len_avg)
        worst_len_avg = la_err;

      gram_schmidt(&oja_live);
      /* The average of unit vectors is not a unit vector and the average of
         perpendicular ones is not perpendicular, so the smoothed copy needs
         straightening too, at the same moment. */
      gram_schmidt(&oja_avg);
      if (have_bkpt && n_seen % cfg.angle_period == 0) {
        double theta_max = 0.0;
        int ang_rc = subspace_angles(&oja_frozen, &oja_avg, angles_scratch,
                                     angles, &theta_max);
        if (ang_rc != 0) {
          fprintf(stderr,
                  "Warning: subspace_angles returned '%d' at window '%ld', "
                  "measurement skipped\n",
                  ang_rc, n_seen - 1);
        } else {
          double theta_noise = 0.0;
          int noise_rc = subspace_angles(&oja_avg, &oja_live, angles_scratch,
                                         NULL, &theta_noise);
          if (noise_rc != 0) {
            fprintf(stderr,
                    "Warning: noise reference returned '%d' at window '%ld', "
                    "reported as zero\n",
                    noise_rc, n_seen - 1);
            theta_noise = 0.0;
          } else if (theta_noise > worst_noise) {
            worst_noise = theta_noise;
          }
          if (theta_prev_at >= 0) {
            long elapsed = n_seen - theta_prev_at;
            double sample = (theta_max - theta_prev) / (double)elapsed;

            if (!have_rate) {
              theta_rate = sample;
              have_rate = 1;
            } else {
              theta_rate =
                  (1.0 - RATE_BLEND) * theta_rate + RATE_BLEND * sample;
            }

            if (theta_rate > worst_rate)
              worst_rate = theta_rate;
          }
          theta_prev = theta_max;
          theta_prev_at = n_seen;

          /* How much angle is left before the policy, and how much the basis
   will travel while a fresh ring fills. Both in radians */
          double remaining = cfg.theta_threshold - theta_max;
          double travel = theta_rate * (double)cfg.ring_w;
          if (travel < 0.0) {
            /* A basis coming back toward the frozen one needs no head start.
               Clamping here rather than testing the sign later keeps the
               condition below a single expression */
            travel = 0.0;
          }

          if (!warned_infeasible && travel >= cfg.theta_threshold) {
            /* A verdict on the configuration, not on the data: at this rate
               the basis crosses the entire policy in fewer windows than the
               preparation needs, so no warning level can arrive in time.
               Printed once, because it would otherwise repeat every reading */
            fprintf(stderr,
                    "Warning: at %.4f deg per window the basis crosses the "
                    "whole %.2f deg policy in fewer than ring_window %d "
                    "windows, so no warning level can prepare in time. Lower "
                    "step, raise theta_threshold_deg, or lower ring_window\n",
                    theta_rate / SPARTAN_DEG_TO_RAD,
                    cfg.theta_threshold / SPARTAN_DEG_TO_RAD, cfg.ring_w);
            warned_infeasible = 1;
          }

          /* The latch. Once staging begins there is a snapshot and a filling
             ring behind it, so a rate that drops must not undo it: restarting
             would throw the collection away and could repeat forever on a
             wobbly rate, never finishing one ring.
             The margin is the travel we need plus the noise reference, which
             covers where we are now. It does NOT cover the error in the rate
             itself; that shows up as occasional lateness and is measured as
             the overshoot */
          if (!staging && remaining <= travel + theta_noise) {
            staging = 1;
            fprintf(stderr, "warn,%ld,%.9f,%.9f,%.9f\n", n_seen - 1, theta_max,
                    travel, theta_noise);
            int cand_rc = oja_copy_from(&oja_cand, &oja_avg);
            if (cand_rc != 0) {
              fprintf(
                  stderr,
                  "Error: could not snapshot candidate basis, returned %d\n",
                  cand_rc);
              t_algo += now_seconds() - t0;
              goto cleanup;
            }
            pring_reset(&cand_ring);
            cand_ready = 0;
          }
          if (theta_max > cfg.theta_threshold) {
            if (was_below) {
              double over = theta_max - cfg.theta_threshold;
              if (over > worst_over)
                worst_over = over;
            }
            was_below = 0;

            if (n_seen >= quiet_until) {
              fprintf(stderr, "cut,%ld,%.9f,%.9f,%.9f\n", n_seen - 1, theta_max,
                      theta_max - cfg.theta_threshold, theta_noise);
              quiet_until = n_seen + cfg.warmup;
            }
            if (staging && cand_ready) {
              /* TODO 1: Χτίσιμο νέου λεξικού από το cand_ring */
              double new_score = 0.0;
              if (build_dictionary(&cfg, cand_ring.buf, cand_ring.count,
                                   "rebasis", ev, bits, alphabet_arr, bkpt,
                                   &new_score) != 0) {
                fprintf(stderr, "Error: could not build candidate dictionary "
                                "at re-basing\n");
                t_algo += now_seconds() - t0;
                goto cleanup;
              }
              /* How far the incoming basis already is from the data on the day
               * it starts work. theta_max measures the gap we are fixing; this
               * measures the gap the fix leaves behind. The candidate was
               * frozen at the warn, and whatever the subspace did between the
               * warn and now, it did without it.
               *
               *   staleness = angle(oja_cand, oja_avg) [radians]
               *
               * A re-basing buys theta_max - staleness. Zero staleness is a
               * perfect replacement; staleness equal to theta_max means it
               * bought nothing.
               *
               * Both bases must be orthonormal or these are not angles. oja_avg
               * was straightened at the top of this block; oja_cand is a copy
               * of a straightened basis that no oja_update ever touches. */
              double staleness = 0.0;
              int stale_rc = subspace_angles(&oja_cand, &oja_avg,
                                             angles_scratch, NULL, &staleness);
              if (stale_rc != 0) {
                fprintf(stderr,
                        "Warning: staleness reference returned '%d' at window "
                        "'%ld', "
                        "reported as zero\n",
                        stale_rc, n_seen - 1);
                staleness = 0.0;
              }
              if (staleness > worst_stale)
                worst_stale = staleness;

              /* TODO 2: Η υποψήφια βάση παίρνει τη θέση της oja_frozen */
              int frz_rc = oja_copy_from(&oja_frozen, &oja_cand);
              if (frz_rc != 0) {
                fprintf(
                    stderr,
                    "Error: could not promote candidate basis, returned %d\n",
                    frz_rc);
                t_algo += now_seconds() - t0;
                goto cleanup;
              }
              symbol_basis = &oja_frozen;

              /* TODO 3: Καθαρισμός του κύριου ring */
              pring_reset(&ring);

              /* TODO 4: Καθαρισμός κατάστασης staging και re-seed του angle
               * rate */
              staging = 0;
              cand_ready = 0;
              theta_prev_at = -1;

              /* TODO 5: Αύξηση εποχής και καταγραφή στο stderr */
              n_epochs++;
              fprintf(stderr, "rebasis,%ld,%.9f,%.9f,%.6f\n", n_seen - 1,
                      theta_max, staleness, new_score);

              /* TODO 6: Επαναφορά της καθυστέρησης */
              if (late_since >= 0) {
                long late = n_seen - late_since;
                if (late > worst_late)
                  worst_late = late;
              }
              late_since = -1;

            } else {
              /* TODO 7: Υπέρβαση χωρίς έτοιμο ring (Policy A waiting) */
              if (late_since < 0) {
                late_since = n_seen;
              }
            }
          } else {
            was_below = 1;
            /* TODO 8: Αν η γωνία πέσει ξανά κάτω από το κατώφλι, η αναμονή
             * τελείωσε */
            late_since = -1;
          }
          fprintf(stderr, "angle,%ld,%.9f", n_seen - 1, theta_max);
          for (int j = 0; j < cfg.k; j++)
            fprintf(stderr, ",%.9f", angles[j]);
          fprintf(stderr, ",%.9f,%.9f\n", theta_noise, theta_rate);
        }
      }
    }

    // Project centered window onto the learned PCA subspace
    project(symbol_basis, xc, y);
    pring_push(&ring, y);

    if (staging) {
      project(&oja_cand, xc, y_cand);
      pring_push(&cand_ring, y_cand);

      if (!cand_ready && cand_ring.count == cand_ring.cap) {
        cand_ready = 1;
        fprintf(stderr, "ready,%ld\n", n_seen - 1);
      }
    }

    // Warmup phase: collect projections to establish DAA bit allocation and
    // breakpoints
    if (!have_bkpt) {
      for (int j = 0; j < cfg.k; j++) {
        warmup_proj[warm_count * cfg.k + j] = y[j];
      }
      warm_count++;

      if (warm_count == cfg.warmup) {
        if (build_dictionary(&cfg, warmup_proj, warm_count, "warmup", ev, bits,
                             alphabet_arr, bkpt, NULL) != 0) {
          fprintf(stderr, "Error: Could not build dictionary\n");
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        // Persist the frozen dictionary for downstream analysis
        if (run.dict_out[0] && write_dictionary(run.dict_out, &cfg, ev, bits,
                                                alphabet_arr, bkpt) != 0) {
          t_algo += now_seconds() - t0;
          goto cleanup;
        }

        gram_schmidt(&oja_live);
        gram_schmidt(&oja_avg);

        int frozen_rc = oja_copy_from(&oja_frozen, &oja_avg);
        if (frozen_rc != 0) {
          fprintf(stderr,
                  "Error: Could not freeze the basis, oja_copy_from returned "
                  "%d\n",
                  frozen_rc);
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        symbol_basis = &oja_frozen;
        pring_reset(&ring);
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

    if (n_seen % cfg.ring_w == 0)
      pring_refresh(&ring);

    if (ring.count == ring.cap) {
      pring_importances(&ring, ev_live);
      double gap =
          daa_gap(ev_live, cfg.k, cfg.total_bits, cfg.lambda, NULL, bits);
      fprintf(stderr, "gap,%ld,%.6f\n", n_seen - 1, gap);
    }

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
          "drift    = overlap %.3e, length %.3e (averaged basis between "
          "straightenings; a health indicator, not the bound, since the basis "
          "that writes is straightened at the freeze)\n",
          worst_dot_avg, worst_len_avg);
  fprintf(stderr,
          "drift raw= overlap %.3e, length %.3e (raw Oja, watch it for a step "
          "that is too big)\n",
          worst_dot_seen, worst_len_seen);
  fprintf(stderr,
          "noise    = %.3f deg worst (averaged basis against the raw one, how "
          "much of any displacement is ours)\n",
          worst_noise / SPARTAN_DEG_TO_RAD);
  fprintf(stderr, "rate     = %.4f deg/window worst smoothed\n",
          worst_rate / SPARTAN_DEG_TO_RAD);
  fprintf(stderr, "overshoot= %.4f deg worst past the policy when first seen\n",
          worst_over / SPARTAN_DEG_TO_RAD);
  fprintf(stderr, "epochs   = %ld re-basings after the warm-up one\n",
          n_epochs);
  fprintf(stderr,
          "late     = %ld windows worst spent past the policy waiting for a "
          "full candidate ring\n",
          worst_late);
  fprintf(stderr,
          "staleness= %.3f deg worst (how far the incoming basis already was "
          "from the data when it took over)\n",
          worst_stale / SPARTAN_DEG_TO_RAD);
  fprintf(stderr, "time     = %.3f s reading, %.3f s algorithm\n", t_io,
          t_algo);

  rc = EXIT_SUCCESS;

cleanup:
  if (out && out_owned)
    fclose(out);
  free(demo_flat);
  free(win);
  free(xc);
  free(y);
  free(y_cand);
  free(word);
  free(bkpt);
  free(warmup_proj);
  free(ev);
  free(ev_live);
  free(bits);
  free(alphabet_arr);
  free(angles_scratch);
  free(angles);
  oja_free(&oja_live);
  oja_free(&oja_frozen);
  oja_free(&oja_avg);
  oja_free(&oja_cand);
  rmean_free(&my_stats);
  stream_free(&stream);
  pring_free(&ring);
  pring_free(&cand_ring);
  return rc;
}
