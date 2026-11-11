/* Online SPARTAN - the driver.
 *
 * Reads settings from a file, pulls windows from the data file it names, and
 * writes one row of symbols per window.
 *
 *   ./spartan run.cfg         everything comes from the settings file
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

/* How an epoch is re-based when the angle crosses the threshold.
 *
 * Default: at once, with exact PCA on the last ring_w windows, the same
 * operation as the warm-up. Nothing is predicted, so nothing can fire early,
 * and the incoming basis is never older than the ring itself.
 *
 * STAGING_A: the previous design, kept compiled out for comparison on real
 * data. Build it with the Makefile's flags plus the switch (CFLAGS+= on the
 * make command line would REPLACE the flags, losing -O2 and the warnings):
 *     make clean && make CFLAGS="-O2 -Wall -Wextra -Wshadow -std=c11
 * -DSTAGING_A" It predicted the crossing from the rate of the angle, froze a
 * candidate at a warn, filled a ring of projections on it and promoted it at
 * the crossing. Measured on tsb_drift_strong: an early warn froze the candidate
 * 2500 windows before the crossing, so it took over 44.6 deg behind the data
 * and a second re-basing followed 165 windows late. Cost of the default: the
 * ring holds windows, ring_w x m doubles (1 MB at 512 x 256), instead of
 * projections, ring_w x k.
 */
#ifdef STAGING_A
// How fast the smoothed angle rate forgets. Memory is about 1/this readings,
// so at 0.3 it reflects roughly the last three. NOT a setting: it is internal
// to the detector, and a knob here would only invite tuning it on one dataset
#define RATE_BLEND 0.3
// Ceiling on the freeze-ahead scale. A tiny theta_max would send the factor
// to the moon, and past four the prediction is guesswork whatever the data
// does. NOT a setting: freeze_ahead is the knob, this only stops it running
// away when the denominator is small
#define FREEZE_AHEAD_MAX 4.0
#endif

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
      double v =
          bkpt[j * MAX_ALPHABET + b];  // Index flat breakpoint table using
                                       // fixed MAX_ALPHABET row stride
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

static int epoch_path(char* buf, size_t cap, const char* dir, const char* what,
                      long n) {
  int len = snprintf(buf, cap, "%s/%s_%ld.csv", dir, what, n);
  if (len < 0 || (size_t)len >= cap) {
    fprintf(stderr, "Error: path truncated or invalid for directory '%s'\n",
            dir);
    return -1;
  }
  return 0;
}

/* Writes one basis to a CSV file: one row per direction, m numbers each.
 * Returns 0 on success, -1 if the file could not be opened.
 */
static int write_basis(const char* path, const OjaPCA* p) {
  FILE* f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error: Cannot write basis to '%s'\n", path);
    return -1;
  }

  // Header line: dir,v0,v1,...,v(m-1)
  fprintf(f, "dir");
  for (int i = 0; i < p->m; i++) {
    fprintf(f, ",v%d", i);
  }
  fprintf(f, "\n");

  // One row per direction: j,v[j*m+0],v[j*m+1],...
  for (int j = 0; j < p->k; j++) {
    fprintf(f, "%d", j);
    for (int i = 0; i < p->m; i++) {
      fprintf(f, ",%.17g", p->v[j * p->m + i]);
    }
    fprintf(f, "\n");
  }

  fclose(f);
  return 0;
}

/* Makes cand the basis that writes, in the row order the new dictionary
 * expects. Shared by all three ways an epoch opens: the warm-up, a type 1
 * (same directions, new split) and a type 2 (new directions).
 *
 *   1. reorder cand's rows by perm, so row j describes cut-point row j
 *   2. if an outgoing basis exists, keep the orientation: when the map from
 *      the old frame to the new one would be a reflection, flip the last
 *      row, the matching column of r, and re-cut
 *   3. copy cand into frozen
 *
 * Type 1 passes cand == frozen: the same basis, only reordered, done in
 * place with no copy and without touching oja_cand, which may be holding a
 * type 2 candidate. The map from the old frame to the new one is then the
 * permutation matrix itself, so its orientation is the parity of perm.
 *
 * frozen     on entry the OUTGOING basis (ignored when have_prev is 0),
 *            on exit the incoming one
 * cand       the incoming basis; reordered, maybe flipped, in place.
 *            May be frozen itself, see above
 * r          the projections the cut points came from, already in perm
 *            order (build_dictionary did that). NULL only when have_prev is 0
 * perm       from build_dictionary
 * alphabet   the new dictionary's alphabets, read only
 * bkpt       the new cut points, re-cut after a flip
 * scratch    k*k doubles, overwritten
 * have_prev  0 at the warm-up: there is nothing to orient against
 * at         window index, for the log line only
 *
 * Returns 0, or -1 if a step refused.
 *
 * Edge cases:
 *   The orientation test MUST come after the reordering: a permutation has
 *   a determinant of its own.
 *   frozen is read in step 2 and only written in step 3, so it is still the
 *   outgoing basis when the orientation is compared against it.
 *   A flip changes no variance, so bits and alphabet stay; only the cut
 *   points of the flipped row move, and they are re-cut by the one function
 *   that makes cut points rather than mirrored by hand.
 */
static int promote_basis(OjaPCA* frozen, OjaPCA* cand, ProjRing* r,
                         const int* perm, const int* alphabet, double* bkpt,
                         double* scratch, int have_prev, long at) {
  int k = cand->k, m = cand->m;
  int in_place = (cand == frozen);

  if (oja_permute_rows(cand, perm) != 0) {
    fprintf(stderr, "Error: could not reorder the incoming basis\n");
    return -1;
  }

  // Would the map from the old frame to the new one be a reflection?
  int reflected = 0;
  if (in_place) {
    // Parity of a permutation: (k - number of cycles) is even or odd
    int seen[k];
    int cycles = 0;
    for (int i = 0; i < k; i++) seen[i] = 0;
    for (int i = 0; i < k; i++) {
      if (seen[i]) continue;
      cycles++;
      for (int j = i; !seen[j]; j = perm[j])  // walk the cycle through i
        seen[j] = 1;
    }
    reflected = (k - cycles) % 2;
  } else if (have_prev) {
    reflected = basis_det_sign(cand, frozen, scratch) < 0;
  }

  if (reflected) {
    int last = k - 1;
    for (int i = 0; i < m; i++) cand->v[last * m + i] = -cand->v[last * m + i];
    for (int row = 0; row < r->count; row++)
      r->buf[row * k + last] = -r->buf[row * k + last];
    pring_refresh(r);  // the sums went stale with the column
    compute_breakpoints(r->buf, r->count, k, alphabet, bkpt);
    fprintf(stderr, "flip,%ld,%d\n", at, last);
  }

  if (!in_place && oja_copy_from(frozen, cand) != 0) {
    fprintf(stderr, "Error: could not promote the incoming basis\n");
    return -1;
  }
  return 0;
}

/* Writes what one epoch leaves behind: dict_n.csv, basis_n.csv and its line
 * in epochs.csv. Does nothing when no out_dir was given.
 *
 * n        epoch number
 * first    index of the first window this epoch WRITES
 * type     0 warm-up, 1 same directions with a new split, 2 new directions
 * theta    angle that closed the previous epoch [radians], 0 for types 0, 1
 * stale    staleness of the incoming basis [radians], 0 for types 0, 1
 * score    DAA objective of the new split
 *
 * Returns 0, or -1 if a file could not be written.
 */
static int record_epoch(const RunOptions* run, FILE* epochs_f,
                        const Config* cfg, long n, long first, int type,
                        double theta, double stale, double score,
                        const double* ev, const int* bits, const int* alphabet,
                        const double* bkpt, const OjaPCA* frozen) {
  if (!run->out_dir[0]) return 0;
  char path[512];
  if (epoch_path(path, sizeof(path), run->out_dir, "dict", n) != 0 ||
      write_dictionary(path, cfg, ev, bits, alphabet, bkpt) != 0)
    return -1;
  if (epoch_path(path, sizeof(path), run->out_dir, "basis", n) != 0 ||
      write_basis(path, frozen) != 0)
    return -1;
  fprintf(epochs_f, "%ld,%ld,%.9f,%.9f,%.6f,%d\n", n, first, theta, stale,
          score, type);
  return 0;
}

/* Writes the level-letter cut points: one row per extra letter, in word
 * order (level0, spread0, level1, spread1, ...), same layout as a dict file
 * minus ev and bits. Fixed for the whole run, so one file, not one per epoch.
 * Returns 0 on success, -1 if the file could not be opened.
 */
static int write_levels(const char* path, int n_lvl, const int* alphabet,
                        const double* bkpt) {
  FILE* f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error: Cannot write level cut points to '%s'\n", path);
    return -1;
  }
  fprintf(f, "pos,alphabet");
  for (int b = 0; b < LEVEL_ALPHABET; b++) fprintf(f, ",bkpt%d", b);
  fprintf(f, "\n");
  for (int j = 0; j < n_lvl; j++) {
    fprintf(f, "%d,%d", j, alphabet[j]);
    for (int b = 0; b < LEVEL_ALPHABET; b++) {
      double v = bkpt[j * MAX_ALPHABET + b];
      if (v == DBL_MAX)
        fprintf(f, ",inf");
      else
        fprintf(f, ",%.17g", v);
    }
    fprintf(f, "\n");
  }
  fclose(f);
  return 0;
}

static int build_dictionary(const Config* cfg, double* proj, int n_proj,
                            const char* why, int* perm, double* ev, int* bits,
                            int* alphabet, double* bkpt, double* score) {
  estimate_eigenvalues(proj, n_proj, cfg->k, ev);

  for (int i = 0; i < cfg->k; i++) {
    perm[i] = i;
  }

  for (int i = 0; i < cfg->k - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < cfg->k; j++) {
      if (ev[perm[j]] >
          ev[perm[max_idx]]) {  // Indirect selection sort on index array perm
                                // to rank eigenvalues descending without
                                // mutating ev yet
        max_idx = j;
      }
    }
    if (max_idx != i) {
      int tmp = perm[i];
      perm[i] = perm[max_idx];
      perm[max_idx] = tmp;
    }
  }

  int is_identity = 1;
  for (int i = 0; i < cfg->k; i++) {
    if (perm[i] != i) {
      is_identity = 0;
      break;
    }
  }

  if (!is_identity) {
    double tmp_ev[cfg->k];
    for (int i = 0; i < cfg->k; i++) {
      tmp_ev[i] = ev[perm[i]];
    }
    for (int i = 0; i < cfg->k; i++) {
      ev[i] = tmp_ev[i];
    }
    double tmp_row[cfg->k];
    for (int r = 0; r < n_proj; r++) {
      for (int i = 0; i < cfg->k; i++) {
        tmp_row[i] =
            proj[r * cfg->k +
                 perm[i]];  // Gather column perm[i] into slot i via temporary
                            // row buffer to avoid overwriting unvisited columns
      }
      for (int i = 0; i < cfg->k; i++) {
        proj[r * cfg->k + i] = tmp_row[i];
      }
    }
  }

  // Step 2: split total_bits across the k positions. Greedy auction, the
  // exact optimum here: config validation keeps us inside daa_exact
  double local_score = 0.0;
  if (daa_allocate(ev, cfg->k, cfg->total_bits, cfg->lambda, NULL, bits,
                   &local_score) != 0) {
    return -1;
  }
  if (score) *score = local_score;
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
  // Step 4: a cut point is a quantile. With alphabet A the thinnest cell
  // holds about n_proj/A rows, and a quantile drawn from a handful of rows is
  // noise
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
  for (int j = 0; j < cfg->k; j++) fprintf(stderr, " %.4f", ev[j]);
  fprintf(stderr, "\nbits     =");
  for (int j = 0; j < cfg->k; j++) fprintf(stderr, " %d", bits[j]);
  fprintf(stderr, "\nalphabet =");
  for (int j = 0; j < cfg->k; j++) fprintf(stderr, " %d", alphabet[j]);
  fprintf(stderr, "\nscore    = %.6f\n", local_score);
  return 0;
}

int main(int argc, char* argv[]) {
  Config cfg;
  RunOptions run;

  // Everything comes from the settings file, and it must name the data
  if (argc != 2) {
    fprintf(stderr, "Usage: %s settings-file\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (config_load(argv[1], &cfg, &run) != 0) return EXIT_FAILURE;
  if (strcmp(run.input, "builtin") == 0) {
    // "builtin" is what config_defaults leaves when the file has no input
    fprintf(stderr, "Error: '%s' has no 'input = <data file>' line\n", argv[1]);
    return EXIT_FAILURE;
  }
  // Every run describes itself. Written to stderr, which is already the
  // diagnostics channel, and in a shape that can be fed straight back in as a
  // settings file
  config_dump(stderr, &cfg, &run);

  // Default exit code and safe initialization of pointers and state
  int rc = EXIT_FAILURE;
  double* win = NULL;
  double* xc = NULL;
  double* y = NULL;
  double* y_cand = NULL;
  int* word = NULL;
  double* bkpt = NULL;
  double* warmup_proj = NULL;
  double* warm_x = NULL; /* the z-normed warm-up windows, warmup rows of m */
#ifndef STAGING_A
  double* win_ring = NULL; /* the last ring_w z-normed windows, ring order */
  int win_pos = 0;         /* where the next window goes, in [0, ring_w) */
  int win_count = 0;       /* how many are in, saturating at ring_w */
#endif
  double* ev = NULL;
  int* bits = NULL;
  int* alphabet_arr = NULL;
  double* ev_live = NULL;
  double* angles_scratch = NULL;
  double* angles = NULL;
  double* advance_scratch = NULL;
  int* perm = NULL;
  double* lvl = NULL;      /* this window's mean and spread, 2*d     */
  double* lvl_warm = NULL; /* warmup rows of them, for the cut points */
  double* lvl_bkpt = NULL; /* their cut points, rows of MAX_ALPHABET  */
  int* lvl_alpha = NULL;   /* LEVEL_ALPHABET in every slot            */
  int* lvl_word = NULL;    /* their letters                           */
  // Extra letters per word: mean and spread of every channel, or none
  int n_lvl = cfg.level_letters ? 2 * cfg.d : 0;
  FILE* out = NULL;
  FILE* epochs_f = NULL;
  int out_owned = 0;

  RunningMean my_stats = {0};
  OjaPCA oja_live = {0};
  OjaPCA oja_avg = {0};
  OjaPCA oja_frozen = {0};
  OjaPCA oja_cand = {0};

  const OjaPCA* symbol_basis =
      &oja_avg;  // Read-only alias pointing to the active projection basis it
                 // switches from oja_avg to oja_frozen after warmup
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
#ifdef STAGING_A
  double theta_prev = 0.0;
  long theta_prev_at = -1;
  double theta_rate = 0.0; /* radians per window, smoothed */
  int have_rate = 0;
  double worst_rate = 0.0;
  int warned_infeasible = 0;
#endif
  long epoch_start = 0; /* window the current epoch's basis took over at; the
                           angles are measured against that basis, so they
                           start at zero there. Used for the REPORTED
                           per-plane rates, not for the warn: see below */
#ifdef STAGING_A
  int staging = 0;
  int cand_ready = 0;
#endif
  long n_epochs = 0;    /* epochs opened after the warm-up one, both types */
  long n_type1 = 0;     /* of those, how many only re-split the bits */
  long late_since = -1; /* window we first went past the policy with no ready
                           ring, or -1 when we are not late */
  long worst_late = 0;  /* worst run of windows spent waiting [windows] */
  double worst_stale =
      0.0; /* worst staleness at a promotion        [radians] */
  long quiet_until = 0;
  double worst_over = 0.0;
  int was_below = 1;
  double t_io = 0.0;
  double t_algo = 0.0;

  // Allocate algorithm buffers based on runtime configuration
  win = calloc((size_t)cfg.m, sizeof(double));
  xc = calloc((size_t)cfg.m, sizeof(double));
  y = calloc((size_t)cfg.k, sizeof(double));
  y_cand = calloc((size_t)cfg.k, sizeof(double));
  word = calloc((size_t)cfg.k, sizeof(int));
  bkpt = calloc((size_t)cfg.k * MAX_ALPHABET, sizeof(double));
  warmup_proj = calloc((size_t)cfg.warmup * cfg.k, sizeof(double));
  warm_x = calloc((size_t)cfg.warmup * cfg.m, sizeof(double));
#ifndef STAGING_A
  win_ring = calloc((size_t)cfg.ring_w * cfg.m, sizeof(double));
  if (!win_ring) {
    fprintf(stderr, "Error: Memory allocation failed for the window ring\n");
    goto cleanup;
  }
#endif
  ev = calloc((size_t)cfg.k, sizeof(double));
  bits = calloc((size_t)cfg.k, sizeof(int));
  angles_scratch = calloc(
      (size_t)cfg.k * (size_t)(cfg.k + 1),
      sizeof(double));  // Pre-allocate k*k cross-Gram matrix plus k singular
                        // values in one contiguous block for subspace_angles
  angles = calloc((size_t)cfg.k, sizeof(double));
  perm = calloc((size_t)cfg.k, sizeof(int));
  alphabet_arr = calloc((size_t)cfg.k, sizeof(int));
  ev_live = calloc((size_t)cfg.k, sizeof(double));
  advance_scratch =
      calloc((size_t)(2 * cfg.k * cfg.k + cfg.k + 2 * cfg.m),
             sizeof(double));  // Overlap table, accumulated Jacobi rotations
                               // and cosines, plus one principal direction and
                               // its partner, for subspace_advance

  if (n_lvl) {
    lvl = calloc((size_t)n_lvl, sizeof(double));
    lvl_warm = calloc((size_t)cfg.warmup * n_lvl, sizeof(double));
    lvl_bkpt = calloc((size_t)n_lvl * MAX_ALPHABET, sizeof(double));
    lvl_alpha = calloc((size_t)n_lvl, sizeof(int));
    lvl_word = calloc((size_t)n_lvl, sizeof(int));
    if (!lvl || !lvl_warm || !lvl_bkpt || !lvl_alpha || !lvl_word) {
      fprintf(stderr, "Error: Memory allocation failed for level letters\n");
      goto cleanup;
    }
    for (int j = 0; j < n_lvl; j++) lvl_alpha[j] = LEVEL_ALPHABET;
  }

  if (!win || !xc || !y || !y_cand || !word || !bkpt || !warmup_proj ||
      !warm_x || !ev || !bits || !ev_live || !alphabet_arr || !angles_scratch ||
      !angles || !advance_scratch || !perm) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    goto cleanup;
  }

  // Initialize computational components
  if (rmean_init(&my_stats, cfg.m) != 0) {
    fprintf(stderr, "Error: rmean_init failed\n");
    goto cleanup;
  }

  if (oja_init(&oja_live, cfg.k, cfg.m) != 0) {
    fprintf(stderr,
            "Error: oja_init failed, components must not exceed "
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

  // The input stream
  if (stream_init_file(&stream, run.input, cfg.d, cfg.m_time, cfg.stride) !=
      0) {
    fprintf(stderr, "Error: Failed to open input '%s'\n", run.input);
    goto cleanup;
  }

  // Open the symbol sink and write the CSV header
  out = open_sink(run.words_out, &out_owned);
  if (!out) {
    fprintf(stderr, "Error: Cannot write words to '%s'\n", run.words_out);
    goto cleanup;
  }

  fprintf(out, "window");
  for (int j = 0; j < cfg.k; j++) fprintf(out, ",sym%d", j);
  for (int c = 0; c < n_lvl / 2; c++) fprintf(out, ",level%d,spread%d", c, c);
  fprintf(out, "\n");
  // The epoch index, one line per epoch. Opened with "w" so every run starts
  // a clean file
  if (run.out_dir[0]) {
    char idx_path[512];
    int len =
        snprintf(idx_path, sizeof(idx_path), "%s/epochs.csv", run.out_dir);
    if (len < 0 || (size_t)len >= sizeof(idx_path)) {
      fprintf(stderr, "Error: path truncated for '%s/epochs.csv'\n",
              run.out_dir);
      goto cleanup;
    }
    epochs_f = fopen(idx_path, "w");
    if (!epochs_f) {
      fprintf(stderr, "Error: Cannot write epoch index to '%s'\n", idx_path);
      goto cleanup;
    }
    fprintf(epochs_f,
            "epoch,first_window,theta_cut_rad,staleness_rad,score,type\n");

    // The settings this run actually used, every default resolved. The
    // tools read the run's settings from here and nowhere else, so no
    // default exists outside the C. The two compile-time constants they
    // need go in as comments, which keeps the file valid as a settings file
    len = snprintf(idx_path, sizeof(idx_path), "%s/settings.cfg", run.out_dir);
    FILE* sf = (len < 0 || (size_t)len >= sizeof(idx_path))
                   ? NULL
                   : fopen(idx_path, "w");
    if (!sf) {
      fprintf(stderr, "Error: Cannot write '%s/settings.cfg'\n", run.out_dir);
      goto cleanup;
    }
    config_dump(sf, &cfg, &run);
    fprintf(sf, "# const MAX_ALPHABET = %d\n", MAX_ALPHABET);
    fprintf(sf, "# const SPARTAN_EPS = %.17g\n", SPARTAN_EPS);
    fclose(sf);
  }
  // Main stream processing loop; reading and computing are timed separately
  // so a slow text parser can never be mistaken for a slow algorithm
  for (;;) {
    double t0 = now_seconds();
    int got = stream_next(&stream, win);
    t_io += now_seconds() - t0;
    if (!got) break;

    t0 = now_seconds();
    n_seen++;

    // Keep what z-norm is about to erase. n_lvl is 0 when the switch is off
    for (int c = 0; c < n_lvl / 2; c++)
      window_level(&win[c * cfg.m_time], cfg.m_time, &lvl[2 * c],
                   &lvl[2 * c + 1]);

    // Channel-wise z-score normalization
    for (int c = 0; c < cfg.d; c++) {
      znorm_row(&win[c * cfg.m_time],
                cfg.m_time);  // Normalize each contiguous m_time channel slice
                              // independently inside the Mode-1 unfolded window
    }

    // Center window by removing the running streaming mean. xc feeds Oja
    // ONLY: Oja learns directions of spread, so it needs a centred input.
    // Every word, cut point and PCA re-basing uses win itself (see project)
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
      if (d_err > worst_dot_seen) worst_dot_seen = d_err;
      if (l_err > worst_len_seen) worst_len_seen = l_err;

      /* The guarantee is stated about the basis that WRITES, and that is now
         the smoothed one, so its drift is the number with meaning. The raw
         one is kept because it is the early warning that the step is too big.
       */
      double da_err, la_err;
      oja_orthonormality_error(&oja_avg, &da_err, &la_err);
      if (da_err > worst_dot_avg) worst_dot_avg = da_err;
      if (la_err > worst_len_avg) worst_len_avg = la_err;

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
#ifdef STAGING_A
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

            if (theta_rate > worst_rate) worst_rate = theta_rate;
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
               Printed once, because it would otherwise repeat every reading
             */
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
            /* The candidate. Plain, it is where the data is NOW, which is
               already behind by the time it takes over: that gap is the
               staleness the run reports, and inside the new epoch the angle
               then runs all the way to the policy.
               With freeze_ahead it is pushed further along the SAME motion,
               so the epoch opens AHEAD of the data and the angle runs from
               roughly minus half to plus half instead, which halves the
               worst angle inside the epoch.
                   f = (theta_max + freeze_ahead * threshold) / theta_max
               A wrong prediction costs tightness and nothing else: the
               bound is computed from the two real bases at translation time
               and never reads this. */
            int cand_rc;
            double f = 0.0;
            if (cfg.freeze_ahead > 0.0 && theta_max > SPARTAN_EPS) {
              f = (theta_max + cfg.freeze_ahead * cfg.theta_threshold) /
                  theta_max;
              if (f > FREEZE_AHEAD_MAX) f = FREEZE_AHEAD_MAX;
            }
            if (f > 1.0) {
              cand_rc = subspace_advance(&oja_frozen, &oja_avg, f, &oja_cand,
                                         advance_scratch);
              if (cand_rc != 0) {
                /* Predicting is optional, so a refusal falls back to the
                   plain snapshot rather than stopping the run. */
                fprintf(stderr,
                        "Warning: subspace_advance returned '%d' at window "
                        "'%ld', freezing where the data is instead\n",
                        cand_rc, n_seen - 1);
                cand_rc = oja_copy_from(&oja_cand, &oja_avg);
              } else {
                fprintf(stderr, "ahead,%ld,%.6f\n", n_seen - 1, f);
              }
            } else {
              cand_rc = oja_copy_from(&oja_cand, &oja_avg);
            }
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
#endif
          if (theta_max > cfg.theta_threshold) {
            if (was_below) {
              double over = theta_max - cfg.theta_threshold;
              if (over > worst_over) worst_over = over;
            }
            was_below = 0;

            if (n_seen >= quiet_until) {
              fprintf(stderr, "cut,%ld,%.9f,%.9f,%.9f\n", n_seen - 1, theta_max,
                      theta_max - cfg.theta_threshold, theta_noise);
              quiet_until = n_seen + cfg.warmup;
            }
#ifdef STAGING_A
            if (staging && cand_ready) {
#else
            // The ring is full from the end of the warm-up on, since the
            // warm-up windows went into it too, so this is always ready
            if (win_count == cfg.ring_w) {
#endif
              double new_score = 0.0;
#ifndef STAGING_A
              // The new basis is exact PCA on the last ring_w windows, cut
              // on their projections: the same operation as the warm-up.
              // warm_pca centres by the ring's own mean internally; the
              // projections below are of the windows as stored, uncentred
              if (oja_copy_from(&oja_cand, &oja_avg) != 0) goto cleanup;
              int pr = warm_pca(win_ring, win_count, cfg.m, &oja_cand);
              fprintf(stderr, "rebase_pca,%ld,%d\n", n_seen - 1, pr);
              pring_reset(&cand_ring);
              for (int r = 0; r < win_count; r++) {
                project(&oja_cand, &win_ring[(size_t)r * cfg.m], y_cand);
                pring_push(&cand_ring, y_cand);
              }
#endif
              if (build_dictionary(&cfg, cand_ring.buf, cand_ring.count,
                                   "rebasis", perm, ev, bits, alphabet_arr,
                                   bkpt, &new_score) !=
                  0)  // Pass flat ring buffer directly, column variance and
                      // quantiles are invariant to circular row order
              {
                fprintf(stderr,
                        "Error: could not build candidate dictionary "
                        "at re-basing\n");
                t_algo += now_seconds() - t0;
                goto cleanup;
              }
              /* How far the incoming basis already is from the data on the
               * day it starts work. theta_max measures the gap we are fixing;
               * this measures the gap the fix leaves behind. The candidate
               * was frozen at the warn, and whatever the subspace did between
               * the warn and now, it did without it.
               *
               *   staleness = angle(oja_cand, oja_avg) [radians]
               *
               * A re-basing buys theta_max - staleness. Zero staleness is a
               * perfect replacement; staleness equal to theta_max means it
               * bought nothing.
               *
               * Both bases must be orthonormal or these are not angles.
               * oja_avg was straightened at the top of this block; oja_cand
               * is a copy of a straightened basis that no oja_update ever
               * touches. */
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
              if (staleness > worst_stale) worst_stale = staleness;
              /* Reorder, keep the orientation against the outgoing
                 basis, promote. cand_ring is what the cut points came
                 from, so a flip negates its column too. */
              if (promote_basis(&oja_frozen, &oja_cand, &cand_ring, perm,
                                alphabet_arr, bkpt, angles_scratch, 1,
                                n_seen - 1) != 0) {
                t_algo += now_seconds() - t0;
                goto cleanup;
              }
              symbol_basis = &oja_frozen;
              pring_reset(&ring);
#ifndef STAGING_A
              // Oja restarts from the new basis, so the angle starts clean
              oja_copy_from(&oja_avg, &oja_frozen);
              oja_copy_from(&oja_live, &oja_frozen);
#endif

#ifdef STAGING_A
              staging = 0;
              cand_ready = 0;
#endif
#ifdef STAGING_A
              theta_prev_at = -1;
#endif
              /* The angles restart against the basis that just took over. */
              epoch_start = n_seen;

              n_epochs++;
              if (record_epoch(&run, epochs_f, &cfg, n_epochs, n_seen - 1, 2,
                               theta_max, staleness, new_score, ev, bits,
                               alphabet_arr, bkpt, &oja_frozen) != 0) {
                t_algo += now_seconds() - t0;
                goto cleanup;
              }
              fprintf(stderr, "rebasis,%ld,%.9f,%.9f,%.6f\n", n_seen - 1,
                      theta_max, staleness, new_score);

              if (late_since >= 0) {
                long late = n_seen - late_since;
                if (late > worst_late) worst_late = late;
              }
              late_since = -1;

            } else {
              if (late_since < 0) {
                late_since = n_seen;
              }
            }
          } else {
            was_below = 1;
            late_since = -1;
          }
          fprintf(stderr, "angle,%ld,%.9f", n_seen - 1, theta_max);
          for (int j = 0; j < cfg.k; j++) fprintf(stderr, ",%.9f", angles[j]);
#ifdef STAGING_A
          fprintf(stderr, ",%.9f,%.9f", theta_noise, theta_rate);
#else
          // no rate is estimated; 0 keeps the column layout of the log line
          fprintf(stderr, ",%.9f,%.9f", theta_noise, 0.0);
#endif
          /* The k per-plane rates, in degrees per window, taken straight
             from the displacement since the epoch opened rather than from a
             difference of two readings. Measured, REPORTED, and deliberately
             not fed to the warn above: an average over the whole epoch is
             dominated by the fast start and fires the warning too early,
             which was measured to make staleness worse. A single scalar also
             hides which directions are moving and which stand still, and
             that difference is what says whether the drift has a direction
             worth predicting or is a random walk. */
          long elapsed_ep = n_seen - epoch_start;
          for (int j = 0; j < cfg.k; j++) {
            double rj = (elapsed_ep > 0) ? angles[j] / (double)elapsed_ep : 0.0;
            fprintf(stderr, ",%.9f", rj / SPARTAN_DEG_TO_RAD);
          }
          fprintf(stderr, "\n");
        }
      }
    }

#ifndef STAGING_A
    memcpy(&win_ring[(size_t)win_pos * cfg.m], win,
           (size_t)cfg.m * sizeof(double));
    win_pos = (win_pos + 1) % cfg.ring_w;
    if (win_count < cfg.ring_w) win_count++;
#endif
    // Project the z-normed window, nothing subtracted, onto the epoch's basis
    project(symbol_basis, win, y);
    pring_push(&ring, y);

#ifdef STAGING_A
    if (staging) {
      project(&oja_cand, win, y_cand);
      pring_push(&cand_ring, y_cand);

      if (!cand_ready && cand_ring.count == cand_ring.cap) {
        cand_ready = 1;
        fprintf(stderr, "ready,%ld\n", n_seen - 1);
      }
    }
#endif

    // Warmup phase: collect projections to establish DAA bit allocation and
    // breakpoints
    if (!have_bkpt) {
      for (int j = 0; j < cfg.k; j++) {
        warmup_proj[warm_count * cfg.k + j] =
            y[j];  // Append k-dimensional projection into row warm_count of
                   // flat warmup matrix
      }
      // Keep the window itself: the epoch-0 basis is computed from these,
      // not from Oja's estimate
      memcpy(&warm_x[(size_t)warm_count * cfg.m], win,
             (size_t)cfg.m * sizeof(double));
      for (int j = 0; j < n_lvl; j++) lvl_warm[warm_count * n_lvl + j] = lvl[j];
      warm_count++;

      if (warm_count == cfg.warmup) {
        if (n_lvl) {
          // Equi-depth cuts over the warm-up, by the same code that cuts
          // the PCA letters. Never re-cut: a level is not tied to a basis,
          // so an epoch change does not move it
          compute_breakpoints(lvl_warm, warm_count, n_lvl, lvl_alpha, lvl_bkpt);
          if (run.out_dir[0]) {
            char path[512];
            int len =
                snprintf(path, sizeof(path), "%s/levels.csv", run.out_dir);
            if (len < 0 || (size_t)len >= sizeof(path) ||
                write_levels(path, n_lvl, lvl_alpha, lvl_bkpt) != 0) {
              t_algo += now_seconds() - t0;
              goto cleanup;
            }
          }
          free(lvl_warm);
          lvl_warm = NULL;
        }
        /* The epoch-0 basis is exact PCA on the warm-up windows, started
           from where Oja got to. Measured on the tsb streams: Oja alone
           needed 2200 to 4400 windows to find the weakest planted part,
           and every type 2 before that was it still converging; PCA on 256
           windows already finds all of it. From here Oja only tracks. */
        int passes = warm_pca(warm_x, warm_count, cfg.m, &oja_avg);
        if (passes == -1) {
          fprintf(stderr, "Error: warm_pca refused its arguments\n");
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        if (passes == -2)
          fprintf(stderr,
                  "Warning: warm_pca did not settle in %d passes, "
                  "using the last one\n",
                  WARM_PCA_MAX_PASSES);
        else
          fprintf(stderr, "warm_pca,%d passes\n", passes);
        if (oja_copy_from(&oja_live, &oja_avg) != 0) {
          fprintf(stderr, "Error: could not seed the live basis\n");
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        // The projections collected during the warm-up were on Oja's basis;
        // the dictionary must be cut on the basis that will write
        for (int r = 0; r < warm_count; r++)
          project(&oja_avg, &warm_x[(size_t)r * cfg.m],
                  &warmup_proj[r * cfg.k]);
        free(warm_x);
        warm_x = NULL;

        double warm_score = 0.0;
        if (build_dictionary(&cfg, warmup_proj, warm_count, "warmup", perm, ev,
                             bits, alphabet_arr, bkpt, &warm_score) != 0) {
          fprintf(stderr, "Error: Could not build dictionary\n");
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        gram_schmidt(&oja_live);
        gram_schmidt(&oja_avg);

        // Nothing to orient against yet: the incoming basis is the
        // averaged one, and promote_basis only reorders it. oja_cand is free
        // here, staging cannot start before the first dictionary exists
        if (oja_copy_from(&oja_cand, &oja_avg) != 0 ||
            promote_basis(&oja_frozen, &oja_cand, NULL, perm, alphabet_arr,
                          bkpt, angles_scratch, 0, n_seen - 1) != 0) {
          fprintf(stderr, "Error: Could not freeze the basis\n");
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        symbol_basis = &oja_frozen;
        if (record_epoch(&run, epochs_f, &cfg, n_epochs, n_seen, 0, 0.0, 0.0,
                         warm_score, ev, bits, alphabet_arr, bkpt,
                         &oja_frozen) != 0) {
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        pring_reset(&ring);
        have_bkpt = 1;
        /* Epoch 0 opens here. */
        epoch_start = n_seen;
        free(warmup_proj);
        warmup_proj = NULL;
      }
      t_algo += now_seconds() - t0;
      continue;
    }

    // Quantization: convert subspace coordinates into discrete symbols
    digitize(y, bkpt, cfg.k, alphabet_arr, word);
    if (n_lvl) digitize(lvl, lvl_bkpt, n_lvl, lvl_alpha, lvl_word);
    n_words++;

    if (n_seen % cfg.ring_w == 0) pring_refresh(&ring);

    if (ring.count == ring.cap) {
      pring_importances(&ring, ev_live);
      double gap =
          daa_gap(ev_live, cfg.k, cfg.total_bits, cfg.lambda, NULL, bits);
      fprintf(stderr, "gap,%ld,%.6f\n", n_seen - 1, gap);

      /* Type 1: same directions, new split and cut points. Same subspace in
         a new row order, so the angle to the last epoch is 0 and the
         cross-epoch bound pays no slack.
         Checked once per ring_w windows, when the ring was just re-summed,
         so each check sees fresh windows and a near tie cannot flip the
         split back and forth (stationary data: 17 re-splits checking every
         window, 1 checking once per ring).
         Allowed while staging: staging is released only by a type 2, so
         blocking there could block type 1 for the whole run. Row order does
         not affect the angle test, the staleness or the type 2 orientation
         check. */
      if (gap > 0.0 && n_seen % cfg.ring_w == 0) {
        double new_score = 0.0;
        if (build_dictionary(&cfg, ring.buf, ring.count, "realloc", perm, ev,
                             bits, alphabet_arr, bkpt, &new_score) != 0) {
          fprintf(stderr, "Error: could not build the type 1 dictionary\n");
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        // build_dictionary reordered ring's columns in place, so its sums
        // are in the old order until recomputed
        pring_refresh(&ring);
        // Same basis, new row order: reordered in place
        if (promote_basis(&oja_frozen, &oja_frozen, &ring, perm, alphabet_arr,
                          bkpt, angles_scratch, 1, n_seen - 1) != 0) {
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        n_epochs++;
        n_type1++;
        // This window's word was already written with the old split, so
        // the new epoch starts with the next one
        if (record_epoch(&run, epochs_f, &cfg, n_epochs, n_seen, 1, 0.0, 0.0,
                         new_score, ev, bits, alphabet_arr, bkpt,
                         &oja_frozen) != 0) {
          t_algo += now_seconds() - t0;
          goto cleanup;
        }
        fprintf(stderr, "realloc,%ld,%.6f,%.6f\n", n_seen - 1, gap, new_score);
      }
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
    // Always numbers: a level letter is a cell index, not a PCA symbol
    for (int j = 0; j < n_lvl; j++) fprintf(out, ",%d", lvl_word[j]);
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
#ifdef STAGING_A
  fprintf(stderr, "rate     = %.4f deg/window worst smoothed\n",
          worst_rate / SPARTAN_DEG_TO_RAD);
#endif
  fprintf(stderr, "overshoot= %.4f deg worst past the policy when first seen\n",
          worst_over / SPARTAN_DEG_TO_RAD);
  fprintf(stderr,
          "epochs   = %ld after the warm-up one: %ld type 1 (new split), "
          "%ld type 2 (new directions)\n",
          n_epochs, n_type1, n_epochs - n_type1);
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
  if (out && out_owned) fclose(out);
  if (epochs_f) fclose(epochs_f);
  free(win);
  free(xc);
  free(y);
  free(y_cand);
  free(word);
  free(bkpt);
  free(warmup_proj);
  free(warm_x);
#ifndef STAGING_A
  free(win_ring);
#endif
  free(ev);
  free(ev_live);
  free(bits);
  free(alphabet_arr);
  free(angles_scratch);
  free(angles);
  free(advance_scratch);
  free(perm);
  free(lvl);
  free(lvl_warm);
  free(lvl_bkpt);
  free(lvl_alpha);
  free(lvl_word);
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
