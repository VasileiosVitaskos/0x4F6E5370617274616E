/* Tests for the rolling projection ring.
 *
 * The job here is to BREAK the ring, not to agree with it. Every case below
 * exists because there is a specific way the code can be wrong and still look
 * fine: the subtraction never runs, the counter never saturates, the sums get
 * indexed with the wrong subscript, the refresh walks the wrong row.
 *
 * The reference for case 1 is estimate_eigenvalues on the ring contents. Using
 * the real function rather than a second copy of the variance formula means a
 * shared misunderstanding cannot hide: if the two disagree on the degenerate
 * case, that is a finding, not noise.
 *
 *   make test_ring && ./test_ring
 */

#include "spartan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-9

static int failures = 0;

static void report(const char *name, int ok, const char *detail) {
  printf("%-44s %s", name, ok ? "pass" : "FAIL");
  if (detail && *detail) printf("   %s", detail);
  printf("\n");
  /* Unbuffered on purpose. A test that crashes must still have told you how
     far it got, and stdio would swallow the last lines. */
  fflush(stdout);
  if (!ok) failures++;
}

/* A repeatable generator. rand() is not portable across libcs and these
   numbers end up in a pass or fail decision, so the sequence is ours. */
static unsigned long seed = 88172645463325252ull;
static double next_unit(void) {
  seed ^= seed << 13;
  seed ^= seed >> 7;
  seed ^= seed << 17;
  return (double)((seed >> 11) & 0xFFFFFF) / 16777216.0;
}

static double max_abs_diff(const double *a, const double *b, int n) {
  double worst = 0.0;
  for (int i = 0; i < n; i++) {
    double d = fabs(a[i] - b[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

/* ------------------------------------------------------------------ */
/* 1. The incremental sums agree with a full recount, at every step.
 *
 * Runs past 3*cap on purpose. Below cap the ring only fills, so the branch
 * that subtracts the departing value NEVER EXECUTES. A test that stops early
 * passes while testing half the function.
 */
static void case_agreement(void) {
  const int cap = 7, k = 3;
  ProjRing r;
  if (pring_init(&r, cap, k) != 0) { report("1  agreement with full recount", 0, "init failed"); return; }

  double y[3], fast[3], slow[3];
  double worst = 0.0;
  int worst_at = -1;
  int seen_wrap = 0;

  for (int t = 1; t <= 3 * cap; t++) {
    for (int j = 0; j < k; j++) y[j] = (next_unit() - 0.5) * (j + 1);
    pring_push(&r, y);
    if (r.count == r.cap) seen_wrap = 1;

    /* count is about to be used as a length. If it ever passes cap the next
       read runs off the end of buf, so say so here rather than crash there. */
    if (r.count > r.cap || r.count < 1) {
      char bad[80];
      snprintf(bad, sizeof bad, "count %d outside [1, %d] at push %d", r.count, r.cap, t);
      report("1  agreement with full recount", 0, bad);
      pring_free(&r);
      return;
    }

    pring_importances(&r, fast);
    estimate_eigenvalues(r.buf, r.count, r.k, slow);

    double d = max_abs_diff(fast, slow, k);
    if (d > worst) { worst = d; worst_at = t; }
  }

  char detail[96];
  snprintf(detail, sizeof detail, "worst %.2e at push %d", worst, worst_at);
  report("1  agreement with full recount", worst < TOL && seen_wrap, detail);
  pring_free(&r);
}

/* ------------------------------------------------------------------ */
/* 2. The counter saturates and the ring holds the LAST cap values.
 *
 * Catches an unconditional count++, and catches a cursor that does not wrap.
 * The contents are compared as a set, because ring order is not chronological
 * and nothing in the design requires it to be.
 */
static void case_saturation(void) {
  const int cap = 5, k = 1;
  ProjRing r;
  if (pring_init(&r, cap, k) != 0) { report("2  counter saturates, last cap kept", 0, "init failed"); return; }

  for (int t = 0; t < 12; t++) { double v = (double)t; pring_push(&r, &v); }

  int count_ok = (r.count == cap);
  int cursor_ok = (r.cursor >= 0 && r.cursor < cap);

  /* the last five pushed were 7,8,9,10,11 */
  int found[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < cap; i++) {
    int v = (int)r.buf[i];
    if (v >= 7 && v <= 11) found[v - 7] = 1;
  }
  int contents_ok = 1;
  for (int i = 0; i < 5; i++) if (!found[i]) contents_ok = 0;

  char detail[96];
  snprintf(detail, sizeof detail, "count %d cursor %d", r.count, r.cursor);
  report("2  counter saturates, last cap kept",
         count_ok && cursor_ok && contents_ok, detail);
  pring_free(&r);
}

/* ------------------------------------------------------------------ */
/* 3. Ten times the signal gives the same importances.
 *
 * The normalisation is what makes the gap fire on redistribution rather than
 * on loudness. Drop it and this is the case that notices.
 */
static void case_scale_invariance(void) {
  const int cap = 9, k = 4;
  ProjRing a, b;
  pring_init(&a, cap, k);
  pring_init(&b, cap, k);

  double y[4], y10[4], ea[4], eb[4];
  seed = 12345;
  for (int t = 0; t < 2 * cap; t++) {
    for (int j = 0; j < k; j++) { y[j] = next_unit() - 0.5; y10[j] = y[j] * 10.0; }
    pring_push(&a, y);
    pring_push(&b, y10);
  }
  pring_importances(&a, ea);
  pring_importances(&b, eb);

  double d = max_abs_diff(ea, eb, k);
  char detail[64];
  snprintf(detail, sizeof detail, "worst %.2e", d);
  report("3  importances ignore a uniform gain", d < 1e-12, detail);
  pring_free(&a); pring_free(&b);
}

/* ------------------------------------------------------------------ */
/* 4. A signal that never moves gives the uniform answer, and gives the SAME
 * uniform answer as estimate_eigenvalues. Two degenerate conventions in one
 * codebase is a bug that only shows up months later.
 */
static void case_constant(void) {
  const int cap = 6, k = 3;
  ProjRing r;
  pring_init(&r, cap, k);

  double y[3] = {2.5, -1.0, 0.0};
  for (int t = 0; t < 2 * cap; t++) pring_push(&r, y);

  double fast[3], slow[3];
  pring_importances(&r, fast);
  estimate_eigenvalues(r.buf, r.count, r.k, slow);

  int uniform_ok = 1;
  for (int j = 0; j < k; j++) if (fabs(fast[j] - 1.0 / k) > 1e-15) uniform_ok = 0;

  char detail[80];
  snprintf(detail, sizeof detail, "ev0 %.6f, agrees %.1e", fast[0], max_abs_diff(fast, slow, k));
  report("4  constant signal gives uniform, same rule",
         uniform_ok && max_abs_diff(fast, slow, k) < 1e-15, detail);
  pring_free(&r);
}

/* ------------------------------------------------------------------ */
/* 5. Reading a ring nobody pushed to. Must not divide by zero and must not
 * read the uninitialised part of the buffer.
 */
static void case_empty(void) {
  const int cap = 4, k = 3;
  ProjRing r;
  pring_init(&r, cap, k);

  double ev[3] = {-1.0, -1.0, -1.0};
  pring_importances(&r, ev);

  int ok = 1;
  for (int j = 0; j < k; j++) {
    if (!(ev[j] == ev[j])) ok = 0;              /* NaN */
    if (fabs(ev[j] - 1.0 / k) > 1e-15) ok = 0;
  }
  char detail[64];
  snprintf(detail, sizeof detail, "ev0 %.6f", ev[0]);
  report("5  empty ring reads as uniform", ok, detail);
  pring_free(&r);
}

/* ------------------------------------------------------------------ */
/* 6. The smallest shapes the contract allows, plus the ones it forbids. */
static void case_edges(void) {
  ProjRing r;

  /* k = 1: one position carries everything, so its share is exactly one */
  pring_init(&r, 3, 1);
  for (int t = 0; t < 7; t++) { double v = next_unit(); pring_push(&r, &v); }
  double ev1[1];
  pring_importances(&r, ev1);
  int k1_ok = fabs(ev1[0] - 1.0) < 1e-15;
  pring_free(&r);

  /* cap = 2: wraps on every other push */
  pring_init(&r, 2, 2);
  double y[2];
  for (int t = 0; t < 9; t++) { y[0] = next_unit(); y[1] = next_unit(); pring_push(&r, y); }
  double fast[2], slow[2];
  pring_importances(&r, fast);
  estimate_eigenvalues(r.buf, r.count, r.k, slow);
  int cap2_ok = (r.count == 2) && max_abs_diff(fast, slow, 2) < TOL;
  pring_free(&r);

  /* arguments the contract rejects */
  int reject_ok = 1;
  if (pring_init(&r, 1, 3) == 0) { reject_ok = 0; pring_free(&r); }   /* cap below 2 */
  if (pring_init(&r, 8, 0) == 0) { reject_ok = 0; pring_free(&r); }   /* k below 1  */
  if (pring_init(NULL, 8, 3) == 0) reject_ok = 0;

  char detail[80];
  snprintf(detail, sizeof detail, "k1 %d  cap2 %d  reject %d", k1_ok, cap2_ok, reject_ok);
  report("6  k=1, cap=2, and rejected arguments", k1_ok && cap2_ok && reject_ok, detail);
}

/* ------------------------------------------------------------------ */
/* 7. Free must survive a struct that never saw init, and a second call.
 * main has one cleanup label and reaches it from every failure path, so this
 * is not hypothetical.
 */
static void case_free_safety(void) {
  ProjRing zeroed;
  memset(&zeroed, 0, sizeof zeroed);
  pring_free(&zeroed);          /* never initialised */
  pring_free(&zeroed);          /* twice */

  ProjRing r;
  pring_init(&r, 4, 2);
  pring_free(&r);
  pring_free(&r);               /* twice after a real init */
  pring_free(NULL);

  report("7  free is safe on zeroed, twice, and NULL", 1, "no crash under asan");
}

/* ------------------------------------------------------------------ */
/* 8. The long run, and the reason refresh exists.
 *
 * Two streams with the same spread, one sitting near zero and one sitting on
 * a large offset. The offset one is where sum-of-squares minus square-of-mean
 * loses its digits. The hard failure is a negative variance; the drift itself
 * is reported so the number is visible rather than assumed.
 */
static void drift_run(const char *tag, double offset, long pushes, int *neg_seen,
                      double *rel_drift) {
  const int cap = 900, k = 4;
  ProjRing r;
  pring_init(&r, cap, k);

  double y[4];
  *neg_seen = 0;

  for (long t = 0; t < pushes; t++) {
    for (int j = 0; j < k; j++) y[j] = offset + (next_unit() - 0.5) * (j + 1);
    pring_push(&r, y);

    if (t % 1000 == 0 && r.count > 1) {
      for (int j = 0; j < k; j++) {
        double mean = r.s1[j] / r.count;
        if (r.s2[j] / r.count - mean * mean < 0.0) *neg_seen = 1;
      }
    }
  }

  /* what the incremental sums believe, against a clean recount of the same
     contents: the gap between them is the accumulated error */
  double before[4];
  for (int j = 0; j < k; j++) before[j] = r.s2[j];
  pring_refresh(&r);

  double worst = 0.0;
  for (int j = 0; j < k; j++) {
    double denom = fabs(r.s2[j]) > 1.0 ? fabs(r.s2[j]) : 1.0;
    double e = fabs(before[j] - r.s2[j]) / denom;
    if (e > worst) worst = e;
  }
  *rel_drift = worst;
  printf("      %-22s relative drift in sum of squares: %.3e\n", tag, worst);
  pring_free(&r);
}

static void case_long_run(void) {
  int neg_a, neg_b;
  double drift_a, drift_b;
  const long pushes = 300000;

  drift_run("values near zero", 0.0, pushes, &neg_a, &drift_a);
  drift_run("values near 1e6", 1.0e6, pushes, &neg_b, &drift_b);

  char detail[96];
  snprintf(detail, sizeof detail, "%ld pushes each, no negative variance", pushes);
  report("8  long run keeps variance non negative", !neg_a && !neg_b, detail);
}

/* ------------------------------------------------------------------ */
/* 9. Refresh rebuilds what push maintained. Catches a refresh that walks the
 * cursor row instead of row i, which leaves every sum equal to cap copies of
 * one window and is invisible to case 1.
 */
static void case_refresh_matches(void) {
  const int cap = 11, k = 3;
  ProjRing r;
  pring_init(&r, cap, k);

  double y[3];
  seed = 999;
  for (int t = 0; t < 3 * cap + 4; t++) {
    for (int j = 0; j < k; j++) y[j] = (next_unit() - 0.5) * (j + 2);
    pring_push(&r, y);
  }

  double s1_before[3], s2_before[3];
  memcpy(s1_before, r.s1, sizeof s1_before);
  memcpy(s2_before, r.s2, sizeof s2_before);

  pring_refresh(&r);

  double d1 = max_abs_diff(s1_before, r.s1, k);
  double d2 = max_abs_diff(s2_before, r.s2, k);

  /* and a second refresh must change nothing at all */
  double s1_once[3], s2_once[3];
  memcpy(s1_once, r.s1, sizeof s1_once);
  memcpy(s2_once, r.s2, sizeof s2_once);
  pring_refresh(&r);
  double idem = max_abs_diff(s1_once, r.s1, k) + max_abs_diff(s2_once, r.s2, k);

  char detail[96];
  snprintf(detail, sizeof detail, "s1 %.1e  s2 %.1e  idempotent %.1e", d1, d2, idem);
  report("9  refresh matches push, and is idempotent",
         d1 < TOL && d2 < TOL && idem == 0.0, detail);
  pring_free(&r);
}

/* ------------------------------------------------------------------ */
/* 10. Refresh on a half filled ring reads only what is there.
 *
 * While count < cap the cursor has never wrapped, so the live rows are exactly
 * 0 .. count-1 and the rest are still zeros from calloc. A refresh that walks
 * all cap rows folds those zeros in and quietly shrinks every variance.
 */
static void case_refresh_partial(void) {
  const int cap = 10, k = 2;
  ProjRing r;
  pring_init(&r, cap, k);

  double y[2];
  seed = 4242;
  for (int t = 0; t < 4; t++) {                 /* four of ten slots used */
    for (int j = 0; j < k; j++) y[j] = 5.0 + next_unit();
    pring_push(&r, y);
  }

  double before[2];
  pring_importances(&r, before);
  pring_refresh(&r);
  double after[2];
  pring_importances(&r, after);

  char detail[64];
  snprintf(detail, sizeof detail, "count %d, worst %.1e", r.count, max_abs_diff(before, after, k));
  report("10 refresh on a half filled ring",
         max_abs_diff(before, after, k) < TOL, detail);
  pring_free(&r);
}

/* ------------------------------------------------------------------ */

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("ring tests\n");
  printf("----------------------------------------------------------------\n");
  case_saturation();
  case_agreement();
  case_scale_invariance();
  case_constant();
  case_empty();
  case_edges();
  case_free_safety();
  case_long_run();
  case_refresh_matches();
  case_refresh_partial();
  printf("----------------------------------------------------------------\n");
  if (failures == 0) {
    printf("all pass\n");
    return 0;
  }
  printf("%d FAILED\n", failures);
  return 1;
}
