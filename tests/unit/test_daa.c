/* Tests for the bit allocator and its staleness test.
 *
 * The job here is to BREAK the allocator, not to agree with it. The greedy
 * auction is claimed to be the exact optimum under one condition (every bit
 * bought at a non-negative marginal gain, daa_exact), to keep the split
 * non-increasing on descending importances, and the bid-ask gap is claimed
 * to decide optimality exactly. Each case below attacks one of those claims
 * where it is most likely to fail: at ties, at zero importances, at the edge
 * of the condition, and against an exhaustive search that knows nothing
 * about greedy.
 *
 *   make unit
 */
#include <math.h>
#include <stdio.h>

#include "spartan.h"

static int fails = 0;

static void check(const char* name, const double* ev, int k, int N, double lam,
                  int want_rc, const int* want_bits, double want_score) {
  int bits[16];
  double sc = -12345.0;
  int rc = daa_allocate(ev, k, N, lam, NULL, bits, &sc);

  int ok = (rc == want_rc);
  if (rc == 0 && want_bits)
    for (int i = 0; i < k; i++)
      if (bits[i] != want_bits[i]) ok = 0;
  if (rc == 0 && !isnan(want_score) && fabs(sc - want_score) > 1e-6) ok = 0;

  printf("%-40s rc=%2d", name, rc);
  if (rc == 0) {
    printf("  bits =");
    for (int i = 0; i < k; i++) printf(" %d", bits[i]);
    printf("  score = %.6f", sc);
  }
  printf("   %s\n", ok ? "ok" : "*** FAIL ***");
  fflush(stdout);
  if (!ok) fails++;
}

static void report(const char* name, int ok, const char* detail) {
  printf("%-40s %s   %s\n", name, ok ? "ok" : "*** FAIL ***", detail);
  fflush(stdout);
  if (!ok) fails++;
}

/* A repeatable generator: rand() differs across libcs, and these numbers
   decide pass or fail */
static unsigned long long seed = 88172645463325252ull;
static double next_unit(void) {
  seed ^= seed << 13;
  seed ^= seed >> 7;
  seed ^= seed << 17;
  return (double)((seed >> 11) & 0xFFFFFF) / 16777216.0;
}

/* k importances, normalised, sorted descending. mode 1 zeroes some, mode 2
   makes them all equal, mode 3 quantises them so that ties are common */
static void random_ev(double* ev, int k, int mode) {
  double s = 0.0;
  for (int j = 0; j < k; j++) {
    double v = next_unit();
    if (mode == 1 && next_unit() < 0.4) v = 0.0;
    if (mode == 2) v = 1.0;
    if (mode == 3) v = floor(v * 4.0) / 4.0;
    ev[j] = v;
    s += v;
  }
  if (s == 0.0) {
    ev[0] = 1.0;
    s = 1.0;
  }
  for (int j = 0; j < k; j++) ev[j] /= s;
  for (int i = 0; i < k; i++) /* insertion sort, descending */
    for (int j = i; j > 0 && ev[j] > ev[j - 1]; j--) {
      double t = ev[j];
      ev[j] = ev[j - 1];
      ev[j - 1] = t;
    }
}

/* The best score over EVERY split with 0 <= b_j <= MAX_BITS and sum N, by
   enumeration: no greedy, no order, no constraint. If greedy is the exact
   optimum it must reach this */
static double brute_best(const double* ev, int k, int N, double lam) {
  int b[16] = {0};
  double best = -1e300;
  for (;;) {
    int s = 0;
    for (int j = 0; j < k; j++) s += b[j];
    if (s == N) {
      double sc = daa_score(ev, k, N, lam, NULL, b);
      if (sc > best) best = sc;
    }
    int j = 0; /* next vector, odometer style */
    while (j < k && b[j] == MAX_BITS) b[j++] = 0;
    if (j == k) break;
    b[j]++;
  }
  return best;
}

static int non_increasing(const int* b, int k) {
  for (int j = 1; j < k; j++)
    if (b[j] > b[j - 1]) return 0;
  return 1;
}

/* ------------------------------------------------------------------ */
/* 7. Greedy against exhaustive search, inside the condition.
 *
 * Three claims at once: greedy reaches the unconstrained optimum, its split
 * is non-increasing, and its score is daa_score of its own bits. Random
 * spectra with zeros, full ties and quantised ties, every lambda the
 * condition allows.
 */
static void case_brute(void) {
  const double lams[] = {0.0, 0.1, 0.5, 0.9, 1.0, 1.5, 3.0};
  int cases = 0, worse = 0, order = 0, selfscore = 0, refused = 0;
  seed = 2026;
  for (int t = 0; t < 1500; t++) {
    int k = 1 + (int)(next_unit() * 5); /* 1 .. 5 */
    int N = k + (int)(next_unit() * (k * MAX_BITS - k + 1));
    double lam = lams[(int)(next_unit() * 7)];
    double ev[16];
    random_ev(ev, k, (int)(next_unit() * 4));
    if (!daa_exact(k, N, lam, NULL)) continue; /* case 8 */
    cases++;
    int b[16];
    double sc;
    if (daa_allocate(ev, k, N, lam, NULL, b, &sc) != 0) {
      refused++;
      continue;
    }
    if (sc < brute_best(ev, k, N, lam) - 1e-12) worse++;
    if (!non_increasing(b, k)) order++;
    if (fabs(sc - daa_score(ev, k, N, lam, NULL, b)) > 1e-12) selfscore++;
  }
  char d[120];
  snprintf(d, sizeof d,
           "%d cases: worse %d, not non-increasing %d, "
           "score mismatch %d, refused %d",
           cases, worse, order, selfscore, refused);
  report("7  greedy = exhaustive optimum",
         cases > 500 && !worse && !order && !selfscore && !refused, d);
}

/* ------------------------------------------------------------------ */
/* 8. Outside the condition the allocator refuses, and the config would
 * have refused first. lambda > 1 with a budget that k does not divide is
 * exactly where greedy hands the leftover bits to the weakest positions
 */
static void case_condition(void) {
  int ok = 1;
  ok &= !daa_exact(3, 7, 1.5, NULL);   /* 7 = 3*2 + 1, leftover at a loss */
  ok &= !daa_exact(8, 17, 1.01, NULL); /* just past 1 */
  ok &= daa_exact(8, 17, 1.0, NULL);   /* at 1 the leftover bid is exactly 0 */
  ok &= daa_exact(8, 16, 3.0, NULL);   /* divisible: every position needs ref */
  ok &= daa_exact(5, 12, 0.0, NULL);   /* no penalty at all */
  double ev[3] = {0.5, 0.3, 0.2};
  int b[3];
  ok &= (daa_allocate(ev, 3, 7, 1.5, NULL, b, NULL) == -1);
  report("8  condition boundary, and refusal", ok,
         "1.5/7 no, 1.01/17 no, 1/17 "
         "yes, 3/16 yes, 0/12 yes");
}

/* ------------------------------------------------------------------ */
/* 9. Exact ties at lambda = 1. The bid at ref is 1 - lambda = 0 for every
 * position; computed as a difference of two scores it came out +-1e-17 and
 * the ties were decided by rounding noise, which gave splits that were not
 * non-increasing. These spectra are ones that failed that way
 */
static void case_lambda_one_ties(void) {
  double a[8] = {0.1765, 0.1756, 0.1673, 0.1333,
                 0.1288, 0.1144, 0.0599, 0.0442};
  double c[8] = {0.3181, 0.3165, 0.2239, 0.1160, 0.0255, 0.0, 0.0, 0.0};
  int b[8], ok = 1;
  double sc;
  ok &= daa_allocate(a, 8, 21, 1.0, NULL, b, &sc) == 0 && non_increasing(b, 8);
  ok &= daa_allocate(c, 8, 22, 1.0, NULL, b, &sc) == 0 && non_increasing(b, 8);
  /* and the gap at the optimum must be exactly 0, not 1e-17 */
  daa_allocate(a, 8, 21, 1.0, NULL, b, &sc);
  ok &= daa_gap(a, 8, 21, 1.0, NULL, b) == 0.0;
  report("9  lambda = 1 ties stay ordered, gap 0", ok,
         "two spectra that "
         "broke the score-difference bids");
}

/* ------------------------------------------------------------------ */
/* 10. Zero importances. Every bid of a dead position is exactly 0, at every
 * level, so a lowest-index tie rule would pile the leftover bits on the
 * first dead position and leave a live one below it
 */
static void case_dead_positions(void) {
  double ev[8] = {0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double all0[4] = {0.0, 0.0, 0.0, 0.0};
  int b[8], ok = 1;
  ok &=
      daa_allocate(ev, 8, 16, 0.5, NULL, b, NULL) == 0 && non_increasing(b, 8);
  int sum = 0;
  ok &=
      daa_allocate(all0, 4, 9, 0.5, NULL, b, NULL) == 0 && non_increasing(b, 4);
  for (int j = 0; j < 4; j++) sum += b[j];
  ok &= (sum == 9);
  report("10 dead positions keep the order", ok,
         "two live of eight, and all "
         "four dead");
}

/* ------------------------------------------------------------------ */
/* 11. The gap decides optimality exactly, in BOTH directions.
 *
 * For random splits (not from the allocator, not ordered): gap <= 0 must
 * hold exactly when the split's score equals the exhaustive optimum. A gap
 * that is 0 on a suboptimal split is the dangerous direction: the dictionary
 * would never be rebuilt
 */
static void case_gap_exact(void) {
  const double lams[] = {0.0, 0.25, 0.5, 1.0, 2.0};
  int cases = 0, missed = 0, false_alarm = 0;
  seed = 777;
  for (int t = 0; t < 3000; t++) {
    int k = 1 + (int)(next_unit() * 5);
    int N = k + (int)(next_unit() * (k * MAX_BITS - k + 1));
    double lam = lams[(int)(next_unit() * 5)];
    if (!daa_exact(k, N, lam, NULL)) continue;
    double ev[16];
    random_ev(ev, k, (int)(next_unit() * 4));
    int b[16] = {0}; /* a random split with sum N inside the box */
    for (int u = 0; u < N; u++) {
      int j;
      do j = (int)(next_unit() * k);
      while (b[j] == MAX_BITS);
      b[j]++;
    }
    cases++;
    double best = brute_best(ev, k, N, lam);
    int optimal = daa_score(ev, k, N, lam, NULL, b) >= best - 1e-12;
    int quiet = daa_gap(ev, k, N, lam, NULL, b) == 0.0;
    if (quiet && !optimal) missed++;
    if (!quiet && optimal) false_alarm++;
  }
  char d[120];
  snprintf(d, sizeof d,
           "%d random splits: suboptimal but quiet %d, optimal "
           "but firing %d",
           cases, missed, false_alarm);
  report("11 gap = 0 exactly when optimal",
         cases > 1000 && !missed && !false_alarm, d);
}

/* ------------------------------------------------------------------ */
/* 12. Arguments the contract rejects, and the smallest shapes it allows */
static void case_edges(void) {
  double one[1] = {1.0};
  double ev[3] = {0.6, 0.3, 0.1};
  int b[3], ok = 1;
  ok &= daa_allocate(one, 1, 3, 0.5, NULL, b, NULL) == 0 && b[0] == 3;
  ok &= daa_allocate(one, 1, MAX_BITS + 1, 0.5, NULL, b, NULL) == -1;
  ok &= daa_allocate(ev, 3, 0, 0.5, NULL, b, NULL) == -1;
  ok &= daa_allocate(ev, 0, 3, 0.5, NULL, b, NULL) == -1;
  ok &= daa_allocate(NULL, 3, 6, 0.5, NULL, b, NULL) == -1;
  ok &= daa_allocate(ev, 3, 6, 0.5, NULL, NULL, NULL) == -1;
  report("12 k=1, and rejected arguments", ok,
         "k=1 takes the budget; N=0, "
         "k=0, NULLs and N>k*MAX_BITS refused");
}

int main(void) {
  double demo[3] = {0.6705, 0.2460, 0.0835};
  double swap[3] = {0.6705, 0.0835, 0.2460}; /* second and third swapped */
  double flat[3] = {0.45, 0.42, 0.13};
  double eight[8];
  for (int i = 0; i < 8; i++) eight[i] = 0.125;
  double dead[3] = {0.70, 0.29, 0.01};
  double tiny[3] = {0.80, 0.15, 0.05};

  int b321[3] = {3, 2, 1};
  int b330[3] = {3, 3, 0};
  int b8[8] = {2, 2, 1, 1, 1, 1, 1, 1};
  int b210[3] = {2, 1, 0};

  puts("");
  check("1  demo spectrum, N=6", demo, 3, 6, 0.5, 0, b321, 2.210000);
  check("2  ceiling no longer binds", flat, 3, 6, 0.5, 0, b321, 2.030000);
  check("3  flat k=8 N=10 now feasible", eight, 8, 10, 0.5, 0, b8, 1.125000);
  check("3b budget past k*MAX_BITS fails", demo, 3, 20, 0.5, -1, NULL, NAN);
  check("4  dead component drops out", dead, 3, 6, 0.5, 0, b330, NAN);
  check("5  N==k shortcut is gone", tiny, 3, 3, 0.5, 0, b210, 1.325000);
  /* Unsorted importances are refused: descending order is the precondition
     of the proof, and build_dictionary sorts before it calls. The old DP
     returned 3 2 1 here, greedy would return 3 1 2, the optimum for this
     order; neither is the split of the paper */
  check("6  unsorted importances refused", swap, 3, 6, 0.5, -1, NULL, NAN);
  case_brute();
  case_condition();
  case_lambda_one_ties();
  case_dead_positions();
  case_gap_exact();
  case_edges();
  puts("");
  printf("%s\n", fails ? "SOME CHECKS FAILED" : "all checks passed");
  return fails != 0;
}
