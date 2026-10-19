/* Checks that the allocator behaves after the fixes. Not part of the build. */
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

  printf("%-36s rc=%2d", name, rc);
  if (rc == 0) {
    printf("  bits =");
    for (int i = 0; i < k; i++) printf(" %d", bits[i]);
    printf("  score = %.6f", sc);
  }
  printf("   %s\n", ok ? "ok" : "*** FAIL ***");
  if (!ok) fails++;
}

int main(void) {
  double demo[3] = {0.6705, 0.2460, 0.0835};
  double swap[3] = {
      0.6705, 0.0835,
      0.2460}; /* sorted, third and second swapped in VALUE only */
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
  check("6  steeper spectrum still 321", swap, 3, 6, 0.5, 0, b321, NAN);
  puts("");
  printf("%s\n", fails ? "SOME CHECKS FAILED" : "all checks passed");
  return fails != 0;
}
