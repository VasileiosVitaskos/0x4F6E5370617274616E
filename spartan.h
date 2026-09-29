#ifndef SPARTAN_H
#define SPARTAN_H

#include <stddef.h>

/* Online SPARTAN - shared types and the list of things the library can do.
 *
 * Words used all over this file:
 *   m      = how many numbers make up one window
 *   k      = how many directions we keep after PCA; also how many letters a
 * word has ev     = how important each direction is; the k values add up to 1
 *   bits   = how many bits each letter position gets; they add up to the budget
 *   bkpt   = the cut points that turn a number into a letter
 */

#define SPARTAN_EPS 1e-12

/* Biggest alphabet one letter position may have. Also the row stride of the
   cut-point table, so every position has the same amount of room. */
#define MAX_ALPHABET 16

/* ------------------------------------------------------------------ */
/* Settings the caller chooses before anything runs.                   */
/* Things the algorithm works out while running are NOT here.          */
/* ------------------------------------------------------------------ */

typedef struct {
  int k;         /* directions kept / letters per word */
  int m;         /* numbers per window */
  int stride;    /* how far the window moves each step; == m means no overlap */
  double lr;     /* how fast the PCA directions move toward the data */
  int gs_period; /* re-straighten the directions every this many windows */
  int warmup;    /* windows collected before the cut points are fixed */
  int total_bits; /* bit budget for one whole word */
  double lambda;
  int d;
  int m_time;
} Config;

Config config_default(int k, int m);

Config config_default_mv(int k, int d, int m_time);

/* ---------------- basic maths ---------------- */

double dot_product(const double* v1, const double* v2, size_t n);
void znorm_row(double* x, int m);

/* ---------------- running average of the stream ---------------- */

typedef struct {
  double* mean;
  long n;
  int m;
} RunningMean;

int rmean_init(RunningMean* rm, int m);
void rmean_free(RunningMean* rm);
void rmean_update(RunningMean* rm, const double* x);

/* ---------------- the PCA directions ---------------- */

typedef struct {
  double* v; /* k rows of m numbers, flat; row j starts at v[j*m] */
  double* u; /* scratch room, m numbers */
  int k, m;
} OjaPCA;

int oja_init(OjaPCA* p, int k, int m);
void oja_free(OjaPCA* p);
void oja_update(OjaPCA* p, const double* xc, double lr);
void gram_schmidt(OjaPCA* p);

/* How far the directions have drifted from being clean and separate.
 *   max_dot = biggest overlap between any two directions; 0 is perfect
 *   max_len = biggest gap between a direction's length and 1; 0 is perfect
 * The distance guarantee only holds while both are small, so this is the
 * number that says how much slack the guarantee really has. Measure it just
 * BEFORE straightening, which is when the drift is at its worst.
 * Either pointer may be NULL.
 */
void oja_orthonormality_error(const OjaPCA* p, double* max_dot,
                              double* max_len);

/* ---------------- from numbers to letters ---------------- */

void project(const OjaPCA* p, const double* xc, double* y);
void estimate_eigenvalues(const double* proj, int n, int k, double* ev);
void compute_breakpoints(const double* proj, int n, int k, const int* alphabet,
                         double* bkpt);
void digitize(const double* y, const double* bkpt, int k, const int* alphabet,
              int* word);

/* ---------------- splitting the bit budget ---------------- */

/* Work out how many bits each letter position should get.
 *
 * ref_bits: what the split should stay close to.
 *   NULL          -> an even split, which is what spartan.py does.
 *   previous bits -> the split resists moving away from what it already was.
 *                    This second option is our addition, not in the reference.
 *
 * score: how good the chosen split is, or NULL if you do not care.
 *        Needed later to decide whether a change is worth making.
 *
 * Returns 0 on success, -1 if the inputs make no sense or memory runs out.
 */
int daa_allocate(const double* ev, int k, int total_bits, double lamda,
                 const int* ref_bits, int* bits, double* score);

/* How good is a split we already have, judged with these importances?
   Compare it against the best split to see if changing is worth it. */
double daa_score(const double* ev, int k, int total_bits, double lamda,
                 const int* ref_bits, const int* bits);

#endif /* SPARTAN_H */
