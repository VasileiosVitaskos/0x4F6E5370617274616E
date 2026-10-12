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
#define MAX_BITS (__builtin_ctz(MAX_ALPHABET))

/* ------------------------------------------------------------------ */
/* Settings the caller chooses before anything runs.                   */
/* Things the algorithm works out while running are NOT here.          */
/* ------------------------------------------------------------------ */

typedef struct {
  int k;         /* directions kept / letters per word */
  int m;         /* numbers per window */
  int stride;    /* how far the window moves each step; == m means no overlap */
  double step;   /* step size for the PCA directions, BEFORE dividing by m */
  double beta;   /* how fast the averaged basis forgets; 0 turns it off */
  double lr;     /* how fast the PCA directions move toward the data */
  int gs_period; /* re-straighten the directions every this many windows */
  int warmup;    /* windows collected before the cut points are fixed */
  int total_bits; /* bit budget for one whole word */
  double lambda;
  int d;
  int m_time;
  int ring_w;
} Config;

Config config_default(int k, int m);

Config config_default_mv(int k, int d, int m_time);

/* ---------------- basic maths ---------------- */

double dot_product(const double *v1, const double *v2, size_t n);
void znorm_row(double *x, int m);

/* ---------------- running average of the stream ---------------- */

typedef struct {
  double *mean;
  long n;
  int m;
} RunningMean;

int rmean_init(RunningMean *rm, int m);
void rmean_free(RunningMean *rm);
void rmean_update(RunningMean *rm, const double *x);

/* ---------------- the last W projections ---------------- */

/* Keeps the most recent W projections so a new dictionary can be built at any
 * moment, and so the importances can be recomputed without a second pass.
 *
 * Layout of buf is the same as the warm-up buffer in main: window r, position
 * j sits at buf[r*k + j]. estimate_eigenvalues and compute_breakpoints read
 * that layout already, so they work on this without a change.
 *
 * s1 and s2 are the running sum and the running sum of squares, one per
 * position, over whatever buf currently holds. They are what makes a refresh
 * cost k operations instead of W*k.
 */
typedef struct {
  double *buf; /* cap rows of k numbers, flat, in ring order */
  double *s1;  /* k sums */
  double *s2;  /* k sums of squares */
  int cursor;  /* where the next projection is written, in [0, cap) */
  int count;   /* how many are in, saturating at cap */
  int cap;     /* W */
  int k;
} ProjRing;

int pring_init(ProjRing *r, int cap, int k);
void pring_free(ProjRing *r);
void pring_push(ProjRing *r, const double *y);
void pring_importances(const ProjRing *r, double *ev);
void pring_refresh(ProjRing *r);
void pring_reset(ProjRing *r);

/* ---------------- the PCA directions ---------------- */

typedef struct {
  double *v; /* k rows of m numbers, flat; row j starts at v[j*m] */
  double *u; /* scratch room, m numbers */
  int k, m;
} OjaPCA;

int oja_init(OjaPCA *p, int k, int m);
void oja_free(OjaPCA *p);
void oja_update(OjaPCA *p, const double *xc, double lr);
int oja_blend(OjaPCA *dst, const OjaPCA *src, double beta);
void gram_schmidt(OjaPCA *p);

/* Returns 0 on success, -1 if either pointer is NULL or either side holds no
 * direction array, and -2 if both are initialised but the shapes differ.
 * The two are kept apart on purpose: -1 is "you handed me nothing", -2 is
 * "you handed me the wrong thing", and only the second tells you to go look
 * at k and m. */
int oja_copy_from(OjaPCA *dst, const OjaPCA *src);

/* How far the directions have drifted from being clean and separate.
 *   max_dot = biggest overlap between any two directions; 0 is perfect
 *   max_len = biggest gap between a direction's length and 1; 0 is perfect
 * The distance guarantee only holds while both are small, so this is the
 * number that says how much slack the guarantee really has. Measure it just
 * BEFORE straightening, which is when the drift is at its worst.
 * Either pointer may be NULL.
 */
void oja_orthonormality_error(const OjaPCA *p, double *max_dot,
                              double *max_len);

int svd_values(double *a, int n, double *sv);

/* The principal angles between the subspace the frozen basis spans and the one
 * the live basis spans. The k numbers are the angles themselves, in radians.
 *
 * Both bases must be orthonormal for these to BE angles, so call this right
 * after a straightening and not at an arbitrary window. The clamp below
 * handles the last bit of rounding; it cannot rescue a basis whose rows are a
 * percent off unit length.
 *
 * frozen, live: same k and same m. Neither is modified.
 * scratch: k*(k+1) doubles owned by the caller, OVERWRITTEN. The first k*k
 *          hold the matrix, the last k are where the singular values land.
 *          They are kept apart on purpose: letting them share would work
 *          today and break silently once svd_values also produces factors.
 *          Allocated once by the caller rather than here, because this runs
 *          inside a stream.
 * angles:  k doubles, or NULL. In NO particular order, and the order carries
 *          no meaning: a principal angle belongs to the PAIR of subspaces, not
 *          to one component, so "angle 2 is large" says nothing about
 *          component 2. Naming which direction moved needs the factors, which
 *          svd_values does not produce yet.
 * max_angle: the largest of them, or NULL. This is the one a guarantee is
 *          stated about: a word is wrong if ANY position is wrong, so the
 *          binding quantity is the worst aligned direction, never an average.
 *
 * Returns 0 on success, -1 on NULL arguments or mismatched shapes, and -2 if
 * the singular values did not converge. On anything but 0 nothing is written.
 */
int subspace_angles(const OjaPCA *frozen, const OjaPCA *live, double *scratch,
                    double *angles, double *max_angle);

/* ---------------- from numbers to letters ---------------- */

void project(const OjaPCA *p, const double *xc, double *y);
void estimate_eigenvalues(const double *proj, int n, int k, double *ev);
void compute_breakpoints(const double *proj, int n, int k, const int *alphabet,
                         double *bkpt);
void digitize(const double *y, const double *bkpt, int k, const int *alphabet,
              int *word);

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
int daa_allocate(const double *ev, int k, int total_bits, double lamda,
                 const int *ref_bits, int *bits, double *score);

/* How good is a split we already have, judged with these importances?
   Compare it against the best split to see if changing is worth it. */
double daa_score(const double *ev, int k, int total_bits, double lamda,
                 const int *ref_bits, const int *bits);

double daa_gap(const double *ev, int k, int total_bits, double lamda,
               const int *bits);

#endif /* SPARTAN_H */
