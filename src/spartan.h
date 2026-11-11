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
#define SPARTAN_DEG_TO_RAD 0.017453292519943295
/* Biggest alphabet one letter position may have. Also the row stride of the
   cut-point table, so every position has the same amount of room. */
#define MAX_ALPHABET 16
#define MAX_BITS (__builtin_ctz(MAX_ALPHABET))
/* Cells for each level letter (mean or spread of one channel). Fixed, and
   outside the DAA budget: these letters are not PCA directions and have no
   eigenvalue to compete with. Must be <= MAX_ALPHABET. */
#define LEVEL_ALPHABET 4
/* warm_pca stops when the span moves less than this between two passes,
   measured as k - ||V_new V_old^T||_F^2, or after this many passes */
#define WARM_PCA_TOL 1e-13
#define WARM_PCA_MAX_PASSES 1000

/* ------------------------------------------------------------------ */
/* Settings the caller chooses before anything runs.                   */
/* Things the algorithm works out while running are NOT here.          */
/* ------------------------------------------------------------------ */

typedef struct {
  int k;       /* directions kept / letters per word */
  int m;       /* numbers per window */
  int stride;  /* how far the window moves each step; == m means no overlap */
  double step; /* step size for the PCA directions, BEFORE dividing by m */
  double beta; /* how fast the averaged basis forgets; 0 turns it off */
  double lr;   /* how fast the PCA directions move toward the data */
  double theta_threshold_deg; /* epoch closes past this angle, as the user
                             wrote it, in DEGREES. Kept so the dump can
                             print back exactly what came in */
  double theta_threshold;     /* the same angle in radians, worked out once.
                                 This is what the loop compares against, and
                                 the only form that may be compared with
                                 anything acos returned */
  int gs_period;    /* re-straighten the directions every this many windows */
  int angle_period; /* windows between angle readings; MUST be a multiple of
                       gs_period, because the inner products are only cosines
                       of principal angles on a basis that was just
                       straightened */
  int warmup;       /* windows collected before the cut points are fixed */
  int total_bits;   /* bit budget for one whole word */
  double lambda; /* L2 penalty weight for deviating from reference bit split */
  int d;         /* input channels (variables per time frame) */
  int m_time;    /* time frames per window (m = d * m_time) */
  int ring_w;    /* projection ring capacity W for dictionary rebuilds */
  double freeze_ahead; /* how far past the data to freeze a candidate, in
                        units of theta_threshold. 0 turns it off and the
                        candidate is a plain snapshot, as before */
  int cross_epoch;     /* how a LIVE query is bounded against a word of an older
                          epoch: CROSS_EPOCH_SEARCH or CROSS_EPOCH_BOX. A join
                          over stored words has no window and is always box */
  int level_letters;   /* 1 = append each channel's mean and spread, taken
                          BEFORE z-norm, as two extra letters (2*d in all).
                          0 = off, the default: words are exactly as before.
                          z-norm erases a level shift or an amplitude jump, so
                          this is for anomaly detection, not for similarity
                          search, where z-normed distance is the target */
} Config;

/* Values of Config.cross_epoch. Both are lower bounds (proofs in the paper):
 *   SEARCH  project the query window again onto the older epoch's basis and
 *           cut it with that epoch's dictionary; then the two words share an
 *           epoch and the ordinary MINDIST applies. No slack. Cost: k*m
 *           multiply-adds per query per older epoch, not per candidate
 *   BOX     carry the query's cell box into the older epoch through the
 *           epoch map and take off sin(theta_max) * r. Cost: k*k per query
 *           per older epoch, but loose at large angles */
#define CROSS_EPOCH_SEARCH 0
#define CROSS_EPOCH_BOX 1
/**
 * @brief Constructs a default univariate SPARTAN configuration for a given
 * subspace and window size.
 *
 * Convenience wrapper around `config_default_mv` with channel count `d = 1`.
 *
 * @param k Number of principal directions (and word symbols) to retain.
 * @param m Number of scalar samples per univariate window.
 * @return Initialized Config structure with default hyperparameters.
 */
Config config_default(int k, int m);
/**
 * @brief Constructs a default multivariate SPARTAN configuration.
 *
 * Initializes flattened dimensionality `m = d * m_time`, scales the effective
 * Oja learning rate `lr = step / m`, converts the angular drift policy to
 * radians, and sets baseline warmup and Gram-Schmidt periods.
 *
 * @param k Number of principal directions (and word symbols) to retain.
 * @param d Number of input channels.
 * @param m_time Number of time frames per window.
 * @return Initialized Config structure with default multivariate
 * hyperparameters.
 */
Config config_default_mv(int k, int d, int m_time);

/* ---------------- basic maths ---------------- */
/**
 * @brief Computes the Euclidean inner product of two contiguous
 * double-precision vectors.
 *
 * Uses 4-way loop unrolling with independent accumulators to break
 * floating-point dependency chains and maximize SIMD instruction-level
 * parallelism.
 *
 * @param v1 Pointer to the first input vector of length n.
 * @param v2 Pointer to the second input vector of length n.
 * @param n Number of elements in each vector.
 * @return The scalar dot product of v1 and v2.
 */
double dot_product(const double* v1, const double* v2, size_t n);
/**
 * @brief Applies in-place population z-score normalization to a single channel
 * vector.
 *
 * Subtracts the sample mean and divides by the population standard deviation
 * (using denominator `m` and `SPARTAN_EPS` regularization to match the
 * reference batch implementation).
 *
 * @param x Pointer to the contiguous array of m elements to normalize in-place.
 * @param m Number of elements in the array.
 * @return void
 */
void znorm_row(double* x, int m);
/**
 * @brief The mean and the population spread of one channel of one window,
 * the two numbers znorm_row erases. Call BEFORE znorm_row.
 *
 * @param x Pointer to m values, not yet z-normed (read only).
 * @param m Number of values.
 * @param mu Output, the mean [units of x].
 * @param sd Output, the population standard deviation [units of x].
 * @return void
 */
void window_level(const double* x, int m, double* mu, double* sd);

/* ---------------- running average of the stream ---------------- */

typedef struct {
  double* mean;
  long n;
  int m;
} RunningMean;
/**
 * @brief Allocates and zeroes a streaming Welford running mean accumulator.
 *
 * Prepares an m-dimensional state vector to track the global stream center
 * online.
 *
 * @param rm Pointer to the RunningMean structure to initialize.
 * @param m Flattened window dimensionality (d * m_time).
 * @return 0 on success, or -1 if m <= 0 or memory allocation fails.
 */
int rmean_init(RunningMean* rm, int m);
/**
 * @brief Frees heap memory owned by a RunningMean accumulator and resets its
 * state.
 *
 * Safely releases the internal mean vector and zeroes counters.
 *
 * @param rm Pointer to the RunningMean structure to deallocate.
 * @return void
 */
void rmean_free(RunningMean* rm);
/**
 * @brief Updates the streaming mean vector in-place with a newly observed
 * window.
 *
 * Applies the incremental recurrence `mean += (x - mean) / n` in O(m) time
 * without storing historical windows.
 *
 * @param rm Pointer to the active RunningMean accumulator.
 * @param x Pointer to the newly observed window vector of length m.
 * @return void
 */
void rmean_update(RunningMean* rm, const double* x);

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
  double* buf; /* cap rows of k numbers, flat, in ring order */
  double* s1;  /* k sums */
  double* s2;  /* k sums of squares */
  int cursor;  /* where the next projection is written, in [0, cap) */
  int count;   /* how many are in, saturating at cap */
  int cap;     /* W */
  int k;
} ProjRing;
/**
 * @brief Allocates a circular buffer and O(k) sufficient statistics trackers
 * for subspace projections.
 *
 * Maintains the most recent `cap` k-dimensional projections along with running
 * first (`s1`) and second (`s2`) moment vectors so variance shares can be
 * queried in O(k) time and full dictionary rebuilds can occur on demand.
 *
 * @param r Pointer to the ProjRing structure to initialize.
 * @param cap Window capacity W of the circular buffer (must be >= 2).
 * @param k Number of subspace dimensions per projection (must be >= 1).
 * @return 0 on success, or -1 on invalid dimensions or allocation failure.
 */
int pring_init(ProjRing* r, int cap, int k);
/**
 * @brief Releases all heap buffers associated with a ProjRing instance.
 *
 * Frees the projection matrix buffer and running moment arrays, zeroing the
 * struct.
 *
 * @param r Pointer to the ProjRing structure to deallocate.
 * @return void
 */
void pring_free(ProjRing* r);
/**
 * @brief Inserts a new k-dimensional projection into the ring and updates
 * running moments in O(k).
 *
 * If the ring is at capacity, subtracts the outgoing oldest projection and its
 * square from `s1` and `s2` before overwriting the slot and accumulating the
 * incoming vector.
 *
 * @param r Pointer to the active ProjRing instance.
 * @param y Pointer to the incoming k-dimensional projection vector.
 * @return void
 */
void pring_push(ProjRing* r, const double* y);
/**
 * @brief Computes normalized variance shares across the k directions from
 * running moments in O(k).
 *
 * Evaluates per-coordinate population variance `s2/n - (s1/n)^2`, clamps
 * negative floating-point drift to zero, and normalizes the vector to sum to 1
 * (falling back to uniform `1/k` shares if total variance is below
 * `SPARTAN_EPS`).
 *
 * @param r Pointer to the active ProjRing instance.
 * @param ev Output array of length k to receive normalized importance shares.
 * @return void
 */
void pring_importances(const ProjRing* r, double* ev);
/**
 * @brief Recomputes running moment sums `s1` and `s2` from scratch over the
 * active ring buffer.
 *
 * Called periodically (every `ring_w` windows) to flush accumulated
 * floating-point cancellation drift caused by incremental additions and
 * subtractions in `pring_push`.
 *
 * @param r Pointer to the active ProjRing instance.
 * @return void
 */
void pring_refresh(ProjRing* r);
/**
 * @brief Clears all stored projections and resets ring cursors and moments to
 * zero without reallocating.
 *
 * Used when transitioning to a new epoch basis or starting candidate staging so
 * stale coordinates from a previous subspace do not pollute the new
 * distribution.
 *
 * @param r Pointer to the active ProjRing instance.
 * @return void
 */
void pring_reset(ProjRing* r);

/* ---------------- the PCA directions ---------------- */

typedef struct {
  double* v; /* k rows of m numbers, flat; row j starts at v[j*m] */
  double* u; /* scratch room, m numbers */
  int k, m;
} OjaPCA;
/**
 * @brief Allocates an Oja streaming PCA basis and initializes directions to the
 * standard canonical axes.
 *
 * Allocates a flat `k * m` row-major basis matrix `v` and an `m`-element
 * deflation scratch vector `u`, setting initial basis rows to the first `k`
 * rows of the identity matrix.
 *
 * @param p Pointer to the OjaPCA structure to initialize.
 * @param k Number of principal directions to track (must satisfy 1 <= k <= m).
 * @param m Ambient window dimensionality.
 * @return 0 on success, or -1 on invalid dimensions or memory allocation
 * failure.
 */
int oja_init(OjaPCA* p, int k, int m);
/**
 * @brief Releases the basis matrix and deflation scratch buffer owned by an
 * OjaPCA instance.
 *
 * Frees heap memory and zeroes dimensions to prevent use-after-free bugs.
 *
 * @param p Pointer to the OjaPCA structure to deallocate.
 * @return void
 */
void oja_free(OjaPCA* p);
/**
 * @brief Performs a sequential Sanger/Oja stochastic gradient update with
 * post-update residual deflation.
 *
 * Copies the centered window `xc` into scratch buffer `u`, updates each basis
 * row `v_j` along the residual projection, and immediately re-projects `u` onto
 * the updated `v_j` to deflate the residual before advancing to direction `j +
 * 1`.
 *
 * @param p Pointer to the active OjaPCA instance.
 * @param xc Pointer to the mean-centered input window of length m.
 * @param lr Effective learning rate (`step / m`).
 * @return void
 */
void oja_update(OjaPCA* p, const double* xc, double lr);
/**
 * @brief Applies exponential moving average (Polyak-Ruppert) smoothing from a
 * live basis into a target basis.
 *
 * Updates `dst->v` in-place via `dst = (1 - beta) * dst + beta * src` to damp
 * out stochastic gradient oscillation around the true principal subspace.
 *
 * @param dst Pointer to the destination (smoothed) OjaPCA basis.
 * @param src Pointer to the source (live) OjaPCA basis.
 * @param beta Blending weight in [0.0, 1.0].
 * @return 0 on success, or -1 on NULL pointers, mismatched dimensions, or
 * out-of-range beta.
 */
int oja_blend(OjaPCA* dst, const OjaPCA* src, double beta);
/**
 * @brief Re-orthonormalizes the rows of an OjaPCA basis in-place using Modified
 * Gram-Schmidt.
 *
 * Sequentially projects out components along preceding rows `0 .. j-1` and
 * normalizes each row to unit length, skipping degenerate rows whose norm falls
 * below `SPARTAN_EPS`.
 *
 * @param p Pointer to the OjaPCA basis to orthonormalize.
 * @return void
 */
void gram_schmidt(OjaPCA* p);

/**
 * @brief Copies the k-by-m basis matrix from a source OjaPCA instance into a
 * pre-allocated destination.
 *
 * Used to snapshot the smoothed basis into a candidate or frozen epoch basis
 * without reallocating heap memory during stream execution.
 *
 * @param dst Pointer to the pre-allocated destination OjaPCA instance.
 * @param src Pointer to the source OjaPCA instance to copy from.
 * @return 0 on success, -1 if either pointer or basis buffer is NULL, or -2 if
 * k or m dimensions differ.
 */
int oja_copy_from(OjaPCA* dst, const OjaPCA* src);
/**
 * @brief Reorders the rows of a snapshot basis in-place according to a gather
 * permutation vector.
 *
 * Places old row `perm[i]` into new row `i` so the frozen basis rows align with
 * the descending eigenvalue order required by the Dynamic Alphabet Allocation
 * allocator. Must only be called on frozen/candidate snapshots, never on the
 * live Oja basis.
 *
 * @param p Pointer to the OjaPCA basis snapshot to reorder in-place.
 * @param perm Array of length k representing a valid permutation of [0, k-1].
 * @return 0 on success, or -1 on NULL arguments, out-of-bounds indices, or
 * scratch allocation failure.
 */
int oja_permute_rows(OjaPCA* p, const int* perm);

/**
 * @brief Measures the worst-case pairwise inner product and unit-norm deviation
 * across basis rows.
 *
 * Quantifies numerical drift away from the Stiefel manifold prior to
 * Gram-Schmidt re-orthonormalization to monitor step-size stability and MINDIST
 * bound slack.
 *
 * @param p Pointer to the OjaPCA basis to inspect.
 * @param max_dot Output pointer for `max_{i != j} |<v_i, v_j>|` (may be NULL).
 * @param max_len Output pointer for `max_j | ||v_j||_2 - 1 |` (may be NULL).
 * @return void
 */
void oja_orthonormality_error(const OjaPCA* p, double* max_dot,
                              double* max_len);
/**
 * @brief Computes the singular values of a small n-by-n matrix in-place via
 * one-sided Jacobi SVD.
 *
 * Orthogonalizes the rows of `a` through cyclic Givens rotations until all
 * off-diagonal Gram inner products fall below `SVD_TOL`, then extracts the
 * Euclidean norms of the orthogonalized rows as singular values. Destroys the
 * input matrix `a`.
 *
 * @param a Pointer to a flat n-by-n row-major matrix (modified in-place).
 * @param n Matrix dimension (number of rows and columns).
 * @param sv Output array of length n to receive the computed singular values.
 * @return Number of sweeps taken (1 to SVD_MAX_SWEEPS) on convergence, or -1 on
 * bad inputs or non-convergence.
 */
int svd_values(double* a, int n, double* sv, double* jacc);
/**
 * @brief Computes the principal angles (in radians) between two k-dimensional
 * orthonormal subspaces.
 *
 * Forms the k-by-k cross-Gram matrix `M = V_frozen * V_live^T` in `scratch`,
 * extracts its singular values via one-sided Jacobi SVD, clamps them to [0, 1]
 * to guard against floating-point NaN in `acos`, and converts each singular
 * value `sigma_j` to `acos(sigma_j)`. Both input bases must be orthonormal
 * prior to invocation.
 *
 * @param frozen Pointer to the reference orthonormal OjaPCA basis (unmodified).
 * @param live Pointer to the comparison orthonormal OjaPCA basis (unmodified).
 * @param scratch Caller-allocated workspace of `k * (k + 1)` doubles
 * (overwritten).
 * @param angles Optional output array of length k to receive all principal
 * angles in radians (may be NULL).
 * @param max_angle Optional output pointer to receive the largest principal
 * angle in radians (may be NULL).
 * @return 0 on success, -1 on NULL pointers or mismatched dimensions, or -2 if
 * Jacobi SVD fails to converge.
 */
int subspace_angles(const OjaPCA* frozen, const OjaPCA* live, double* scratch,
                    double* angles, double* max_angle);
/**
 * @brief Returns the sign of the determinant of the map from one basis to
 * another.
 *
 * Builds the k-by-k overlap table between the two bases and runs Gaussian
 * elimination with partial pivoting, tracking only the sign: each row swap
 * flips it and each pivot contributes its own. No SVD is needed, because
 * the nearest rotation differs from the overlap table only by positive
 * stretching factors, which cannot change a determinant's sign.
 *
 * @param to Pointer to the destination orthonormal basis (unmodified).
 * @param from Pointer to the source orthonormal basis (unmodified).
 * @param scratch Caller-allocated workspace of `k * k` doubles
 * (overwritten).
 * @return +1 if the two frames carry the same orientation, -1 if they do
 * not, 0 if the overlap is singular or an argument is bad.
 */
int basis_det_sign(const OjaPCA* to, const OjaPCA* from, double* scratch);
/**
 * @brief Constructs the nearest orthogonal transition matrix between two epoch
 * bases and computes the worst-case subspace leakage factor `sin(theta_max)`.
 *
 * Forms the k-by-k overlap matrix `M = V_to * V_from^T` and decomposes it via
 * one-sided Jacobi SVD (`J * M = Q * diag(sv)`) to recover the Procrustes
 * orthogonal map `rot = J^T * Q` that carries coordinates from `from` into
 * `to` without contraction. Simultaneously extracts the smallest singular value
 * `s_min = cos(theta_max)` and evaluates `sin_theta = sqrt(1 - s_min^2)` to
 * bound orthogonal residual leakage across epochs.
 *
 * @param to Pointer to the target orthonormal OjaPCA basis (unmodified).
 * @param from Pointer to the source orthonormal OjaPCA basis (unmodified).
 * @param scratch Caller-allocated workspace of `2 * k * k + k` doubles
 * (overwritten).
 * @param rot Output k-by-k row-major orthogonal transition matrix.
 * @param sin_theta Output pointer to receive `sin(theta_max)` in [0.0, 1.0].
 * @return 0 on success, -1 on NULL pointers or mismatched dimensions, -2 if
 * Jacobi SVD fails to converge, or -3 if a singular value collapses below
 * `SPARTAN_EPS`.
 */
int epoch_map(const OjaPCA* to, const OjaPCA* from, double* scratch,
              double* map, double* rot, double* sin_theta);
void word_box(const int* word, const double* bkpt, const int* alphabet, int k,
              double x_max, double* lo, double* hi);

void box_through(const double* map, const double* lo, const double* hi, int k,
                 double* out_lo, double* out_hi);

double residual_len(const double* y, int k, double x_norm);

double cross_epoch_mindist(const double* q_lo, const double* q_hi,
                           const double* c_lo, const double* c_hi, int k,
                           double sin_theta, double r);
/**
 * @brief Moves a subspace along the motion that carried one basis onto
 * another, scaled by a factor.
 *
 * The motion between two k-dimensional subspaces is k simultaneous plane
 * rotations whose angles are exactly the principal angles. The planes are
 * mutually orthogonal, so scaling every angle by the same factor and
 * rebuilding is exact: no series, no iteration, no re-orthonormalisation.
 *
 * @param from Starting orthonormal basis (unmodified).
 * @param to Orthonormal basis the motion has reached (unmodified).
 * @param f Scale; 0 reproduces `from`, 1 the span of `to`, more extrapolates.
 * @param out Pre-allocated result with the same k and m; must not alias.
 * @param scratch Caller-allocated workspace of `2*k*k + k + 2*m` doubles.
 * @return 0 on success, -1 on a bad argument, -2 if the Jacobi sweep failed,
 * -3 if a principal pair is degenerate and the motion is undefined.
 */
int subspace_advance(const OjaPCA* from, const OjaPCA* to, double f,
                     OjaPCA* out, double* scratch);
/* ---------------- the epoch-0 basis ---------------- */
/**
 * @brief The k leading directions of a set of windows (PCA, centred by their
 * own mean), by orthogonal iteration on the windows themselves (no m-by-m
 * covariance is formed, no centred copy is written).
 *
 * @param X n rows of m, the windows as the words see them (z-normed, nothing
 * subtracted); read only.
 * @param n Number of rows, at least k.
 * @param m Row length, must equal p->m.
 * @param p In: a starting orthonormal basis. Out: the k leading directions,
 * orthonormal, in no particular order.
 * @return Passes taken; -1 on a bad argument or failed allocation; -2 if the
 * span had not settled after WARM_PCA_MAX_PASSES (result still usable).
 */
int warm_pca(const double* X, int n, int m, OjaPCA* p);

/* ---------------- from numbers to letters ---------------- */
/**
 * @brief Projects a window onto the k rows of an OjaPCA basis.
 *
 * Computes `y[j] = <x, v_j>` for each basis direction `j in [0, k-1]`.
 * Every projection that becomes a letter, or a cut point, is of the z-normed
 * window with NOTHING subtracted: then two words of one epoch differ by
 * V (z_a - z_b) exactly, which is what makes MINDIST a lower bound.
 *
 * @param p Pointer to the OjaPCA basis used for projection.
 * @param x Pointer to the input window of length m.
 * @param y Output array of length k to receive the subspace coordinates.
 * @return void
 */
void project(const OjaPCA* p, const double* x, double* y);
/**
 * @brief Estimates normalized variance shares across k subspace coordinates
 * from a batch of projections.
 *
 * Computes the population variance of each column in the `n`-by-`k` projection
 * buffer and normalizes the vector to sum to 1 (falling back to uniform `1/k`
 * shares if total variance is below `SPARTAN_EPS`).
 *
 * @param proj Flat row-major matrix of `n` rows and `k` columns containing
 * subspace projections.
 * @param n Number of projection rows (windows) in the buffer.
 * @param k Number of subspace dimensions (columns).
 * @param ev Output array of length k to receive the normalized importance
 * shares.
 * @return void
 */
void estimate_eigenvalues(const double* proj, int n, int k, double* ev);
/**
 * @brief Computes empirical equiprobable quantile breakpoints for each subspace
 * coordinate.
 *
 * Sorts each column of the `n`-by-`k` projection buffer independently and
 * extracts `alphabet[j] - 1` empirical quantiles, placing a sentinel wall of
 * `DBL_MAX` at slot `alphabet[j] - 1` within each `MAX_ALPHABET`-strided row.
 *
 * @param proj Flat row-major matrix of `n` rows and `k` columns containing
 * subspace projections.
 * @param n Number of projection rows (windows).
 * @param k Number of subspace dimensions (columns).
 * @param alphabet Array of length k specifying the alphabet size `2^{bits[j]}`
 * for each position.
 * @param bkpt Output table of size `k * MAX_ALPHABET` to receive the
 * per-coordinate cut points.
 * @return void
 */
void compute_breakpoints(const double* proj, int n, int k, const int* alphabet,
                         double* bkpt);
/**
 * @brief Quantizes a k-dimensional projection vector into a discrete symbolic
 * word.
 *
 * Scans each coordinate's row in the breakpoint table `bkpt` to locate the
 * quantile bin containing `y[j]`, writing integer symbol indices in `[0,
 * alphabet[j] - 1]`.
 *
 * @param y Pointer to the k-dimensional subspace projection vector.
 * @param bkpt Flat breakpoint table of size `k * MAX_ALPHABET`.
 * @param k Number of symbols per word.
 * @param alphabet Array of length k containing the alphabet size for each
 * position.
 * @param word Output integer array of length k to receive the quantized symbol
 * indices.
 * @return void
 */
void digitize(const double* y, const double* bkpt, int k, const int* alphabet,
              int* word);

/* ---------------- splitting the bit budget ---------------- */

/**
 * @brief Whether the greedy allocation is the proven optimum: every bit of
 * the budget can be bought at a non-negative marginal gain. Always true for
 * lamda <= 1; for lamda > 1 true when k divides total_bits.
 *
 * @param k Number of word positions.
 * @param total_bits Total bit budget.
 * @param lamda Regularization weight.
 * @param ref_bits Reference bits per position, or NULL for total_bits / k.
 * @return 1 if the condition holds, 0 if not.
 */
int daa_exact(int k, int total_bits, double lamda, const int* ref_bits);

/**
 * @brief The regularized Dynamic Alphabet Allocation (DAA) bit split, by
 * greedy auction: each bit goes to the position bidding most for it.
 *
 * Maximizes sum_j ev_j * (b_j - lamda * (b_j - ref_j)^2) subject to
 * sum_j b_j = total_bits and 0 <= b_j <= MAX_BITS. Exact (the objective is
 * separable and concave) and non-increasing across descending `ev` under
 * daa_exact, which it enforces. O(k * total_bits), no allocation.
 *
 * @param ev Array of length k containing normalized importance shares in
 * descending order.
 * @param k Number of word positions (components).
 * @param total_bits Total bit budget to distribute across all k positions.
 * @param lamda Regularization penalty weight against deviating from the
 * reference allocation.
 * @param ref_bits Optional reference bit allocation of length k (NULL defaults
 * to uniform `total_bits / k`).
 * @param bits Output array of length k to receive the optimal bit counts.
 * @param score Optional output pointer to receive the objective of the
 * result (may be NULL).
 * @return 0 on success, or -1 on invalid inputs, a budget above
 * k * MAX_BITS, settings outside daa_exact, or `ev` not descending.
 */
int daa_allocate(const double* ev, int k, int total_bits, double lamda,
                 const int* ref_bits, int* bits, double* score);

/**
 * @brief Evaluates the regularized DAA objective value for a given bit
 * allocation vector.
 *
 * Computes the sum of linear variance gains `bits[i] * ev[i]` minus the
 * quadratic deviation penalty from `ref_bits` (or `total_bits / k` if NULL).
 *
 * @param ev Array of length k containing normalized importance shares.
 * @param k Number of word positions.
 * @param total_bits Total bit budget across positions.
 * @param lamda Regularization weight.
 * @param ref_bits Optional reference bit allocation of length k (may be NULL).
 * @param bits Bit allocation vector of length k to evaluate.
 * @return The scalar DAA objective score.
 */
double daa_score(const double* ev, int k, int total_bits, double lamda,
                 const int* ref_bits, const int* bits);
/**
 * @brief Computes the marginal 1-bit swap optimality gap of the active bit
 * allocation in O(k) time.
 *
 * Exploits concavity of the per-position objective to evaluate the highest
 * marginal gain from adding one bit (`best_bid`) minus the lowest marginal loss
 * from removing one bit
 * (`best_ask`) without re-solving the allocation. A positive gap means a
 * one-bit transfer improves the split under the current live shares `ev`;
 * under daa_exact a gap of 0 means no transfer of any size can, so the
 * split is optimal. Gaps at or below SPARTAN_EPS count as 0 (rounding).
 *
 * @param ev Array of length k containing live normalized importance shares.
 * @param k Number of word positions.
 * @param total_bits Total bit budget across positions.
 * @param lamda Regularization weight.
 * @param ref_bits Reference bit allocation used when the active dictionary was
 * built (may be NULL).
 * @param bits Active bit allocation vector of length k.
 * @return Positive marginal gain of the best 1-bit transfer, or 0.0 if the
 * current split is optimal.
 */
double daa_gap(const double* ev, int k, int total_bits, double lamda,
               const int* ref_bits, const int* bits);
#endif /* SPARTAN_H */
