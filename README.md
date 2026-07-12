# spartanon — Online SPARTAN prototype (C, from scratch)

Educational step-by-step C implementation of a streaming SPARTAN
(symbolic time-series representation) core: per-window z-norm,
running mean, Sanger/Oja online PCA. No external dependencies.

## Build

Release:

    gcc -O2 -Wall -Wextra -Wshadow -std=c11 fake_ts.c -o spartan_toy -lm

Debug (ASan/UBSan, slower):

    gcc -g -O0 -Wall -Wextra -Wshadow -fsanitize=address,undefined \
        -std=c11 fake_ts.c -o spartan_toy_dbg -lm

## Run

    ./spartan_toy

Expected output: v0 ~ monotone "slope" pattern, v1 ~ V-shape
(curvature), norms ~ 1.0. v1.v2 orthogonality pending the
Gram-Schmidt step (work in progress).

## Status

- [x] per-window z-normalization
- [x] running mean (incremental)
- [x] Oja + Sanger deflation (K=3 components)
- [ ] periodic Gram-Schmidt re-orthogonalization
- [ ] projections + quantile breakpoints + digitize (symbols)
- [ ] chi-squared drift detector
- [ ] epoch rebuild + Procrustes/SO(3) symbol translation
