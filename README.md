<h1 align="center">Online SPARTAN</h1>

<p align="center">
  <em>A symbolic representation for unbounded time-series streams, whose distance<br>
  stays a valid lower bound while the subspace underneath it moves.</em>
</p>

<p align="center">
  <img alt="language"   src="https://img.shields.io/badge/language-C11-39FF14?style=flat-square&labelColor=0D1117&logo=c&logoColor=39FF14">
  <img alt="compiler"   src="https://img.shields.io/badge/gcc-13.4-39FF14?style=flat-square&labelColor=0D1117&logo=gnu&logoColor=39FF14">
  <img alt="build"      src="https://img.shields.io/badge/build-make-39FF14?style=flat-square&labelColor=0D1117">
  <img alt="tests"      src="https://img.shields.io/badge/tests-unit%20%7C%20golden%20%7C%20re--derivation-39FF14?style=flat-square&labelColor=0D1117">
  <img alt="sanitizers" src="https://img.shields.io/badge/asan%20%7C%20ubsan-clean-39FF14?style=flat-square&labelColor=0D1117">
  <img alt="status"     src="https://img.shields.io/badge/status-research%20prototype-39FF14?style=flat-square&labelColor=0D1117">
</p>

SPARTAN turns a time series into a short word by projecting it onto its principal
components and quantising each coordinate, spending more alphabet on the directions
that carry more variance. It assumes a corpus: one pass over everything, one PCA, one
dictionary.

A stream has no corpus. It arrives one window at a time, it never ends, and the
directions that mattered an hour ago may not matter now. Online SPARTAN keeps the
representation current without revisiting the stream, and keeps the lower bound
intact across every change it makes.

---

## Requirements

- **C compiler:** `gcc` or `clang` (tested on `gcc 13.4`), plus GNU `make`. The C core has zero external dependencies and links only `libm`.
- **Platform:** Linux / POSIX.
- **Python environment (for evaluation scripts only):** Python 3.8+ with `numpy`, `scipy`, `scikit-learn`, `numba`, `pandas`, `matplotlib`, and `kmedoids`, alongside a local checkout of the reference SPARTAN repository.

---

## Build and run

Build the release binary and run the test suite:

```bash
make              # release build (./build/spartan_toy)
make A            # alternative staging build (./build/spartan_A)
make debug        # build with AddressSanitizer and UndefinedBehaviorSanitizer
make unit         # run C unit tests
make check        # run sanitized build against golden files
make affine       # verify invariance under x -> a*x + b
make clean        # remove build artifacts
```

Run the binary directly by passing a configuration file:

```bash
./build/spartan_toy tests/tsb/cfg/tsb_drift.cfg
```

Run the synthetic stream comparison against the batch reference:

```bash
make data
make compare CFG=tests/tsb/cfg/tsb_drift.cfg SPARTAN_REPO=/path/to/SPARTAN
```

Run the archive benchmark scripts (UCR classification, UCR clustering, and TSB-UAD anomaly detection):

```bash
python3 tools/eval_ucr_task1.py --blocks all
python3 tools/eval_ucr_task2_clustering.py --blocks all
python3 tools/eval_task3_anomaly.py --blocks all --max-per-folder 5 --workers 12
```

---

## Settings

Configuration files use one `key = value` pair per line (`#` starts a comment).

| Key | Description |
| --- | --- |
| `input` | Path to the input data file (one frame per line) |
| `channels` | Number of channels per frame |
| `window` | Frames per window |
| `stride` | Step size between consecutive windows (default: `window`) |
| `components` | Number of principal directions and letters per word |
| `total_bits` | Total bit budget across all positions (default: `2 * components`) |
| `lambda` | Regularisation weight toward an even bit split |
| `step` | Oja learning rate scale (divided by window length internally) |
| `basis_average` | Exponential smoothing weight for the tracked basis in `[0, 1]` |
| `gs_period` | Windows between Gram-Schmidt orthonormalisations |
| `angle_period` | Windows between subspace angle checks (multiple of `gs_period`) |
| `theta_threshold_deg` | Maximum drift angle in degrees before opening a new basis epoch |
| `ring_window` | Ring buffer size for rebuilding dictionaries and bases |
| `warmup` | Initial windows used for Epoch 0 PCA and cut points |
| `cross_epoch` | Cross-epoch bound mode: `search` (default) or `box` |
| `window_znorm` | `true` (default) to z-normalise each window, or `false` for raw windows |
| `words` | Output path for symbolic words (`-` for stdout) |
| `out_dir` | Existing directory where per-epoch files (`settings.cfg`, `epochs.csv`, `dict_<n>.csv`, `basis_<n>.csv`) are written |

---

## Repository layout

```text
src/              C implementation (maths, stream reader, config parser, driver)
tests/unit/       C unit tests
tests/tsb/        Synthetic streams, configs, and golden files
tools/            Python evaluation, conversion, and benchmark scripts
build/            Compiled binaries
out/              Benchmark outputs and figures
```
