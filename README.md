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

## Table of contents

- [Background](#background)
- [The pipeline](#the-pipeline)
- [Design principles](#design-principles)
- [Requirements](#requirements)
- [Build and run](#build-and-run)
- [Settings](#settings)
- [What a run writes](#what-a-run-writes)
- [Verification](#verification)
- [Repository layout](#repository-layout)

---

## Background

Similarity search over a large corpus of time series is expensive if every candidate
must be compared exactly. Symbolic representations make it cheap: encode each series as
a word, define a distance on words that is provably no larger than the true distance,
and use it to discard candidates before touching the raw data. The guarantee is what
matters. A lower bound that is sometimes wrong is not a faster search, it is a broken
one.

Doing this on a stream raises a problem the batch setting never faces. The basis is
learned, and a learned basis drifts: partly because the data genuinely changes, partly
because the estimator itself never stops moving. Freeze the basis and the
representation goes stale. Follow it and words written minutes apart live in different
coordinate systems and cannot be compared.

Online SPARTAN resolves this with **epochs**. Within an epoch one frozen basis and one
dictionary write every word, so words are mutually comparable by construction. Between
epochs the relation between the two is handled explicitly, in one of two ways that are
both proven lower bounds (see [the cross-epoch bound](#the-cross-epoch-bound-has-two-forms)).
The angle never makes an answer wrong. It can make a search slower, and the user
decides how much of that to accept.

---

## The pipeline

**1. Normalise.** Each channel of the window is z-normalised on its own. Every letter,
every cut point and every PCA is computed on this z-normed window with **nothing
subtracted**: two words of one epoch then differ by exactly the projection of the
difference of their windows, which is what the lower bound needs. A running mean of
the stream is kept for one purpose only, to centre the input of Oja's rule.

**2. Warm-up.** The first windows are collected, and the epoch-0 basis is **exact PCA**
on them, by orthogonal iteration on the windows themselves (no m-by-m covariance is
formed). The cut points of epoch 0 come from the same windows. Oja is then seeded from
this basis and only has to track, not converge.

**3. Track.** Oja's rule moves a basis one step toward each window, with deflation so
later directions only see what earlier ones left behind. The step is divided by the
window length, so one setting means the same thing at every window size. A
Polyak-Ruppert average of the raw iterate cancels the estimator's own wandering, and
Gram-Schmidt restores orthonormality periodically. Only right after a straightening is
an angle between two bases an angle, which is why every measurement is scheduled
against it.

**4. Write.** The window is projected onto the frozen basis of the current epoch and
each coordinate becomes a letter through that epoch's cut points.

**5. Type 1 epoch: same directions, new split.** A ring of the latest projections gives
the current importance of each position. Once per ring the allocator's bid-ask spread is
tested (below). When it is positive the split no longer fits: bits and cut points are
rebuilt from the ring, the basis is only reordered, and a new epoch opens at zero
angle from the old one.

**6. Type 2 epoch: new directions.** The largest principal angle between the frozen
basis and the averaged one is measured. When it passes the user's threshold, the new
basis is exact PCA on a ring of the last windows, the same operation as the warm-up,
and the new dictionary is cut on their projections. Nothing is predicted, so nothing can
fire early, and the incoming basis is never older than the ring. Oja restarts from it.

The previous design for step 6, which predicted the crossing from the rate of the angle
and staged a candidate basis ahead of it, is kept behind a compile switch
(`make A`, `-DSTAGING_A`) to compare on real data.

---

## Design principles

### The bit split is an auction over a concave objective

Each position's reward is its importance multiplied by a function of the bits it
receives, and that function is the same concave shape for every position: linear gain
from each bit, minus a quadratic penalty for straying from the even split.

Concavity is what makes the problem an auction. The marginal value of a position's next
bit is its importance times the increment of that concave function, so the more bits a
position already holds, the less it is willing to pay for another.

This gives a cheap online test for whether the current split has gone stale: compute
the best bid still outstanding, the largest gain from granting one more bit, and the
best ask, the smallest loss from taking one away. A positive spread means a profitable
trade exists and the allocation no longer fits the data. It costs two passes over k
numbers and decides when a type 1 epoch opens.

The split itself is that auction run to the end: each bit goes to the highest bid.
For a separable concave objective under one budget this greedy allocation is the exact
optimum, not an approximation (Edmonds 1970, Federgruen and Groenevelt 1986), at
O(k·N) and with no memory. The published allocator is a dynamic program; the two agree.

One condition makes that hold. No position may receive more bits than the one before
it, which is not part of the classical structure; greedy keeps it by itself, once the
importances are sorted, as long as every bit is bought at a non-negative marginal gain.
That is always true for λ ≤ 1, and for λ > 1 when the budget is a multiple of the
number of positions. Outside it the leftover bits would go to the least important
positions, so the configuration refuses those settings. Under the same condition a
non-positive spread proves the split optimal. Importances arrive in descending order
from a batch PCA but not from a streaming estimator, so the order is imposed before the
split rather than assumed. Bids are computed from the closed form of the marginal gain,
so exact ties stay exact and go to the position holding fewer bits.

Unlike the published allocator, a position may receive **zero** bits: a direction
carrying nothing gets no letters but keeps its row in the basis, so the number of
directions never changes.

### A turning basis costs speed, never correctness

Projection onto any orthonormal basis is a contraction: it cannot increase a distance.
The lower bound therefore holds for a word written by any such basis, however poorly
that basis fits the data. A basis that has turned away from the data keeps less
variance, so the bound is looser, so fewer candidates are discarded. The search gets
slower and the answer stays exact.

This is why the angle threshold is a **policy** and not a safety limit. It is the user
stating how much pruning power they are willing to lose before paying for a new epoch.

### The cross-epoch bound has two forms

Every closed epoch keeps its basis and its dictionary, so a word of an older epoch can
always be reached. Two ways, chosen by `cross_epoch`:

- **search** (default). The query window is in hand, as in a k-NN search. It is
  projected again onto the older epoch's basis and cut with that epoch's dictionary;
  the two words then share an epoch and the ordinary MINDIST applies, with no slack.
  Cost: k·m multiply-adds per query per older epoch, not per candidate.
- **box**. The query's cell box is carried into the older epoch by the epoch map, and
  the slack sin(θ_max)·r is taken off, where r is the part of the query its own basis
  did not capture. Cost k·k per query per older epoch, loose at large angles.

A join over stored words has no window to re-project, so it is always box, with r
replaced by the largest value it could have.

### Measured, not tuned

Every number in the system is one of two things: a setting the user states, or a
quantity measured at runtime and reported in the summary. Every default lives in one
place, `config_default_mv` in `src/spartan.c`, with the measurement that chose it in a
comment beside it. Each run writes the settings it actually used, every default
resolved, to `settings.cfg`, and every tool reads them from there: no Python file
carries a setting or a copy of a C constant.

The run reports what it cannot promise: the worst overshoot past the policy, how stale
an incoming basis already was when it took over, and how much of the observed
displacement was the estimator rather than the data.

### Four bases and a ring, one job each

- **live**: the raw Oja iterate. Trembles. Watched as an early warning that the step is
  too large.
- **averaged**: the smoothed copy. Where the data is *now*. The angle is measured
  against it.
- **frozen**: writes the letters. Never moves within an epoch, which is what makes the
  words of an epoch mutually comparable.
- **candidate**: the incoming basis while a type 2 epoch is built.
- **the window ring**: the last `ring_window` z-normed windows, which a type 2 epoch is
  computed from. This is the largest piece of state, `ring_window · m` doubles, and it
  does not grow with the stream.

### Optional level letters

z-norm erases each window's mean and spread, so a level shift or an amplitude jump is
invisible to the words. For anomaly detection, `level_letters = true` appends, per
channel, the mean and the spread taken **before** z-norm as two extra letters, cut once
from the warm-up. They are outside the DAA budget and do not touch the PCA letters. Off
by default: for similarity search the z-normed distance is the target.

---

## Requirements

| | |
| --- | --- |
| Compiler | gcc or clang, tested on gcc 13.4 |
| Platform | POSIX. `clock_gettime` needs `_POSIX_C_SOURCE 200809L`, already set |
| Build | GNU make |
| Debug build | AddressSanitizer and UndefinedBehaviorSanitizer |
| Tools | Python 3.8+ with NumPy. The comparison tools also need the authors' virtualenv (NumPy, SciPy, scikit-learn, numba) and their checkout, `SPARTAN_REPO` |

No third-party C libraries. The program links only `libm`.

Two things are used that plain C11 does not guarantee, which is why the compiler is
named rather than the standard. Variable length arrays, in the dictionary builder: C11
marks them optional, and a compiler defining `__STDC_NO_VLA__` will not build it as
written. And `__builtin_ctz`, in the alphabet limit: a GNU extension that gcc and clang
both provide.

---

## Build and run

```
make              release build, -O2
make A            the previous re-basing design, build/spartan_A
make debug        -O0 with AddressSanitizer and UndefinedBehaviorSanitizer
make unit         build and run the C unit tests
make data         generate the synthetic streams under tests/tsb
make run          one run of the C on CFG, summary to the terminal
make compare      run, both batch references, the full report; re-derives every word
make truth        batch PCA on the stream, and the angle to what was planted
make sweep        the sensitivity sweep (datasets and levels: top of tools/sweep.py)
make sweep-plan   what the sweep would run, running only the baselines
make check        sanitized run on CFG, diff against the golden files
make golden       regenerate the golden files for CFG, after reading the diff
make clean        remove build/ and every tests/*/out/ (keeps out/, the sweeps)
```

`CFG` defaults to `tests/tsb/cfg/tsb_drift.cfg`; any cfg works:

```
make compare CFG=tests/tsb/cfg/tsb_noisy.cfg
make compare SPARTAN_REPO=/path/to/SPARTAN-main
```

The program itself takes one argument, the settings file, which must name the data:

```
./build/spartan_toy run.cfg
```

Words go to the file named by `words` (`-` for stdout), diagnostics to stderr. Every
run begins by echoing the settings actually in force, in a shape that can be fed
straight back in as a settings file. The folder named by `out_dir` must already exist;
the make targets create it.

---

## Settings

One `key = value` per line, `#` begins a comment, and an unknown key or a malformed
value is an error rather than a shrug. A key left out takes the C's default; the value
used is in the run's `settings.cfg`.

| key | meaning |
| --- | --- |
| `input` | data file, one frame per line, `channels` numbers per frame |
| `channels` | channels per frame, so a window is `channels * window` numbers |
| `window` | frames per window |
| `stride` | how far the window advances, between 1 and `window`; default `window` |
| `components` | how many principal directions, and therefore letters per word |
| `total_bits` | the budget split across positions; default twice `components` |
| `lambda` | how strongly the split is pulled toward the even allocation; above 1 only with a `total_bits` that is a multiple of `components` |
| `step` | Oja step, divided by the window length internally |
| `basis_average` | how fast the smoothed basis follows the raw one, between 0 and 1 |
| `gs_period` | windows between orthonormalisations |
| `angle_period` | windows between angle measurements; a multiple of `gs_period`, default equal to it |
| `theta_threshold_deg` | the policy: how far the basis may turn before a type 2 epoch. Above 0 and below 90 |
| `ring_window` | windows in the ring a type 2 epoch is computed from, and projections in the type 1 ring |
| `warmup` | windows before the first word, the epoch-0 PCA and cut points; default `ring_window` |
| `cross_epoch` | `search` or `box`: how a live query is bounded against an older epoch |
| `level_letters` | `true` or `false`: append each channel's mean and spread as two extra letters |
| `freeze_ahead` | design A only: how far past the data a staged candidate is frozen |
| `words` | where the words go, `-` for stdout |
| `out_dir` | folder for the per-epoch files; empty writes none, and the folder must exist |

---

## What a run writes

With `out_dir` set:

| file | contents |
| --- | --- |
| `settings.cfg` | every setting the run used, defaults resolved, plus the compile-time constants the tools need. Valid as a settings file |
| `epochs.csv` | one line per epoch: first window it writes, the angle it was cut at, the staleness of the incoming basis, the objective its split reached, its type (0 warm-up, 1 new split, 2 new directions) |
| `dict_<n>.csv` | importances, bits, alphabet sizes and cut points, one row per position |
| `basis_<n>.csv` | the directions that wrote this epoch's words, one row each |
| `levels.csv` | the level-letter cut points, only with `level_letters = true` |

`epochs.csv` has no "last window" column by design. An epoch ends where the next one
begins, and the last one ends with the stream, so no field has to be revisited after it
is written.

---

## Verification

Four layers, each catching what the others cannot.

**Unit tests** (`make unit`) pin the allocator and the projection ring against
hand-computed expectations, including the cases where this implementation deliberately
departs from the batch reference.

**Golden files** (`make check`) prove that a change did not alter the output, words and
summary, under the sanitizers. They say nothing about whether the output is right. The
release and the sanitized `-O0` builds write identical words.

**Re-derivation** (`make compare`). `tools/compare_stages.py` reads only the per-epoch
files and the raw stream, recomputes every projection and every letter, and stops on a
single mismatch. It then measures, on the same sampled windows, the tightness (TLB) and
the pruning power of our words against two references from the authors' own code: batch
SPARTAN fitted on the warm-up windows, and fitted on every scored window, which has seen
the whole stream in advance and is the ceiling. It counts every pair where a bound came
out above the true distance; that count must be zero.

**The sweep** (`make sweep`) repeats the comparison over the sensitivity grid. On the
synthetic streams, 114 configurations, zero violations.

Anything that changes the summation order, such as `-ffast-math` or hand-written SIMD,
can move a projection that sits within a rounding error of a cut point, and with it a
letter. The C sums its dot product in four partial accumulators and NumPy in its own
order, so the re-derivation compares letters, which agree, and not projections bit for
bit.

---

## Repository layout

```
src/              the program: maths, stream, settings, driver
tests/unit/       C unit tests, each with its own main
tests/tsb/        synthetic streams: cfg, data, expected (golden), ref, truth, out
tests/real/       archive datasets
tools/            data generation and conversion, the comparison, the sweep
build/            every binary, not tracked
out/              sweep results, not tracked
```

| tool | does |
| --- | --- |
| `make_tsb.py` | the synthetic streams; data only, no algorithm setting |
| `convert.py` | TSB-AD, UCR anomaly and UCR classification files into the frame format |
| `read_settings.py` | reads a run's `settings.cfg`; every tool goes through it |
| `ref_spartan.py` | the two batch references, from the authors' own SPARTAN class |
| `compare_stages.py` | re-derivation, TLB, pruning, agreement, memory |
| `batch_truth.py` | batch PCA on a stream and the angle to what was planted |
| `sweep.py` | the sensitivity sweep |

Headers live beside their sources on purpose: a quoted include searches the including
file's own directory first, so nothing in `src/` needs an include path. The unit tests
sit outside that directory and are given `-Isrc`.
