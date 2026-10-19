<h1 align="center">Online SPARTAN</h1>

<p align="center">
  <em>A symbolic representation for unbounded time-series streams, whose distance<br>
  stays a valid lower bound while the subspace underneath it moves.</em>
</p>

<p align="center">
  <img alt="language"   src="https://img.shields.io/badge/language-C11-39FF14?style=flat-square&labelColor=0D1117">
  <img alt="compiler"   src="https://img.shields.io/badge/gcc-13.4-39FF14?style=flat-square&labelColor=0D1117&logo=gnu&logoColor=39FF14">
  <img alt="build"      src="https://img.shields.io/badge/build-make-39FF14?style=flat-square&labelColor=0D1117">
  <img alt="tests"      src="https://img.shields.io/badge/tests-unit%20%7C%20golden%20%7C%20replay-39FF14?style=flat-square&labelColor=0D1117">
  <img alt="sanitizers" src="https://img.shields.io/badge/asan%20%7C%20ubsan-clean-39FF14?style=flat-square&labelColor=0D1117">
  <img alt="status"     src="https://img.shields.io/badge/status-research%20prototype-39FF14?style=flat-square&labelColor=0D1117">
</p>

SPARTAN turns a time series into a short word by projecting it onto its principal components

**A symbolic representation for unbounded time-series streams, whose distance stays a
valid lower bound while the subspace underneath it moves.**

SPARTAN turns a time series into a short word by projecting it onto its principal
components and quantising each coordinate, spending more alphabet on the directions
that carry more variance. It assumes a corpus: one pass over everything, one PCA, one
dictionary.

A stream has no corpus. It arrives one window at a time, it never ends, and the
directions that mattered an hour ago may not matter now. Online SPARTAN keeps the
representation current without revisiting a single window, and keeps the lower bound
intact across every change it makes.

---

## Table of contents

- [Background](#background)
- [The pipeline](#the-pipeline)
- [Design principles](#design-principles)
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
because the estimator itself never stops moving. Freeze the basis and the representation
goes stale. Follow it and words written minutes apart live in different coordinate
systems and cannot be compared.

Online SPARTAN resolves this with **epochs**. Within an epoch one frozen basis writes
every word, so words are mutually comparable by construction. Between epochs the
relation between the two bases is measured rather than assumed, and the bound carries a
computable slack that grows with the angle. The angle never makes an answer wrong. It
makes the search slower, and the user decides how much slowness to accept.

---

## The pipeline

Each window passes through the same seven stages. Stages 1 to 4 run every window;
stage 5 runs periodically; stages 6 and 7 run once per epoch.

**1. Normalise and centre.** Each channel of the window is z-normalised on its own, then
the running mean of the stream is subtracted. The mean is maintained incrementally, so
no window is ever stored for this purpose.

**2. Track the subspace.** Oja's rule moves a basis one step toward the current window,
with deflation so later directions only see what earlier ones left behind. The step is
divided by the window length, so one setting means the same thing at every window size.

**3. Smooth.** Oja with a constant step does not converge to a point instead it settles into a
cloud. A Polyak–Ruppert average of the raw iterate cancels what the wandering adds, and
it is the averaged basis, not the raw one, that the rest of the system uses.

**4. Straighten.** Averaging unit vectors does not give a unit vector, and averaging
perpendicular ones does not give perpendicular ones. Gram–Schmidt restores
orthonormality periodically. This is also the only moment at which an angle between two
bases means anything, which is why every measurement is scheduled against it.

**5. Measure.** Principal angles between the frozen basis and the working one give the
displacement and principal angles between the averaged and raw bases give the estimator's
own tremble, with the signal removed. A smoothed derivative gives the rate.

**6. Allocate and quantise.** Variance per projected coordinate gives the importance of
each position. A dynamic program splits the bit budget across positions. Breakpoints are
empirical quantiles of the collected projections, and each coordinate becomes a letter.

**7. The epoch gate.** When the projected angle plus a margin would reach the user's
policy before a replacement could be prepared, a candidate basis is frozen and a second
ring of projections begins filling behind the one in use — one stream, two projections.
When the policy is crossed and the candidate is ready, it is promoted: a new dictionary
is built from its ring, the candidate becomes the writing basis, and a new epoch opens.
If the policy is crossed before the candidate is ready, the old basis keeps writing and
the wait is measured rather than hidden.

---

## Design principles

### The bit split is an auction over a concave objective

Each position's reward is its importance multiplied by a function of the bits it
receives, and that function is the same concave shape for every position: linear gain
from each bit, minus a quadratic penalty for straying from the even split.

Concavity is what makes the problem an auction. The marginal value of a position's next
bit is its importance times the increment of that concave function, so the more bits a
position already holds, the less it is willing to pay for another. Positions bid, the
highest bid takes the next bit, and every bid falls as its owner accumulates.

This framing is not decoration. It gives a cheap online test for whether the current
split has gone stale: compute the best bid still outstanding — the largest gain from
granting one more bit — and the best ask — the smallest loss from taking one away. A
positive spread means a profitable trade exists and the allocation no longer fits the
data. The spread is reported every window and costs a handful of operations.

The split itself is solved exactly by a dynamic program rather than greedily, because
the budget is finite and a constraint couples the positions: no position may receive
more bits than the one before it. That constraint is a theorem only when importances
arrive in descending order, which a batch PCA guarantees and a streaming estimator does
not — so the order is imposed before the split, not assumed.

### A turning basis costs speed, never correctness

Projection onto any orthonormal basis is a contraction: it cannot increase a distance.
The lower bound therefore holds for a word written by any such basis, however poorly
that basis fits the data. A basis that has turned away from the data keeps less variance,
so the bound is looser, so fewer candidates are discarded. The search gets slower and
the answer stays exact.

This is why the angle threshold is a **policy** and not a safety limit. It is the user
stating how much pruning power they are willing to lose before paying for a new epoch.

### Measured, not tuned

Every number in the system is one of two things: a setting the user states, or a
quantity measured at runtime and reported in the summary. There are no constants chosen
because they looked reasonable on one dataset. Where a constant was unavoidable it was
selected by measurement across a sweep, and the measurement is recorded in a comment
next to it.

The run reports what it cannot promise: the worst overshoot past the policy, how long a
needed re-basing had to wait, how stale an incoming basis already was when it took over,
and how much of the observed displacement was the estimator rather than the data.

### Three bases, one job each

- **live** — the raw Oja iterate. Trembles. Watched as an early warning that the step is
  too large.
- **averaged** — the smoothed copy. Where the data is *now*. Everything measures against
  it.
- **frozen** — writes the letters. Never moves within an epoch, which is what makes the
  words of an epoch mutually comparable.

A fourth, the **candidate**, exists only while a replacement is being prepared. It is a
snapshot, so every projection collected behind it sits in one fixed frame.

### Everything is reproducible from what is written

A run can dump, per epoch, the basis that wrote the words and the dictionary that
quantised them, plus an index of which window each epoch starts at. These are sufficient
to reconstruct every letter from the raw stream without re-running the C. A Python
verifier does exactly that and compares letter by letter.

---

## Requirements

| | |
| --- | --- |
| Compiler |  C11 compiler or clang, tested on gcc 13.4 |
| Platform | POSIX. `clock_gettime` needs `_POSIX_C_SOURCE 200809L`, already set |
| Build | GNU make |
| Debug build | AddressSanitizer and UndefinedBehaviorSanitizer |
| Tools | Python 3.8+, standard library only |

No third-party C libraries. The program links only `libm`.

One C11 feature is used that the standard marks optional: variable length
arrays in the dictionary builder, and nothing else. A compiler that defines
`__STDC_NO_VLA__` will not build it as written.

## Build and run

```
make            release build, -O2
make debug      -O0 with AddressSanitizer and UndefinedBehaviorSanitizer
make check      run the built-in demo and diff against the golden files
make golden     regenerate the golden files, after reading the diff
make unit       build and run the C unit tests
make clean      remove build/ and every tests/*/out/
```

Running:

```
./build/spartan_toy                 built-in demo stream, defaults
./build/spartan_toy run.cfg         everything from the settings file
```

Words go to stdout, diagnostics to stderr. Every run begins by echoing the settings
actually in force, in a shape that can be fed straight back in as a settings file.

The regression target is parameterised by the test folder:

```
make check
make check TEST=synthetic CFG=tests/synthetic/cfg/stationary.cfg
```

---

## Settings

One `key = value` per line, `#` begins a comment, an unknown key is an error rather than
a shrug. The ones that change behaviour most:

| key | meaning |
| --- | --- |
| `components` | how many principal directions, and therefore letters per word |
| `window` | frames per window |
| `channels` | channels per frame so a window is `channels * window` numbers |
| `stride` | how far the window advances |
| `total_bits` | the budget split across positions |
| `lambda` | how strongly the split is pulled toward the even allocation |
| `step` | Oja step, divided by the window length internally |
| `basis_average` | how fast the smoothed basis follows the raw one |
| `gs_period` | windows between orthonormalisations |
| `angle_period` | windows between angle measurements must be a multiple of `gs_period` |
| `theta_threshold_deg` | the policy: how far the basis may turn before the epoch closes |
| `ring_window` | projections collected before a replacement dictionary is built |
| `warmup` | windows collected before the first dictionary |
| `out_dir` | The folder where files for each epoch are saved, and if left empty, no files will be written|

---

## What a run writes

With `out_dir` set, each epoch leaves three artefacts behind:

| file | contents |
| --- | --- |
| `dict_<n>.csv` | importances, bits, alphabet sizes and breakpoints, one row per position |
| `basis_<n>.csv` | the directions that wrote this epoch's words, one row each |
| `epochs.csv` | one line per epoch: first window, the angle it was cut at, the staleness of the incoming basis, the objective its split reached |

`epochs.csv` has no "last window" column by design. An epoch ends where the next one
begins, and the last one ends with the stream.

---

## Verification

**Unit tests** (`make unit`) pin the dynamic program and the projection ring against computed expectations, including the cases where this implementation deliberately
departs from the batch reference SPARTAN.

**Golden files** (`make check`) prove that a change did not alter the output. They say
nothing about whether the output is right. Mainly for debugging purposes and moving step by step without breaking between commits

**The verifier** (`tools/verify_words.py`). It reads only the
per-epoch files and the raw stream, recomputes every projection and every letter from
scratch, and compares. It does not read the C source. Agreement means the freeze, the
promotion, the row ordering and the breakpoints are consistent with each other and with
what was written to disk.

```
python3 tools/verify_words.py tests/demo/out/epochs tests/demo/out/words.csv
```

---

## Repository layout

```
src/            the program: maths, stream, settings, driver
tests/unit/     C unit tests, each with its own main
tests/demo/     the built-in stream: cfg, expected, out
tests/synthetic/  generated data with known properties
tests/real/     archive datasets
tools/          data generation, verification and comparison, in Python
build/          every binary, not tracked
```

