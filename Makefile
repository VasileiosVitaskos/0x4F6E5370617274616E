CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -Wshadow -std=c11
DBGFLAGS = -g -O0 -Wall -Wextra -Wshadow -std=c11 -fsanitize=address,undefined
LDLIBS   = -lm
PYTHON   = python3

# The authors' checkout: ref_spartan.py loads their SPARTAN class from it and
# compare_stages.py their distance code. Override per call if it lives
# elsewhere:  make compare SPARTAN_REPO=/path/to/SPARTAN-main
SPARTAN_REPO = ~/dev/SPARTAN-main

# Everything except main.c, so a unit test can link the same code without
# dragging in a second main()
LIBSRC = src/spartan.c src/stream.c src/config.c
SRC    = $(LIBSRC) src/main.c
HDR    = src/spartan.h src/stream.h src/config.h

BIN = build

# The run most targets work on. Any cfg works:
#     make compare CFG=tests/tsb/cfg/tsb_noisy.cfg
# out_dir and words are READ from the cfg, so this file never carries a copy
# of where a run writes
CFG     = tests/tsb/cfg/tsb_drift.cfg
NAME    = $(basename $(notdir $(CFG)))
OUT_DIR = $(shell sed -n 's/^[[:space:]]*out_dir[[:space:]]*=[[:space:]]*//p' $(CFG))
WORDS   = $(shell sed -n 's/^[[:space:]]*words[[:space:]]*=[[:space:]]*//p' $(CFG))
GOLD    = $(dir $(patsubst %/,%,$(dir $(CFG))))expected/$(NAME)

# The summary lines worth guarding. Defined ONCE and used by run, check and
# golden. "time" is deliberately absent: it changes every run
KEEP = '^(ev|bits|alphabet|score|windows|drift|noise|rate|overshoot|epochs|late|staleness)'

all: $(BIN)/spartan_toy

# Order-only prerequisite, after the pipe. Writing any file into build/ updates
# that directory's timestamp; as a normal prerequisite it would mark every
# binary out of date on every single run
$(BIN):
	@mkdir -p $(BIN)

# The headers are prerequisites on purpose: a change to spartan.h alone must
# force a rebuild, or you link code built against an older struct Config
$(BIN)/spartan_toy: $(SRC) $(HDR) | $(BIN)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDLIBS)

# The previous re-basing design (predict, stage, promote), compiled out of
# the default build. Kept to compare on real data
$(BIN)/spartan_A: $(SRC) $(HDR) | $(BIN)
	$(CC) $(CFLAGS) -DSTAGING_A $(SRC) -o $@ $(LDLIBS)

$(BIN)/spartan_toy_dbg: $(SRC) $(HDR) | $(BIN)
	$(CC) $(DBGFLAGS) $(SRC) -o $@ $(LDLIBS)

# These need -Isrc and the driver does not. A quoted include searches the
# folder of the file doing the including first: src/main.c finds src/spartan.h
# for free, tests/unit/test_daa.c does not
$(BIN)/test_daa: tests/unit/test_daa.c $(LIBSRC) $(HDR) | $(BIN)
	$(CC) $(DBGFLAGS) -Isrc $< $(LIBSRC) -o $@ $(LDLIBS)

$(BIN)/test_ring: tests/unit/test_ring.c $(LIBSRC) $(HDR) | $(BIN)
	$(CC) $(DBGFLAGS) -Isrc $< $(LIBSRC) -o $@ $(LDLIBS)

A: $(BIN)/spartan_A

debug: $(BIN)/spartan_toy_dbg

unit: $(BIN)/test_daa $(BIN)/test_ring
	@./$(BIN)/test_daa
	@./$(BIN)/test_ring

# The four synthetic streams, their cfgs and their out folders
data:
	$(PYTHON) tools/make_tsb.py

# One run of the C. Words go where the cfg says, the full log to run.log,
# the summary to the terminal
run: $(BIN)/spartan_toy
	@mkdir -p $(OUT_DIR)
	@./$(BIN)/spartan_toy $(CFG) 2> $(OUT_DIR)/run.log
	@grep -E $(KEEP) $(OUT_DIR)/run.log

# The whole comparison for one cfg: our run, both batch references, the
# report. compare_stages re-derives every word from the raw stream and stops
# on a single mismatch, so this is also the verification of the run
compare: run
	@$(PYTHON) tools/ref_spartan.py $(CFG) --spartan-repo $(SPARTAN_REPO) > $(OUT_DIR)/ref.log
	@$(PYTHON) tools/compare_stages.py $(CFG) --spartan-repo $(SPARTAN_REPO) --out $(OUT_DIR)/cmp

# What batch PCA says about the stream, and for a make_tsb stream the angle
# to what was planted. Needs the C to have run on the cfg
truth: run
	@$(PYTHON) tools/batch_truth.py $(CFG)

# The sensitivity sweep. Datasets and levels live in the CONTROL PANEL at
# the top of tools/sweep.py
sweep: $(BIN)/spartan_toy
	$(PYTHON) tools/sweep.py --bin ./$(BIN)/spartan_toy --spartan-repo $(SPARTAN_REPO)

sweep-plan: $(BIN)/spartan_toy
	$(PYTHON) tools/sweep.py --bin ./$(BIN)/spartan_toy --spartan-repo $(SPARTAN_REPO) --dry-run

# Regression: the sanitized build on CFG against the golden words and
# summary in tests/<set>/expected/<name>/. Says nothing about whether the
# output is right, only that a change did not alter it
check: $(BIN)/spartan_toy_dbg
	@mkdir -p $(OUT_DIR)
	@./$(BIN)/spartan_toy_dbg $(CFG) 2> $(OUT_DIR)/check.log
	@grep -E $(KEEP) $(OUT_DIR)/check.log > $(OUT_DIR)/summary.txt
	@diff $(GOLD)/words.csv   $(WORDS)                && echo "words    OK"
	@diff $(GOLD)/summary.txt $(OUT_DIR)/summary.txt  && echo "summary  OK"

golden: $(BIN)/spartan_toy_dbg
	@mkdir -p $(GOLD) $(OUT_DIR)
	@./$(BIN)/spartan_toy_dbg $(CFG) 2> $(OUT_DIR)/golden.log
	@cp $(WORDS) $(GOLD)/words.csv
	@grep -E $(KEEP) $(OUT_DIR)/golden.log > $(GOLD)/summary.txt
	@echo "golden updated in $(GOLD), read it before you commit it:"
	@cat $(GOLD)/summary.txt

# Build products and run outputs only. Sweep results in out/ are kept: they
# take tens of minutes to make
clean:
	rm -rf $(BIN)
	rm -rf tests/*/out

.PHONY: all A debug unit data run compare truth sweep sweep-plan check golden clean
