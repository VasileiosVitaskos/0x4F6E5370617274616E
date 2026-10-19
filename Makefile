CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -Wshadow -std=c11
DBGFLAGS = -g -O0 -Wall -Wextra -Wshadow -std=c11 -fsanitize=address,undefined
LDLIBS   = -lm

# Everything except main.c, so a unit test can link the same code without
# dragging in a second main()
LIBSRC = src/spartan.c src/stream.c src/config.c
SRC    = $(LIBSRC) src/main.c
HDR    = src/spartan.h src/stream.h src/config.h

BIN = build

# Which test folder check and golden work on. All three have the same shape,
# so the rules below are written once:
#     make check
#     make check TEST=synthetic CFG=tests/synthetic/cfg/stationary.cfg
# An empty CFG means no argument, which is the built-in demo stream
TEST = demo
TDIR = tests/$(TEST)
CFG  =

# The summary lines worth guarding. Defined ONCE and used by both check and
# golden, so the two can never end up asking for different things. "time" is
# deliberately absent: it changes every run and would make the golden useless
KEEP = '^(ev|bits|alphabet|score|windows|drift|noise|rate|overshoot|epochs|late|staleness)'

all: $(BIN)/spartan_toy

# Order-only prerequisite, after the pipe. Writing any file into build/ updates
# that directory's timestamp; as a normal prerequisite it would mark every
# binary out of date on every single run. After the pipe, make only checks that
# it exists
$(BIN):
	@mkdir -p $(BIN)

# Real file targets with real prerequisites. The headers are in the list on
# purpose: a change to spartan.h alone must force a rebuild, or you link object
# code built against an older struct Config and read numbers from the wrong
# offsets
$(BIN)/spartan_toy: $(SRC) $(HDR) | $(BIN)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDLIBS)

$(BIN)/spartan_toy_dbg: $(SRC) $(HDR) | $(BIN)
	$(CC) $(DBGFLAGS) $(SRC) -o $@ $(LDLIBS)

# These need -Isrc and the driver does not. A quoted include searches the
# folder of the file doing the including first: src/main.c finds src/spartan.h
# for free, tests/unit/test_daa.c does not
$(BIN)/test_daa: tests/unit/test_daa.c $(LIBSRC) $(HDR) | $(BIN)
	$(CC) $(DBGFLAGS) -Isrc $< $(LIBSRC) -o $@ $(LDLIBS)

$(BIN)/test_ring: tests/unit/test_ring.c $(LIBSRC) $(HDR) | $(BIN)
	$(CC) $(DBGFLAGS) -Isrc $< $(LIBSRC) -o $@ $(LDLIBS)

debug: $(BIN)/spartan_toy_dbg

unit: $(BIN)/test_daa $(BIN)/test_ring
	@./$(BIN)/test_daa
	@./$(BIN)/test_ring

check: $(BIN)/spartan_toy_dbg
	@mkdir -p $(TDIR)/out
	@./$(BIN)/spartan_toy_dbg $(CFG) > $(TDIR)/out/words.csv 2> $(TDIR)/out/run.log
	@grep -E $(KEEP) $(TDIR)/out/run.log > $(TDIR)/out/summary.txt
	@diff $(TDIR)/expected/words.csv   $(TDIR)/out/words.csv   && echo "words    OK"
	@diff $(TDIR)/expected/summary.txt $(TDIR)/out/summary.txt && echo "summary  OK"

golden: $(BIN)/spartan_toy_dbg
	@mkdir -p $(TDIR)/expected $(TDIR)/out
	@./$(BIN)/spartan_toy_dbg $(CFG) > $(TDIR)/expected/words.csv 2> $(TDIR)/out/golden.log
	@grep -E $(KEEP) $(TDIR)/out/golden.log > $(TDIR)/expected/summary.txt
	@echo "golden updated, read it before you commit it:"
	@cat $(TDIR)/expected/summary.txt

clean:
	rm -rf $(BIN)
	rm -rf tests/*/out

.PHONY: all debug unit check golden clean
