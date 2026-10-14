CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -Wshadow -std=c11
DBGFLAGS = -g -O0 -Wall -Wextra -Wshadow -std=c11 -fsanitize=address,undefined
LDLIBS   = -lm
SRC      = spartan.c stream.c config.c main.c
HDR      = spartan.h stream.h config.h

# The summary lines worth guarding. Defined ONCE and used by both check and
# golden, so the two can never end up asking for different things
KEEP = '^(ev|bits|alphabet|score|windows|drift|noise|rate)'

all: spartan_toy

# Real file targets with real prerequisites. The headers are in the list on
# purpose: a change to spartan.h alone must force a rebuild, or you link object
# code built against an older struct Config and get numbers read from the
# wrong offsets
spartan_toy: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDLIBS)

spartan_toy_dbg: $(SRC) $(HDR)
	$(CC) $(DBGFLAGS) $(SRC) -o $@ $(LDLIBS)

# An alias, so "make debug" still works and anything depending on the binary
# rebuilds it properly
debug: spartan_toy_dbg

check: spartan_toy_dbg
	@mkdir -p debug_tests/out
	@./spartan_toy_dbg > debug_tests/out/words.csv 2> debug_tests/out/run.log
	@grep -E $(KEEP) debug_tests/out/run.log > debug_tests/out/summary.txt
	@diff debug_tests/expected/demo_words.csv   debug_tests/out/words.csv   && echo "words    OK"
	@diff debug_tests/expected/demo_summary.txt debug_tests/out/summary.txt && echo "summary  OK"

golden: spartan_toy_dbg
	@mkdir -p debug_tests/expected debug_tests/out
	@./spartan_toy_dbg > debug_tests/expected/demo_words.csv 2> debug_tests/out/golden.log
	@grep -E $(KEEP) debug_tests/out/golden.log > debug_tests/expected/demo_summary.txt
	@echo "golden updated, read it before you commit it:"
	@cat debug_tests/expected/demo_summary.txt

clean:
	rm -f spartan_toy spartan_toy_dbg
	rm -rf debug_tests/out

.PHONY: all debug check golden clean
