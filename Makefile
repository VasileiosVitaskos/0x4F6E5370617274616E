CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -Wshadow -std=c11
DBGFLAGS= -g -O0 -Wall -Wextra -Wshadow -std=c11 -fsanitize=address,undefined
LDLIBS  = -lm
SRC     = spartan.c main.c

all: spartan_toy

spartan_toy: $(SRC) spartan.h
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDLIBS)

debug: $(SRC) spartan.h
	$(CC) $(DBGFLAGS) $(SRC) -o spartan_toy_dbg $(LDLIBS)

clean:
	rm -f spartan_toy spartan_toy_dbg

.PHONY: all debug clean
