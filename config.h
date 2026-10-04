#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>

#include "spartan.h"

/* Run-level settings: where data comes from and where results go.
 * Everything the ALGORITHM needs lives in Config (spartan.h); everything
 * about files and formatting lives here. Two structs, two concerns.
 */

#define CFG_PATH_MAX 512

typedef struct {
  char input[CFG_PATH_MAX];     /* data file, or "builtin" for the demo array */
  char words_out[CFG_PATH_MAX]; /* where the symbols go; "-" means stdout   */
  char dict_out[CFG_PATH_MAX];  /* where the frozen dictionary goes; "" off */
  int pretty;                   /* 1 = letters for humans, 0 = numbers      */
} RunOptions;

/* Fills both structs with the defaults used when no file is given. */
void config_defaults(Config *cfg, RunOptions *run);

/* Reads a "key = value" file over the defaults, then resolves the settings
 * whose default depends on another setting, then validates everything.
 *
 * Returns 0 on success, -1 on any problem. Every failure prints one line to
 * stderr naming the key, the offending value and the line number.
 */
int config_load(const char *path, Config *cfg, RunOptions *run);

/* Prints the settings actually in force, as a valid config file. Handy for
 * the log of an experiment: the run describes itself. */
void config_dump(FILE *f, const Config *cfg, const RunOptions *run);

#endif /* CONFIG_H */
