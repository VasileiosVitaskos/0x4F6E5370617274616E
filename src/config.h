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
  char out_dir[CFG_PATH_MAX];
  int pretty; /* 1 = letters for humans, 0 = numbers      */
} RunOptions;

/**
 * @brief Initializes algorithm and runtime configuration structures with safe
 * default parameters.
 *
 * Sets up a baseline 3-component univariate configuration with a window size of
 * 8 and binds I/O paths to the built-in synthetic array and standard output.
 *
 * @param cfg Pointer to the algorithm configuration structure to populate.
 * @param run Pointer to the runtime I/O options structure to populate.
 * @return void
 */
void config_defaults(Config* cfg, RunOptions* run);

/**
 * @brief Parses a key-value configuration file and validates all derived
 * mathematical invariants.
 *
 * Overwrites default parameters with user-defined settings from disk, resolves
 * order-independent dependent settings (such as flattened window
 * dimensionality, effective Oja learning rate scaled by window length, and
 * default bit budgets), and enforces strict domain bounds.
 *
 * @param path File system path to the configuration file.
 * @param cfg Pointer to the algorithm configuration structure to update.
 * @param run Pointer to the runtime I/O options structure to update.
 * @return 0 on success, or -1 on file I/O error, syntax error, unknown key, or
 * domain violation.
 */
int config_load(const char* path, Config* cfg, RunOptions* run);

/**
 * @brief Serializes the active configuration to an output stream as a valid,
 * self-delimiting config file.
 *
 * Emits all active algorithm parameters and runtime paths bracketed by header
 * and footer comment markers, allowing experiment logs to be sliced and re-fed
 * directly as input configs.
 *
 * @param f Output file stream pointer (e.g., stderr or a log file).
 * @param cfg Pointer to the active algorithm configuration structure.
 * @param run Pointer to the active runtime I/O options structure.
 * @return void
 */
void config_dump(FILE* f, const Config* cfg, const RunOptions* run);

#endif /* CONFIG_H */
