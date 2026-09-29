/* Reading the settings file. Knows nothing about the algorithm beyond the
 * names of its knobs.
 *
 * Format: one "key = value" per line. Anything after '#' is a comment.
 * Blank lines are ignored. An unknown key is an ERROR, not a shrug: a typo
 * in a settings file must never be silently ignored, or you will spend an
 * afternoon wondering why a knob did nothing.
 */

#include "config.h"

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* A value that means "the user did not say". Resolved after parsing, because
 * some defaults depend on other settings and the file may list them in any
 * order. */
#define UNSET (-1)

/* ---------------- small parsing helpers ---------------- */

/* Moves the start forward and writes a '\0' over the trailing blanks. */
static char* trim(char* s) {
  while (*s && isspace((unsigned char)*s)) s++;
  char* e = s + strlen(s);
  while (e > s && isspace((unsigned char)e[-1])) e--;
  *e = '\0';
  return s;
}

/* strtol with the three checks people forget: nothing read, junk after the
 * number, out of range. atoi has none of them and returns 0 on garbage. */
static int parse_long(const char* text, long* out) {
  errno = 0;
  char* end;
  long v = strtol(text, &end, 10);
  if (end == text) return -1;
  while (*end && isspace((unsigned char)*end)) end++;
  if (*end != '\0') return -1;
  if (errno == ERANGE) return -1;
  *out = v;
  return 0;
}

static int parse_double(const char* text, double* out) {
  errno = 0;
  char* end;
  double v = strtod(text, &end);
  if (end == text) return -1;
  while (*end && isspace((unsigned char)*end)) end++;
  if (*end != '\0') return -1;
  if (errno == ERANGE) return -1;
  *out = v;
  return 0;
}

static void copy_path(char* dst, const char* src) {
  snprintf(dst, CFG_PATH_MAX, "%s", src);
}

/* ---------------- defaults ---------------- */

void config_defaults(Config* cfg, RunOptions* run) {
  *cfg = config_default_mv(3, 1, 8);
  copy_path(run->input, "builtin");
  copy_path(run->words_out, "-");
  copy_path(run->dict_out, "");
  run->pretty = 0;
}

/* ---------------- validation ---------------- */

static int fail(const char* msg, const char* key, long got) {
  fprintf(stderr, "config: %s (%s = %ld)\n", msg, key, got);
  return -1;
}

static int validate(const Config* c) {
  if (c->k < 1)
    return fail("components must be at least 1", "components", c->k);
  if (c->d < 1) return fail("channels must be at least 1", "channels", c->d);
  if (c->m_time < 1)
    return fail("window must be at least 1", "window", c->m_time);
  if (c->k > c->m)
    return fail("components cannot exceed channels * window", "components",
                c->k);
  if (c->stride < 1 || c->stride > c->m_time)
    return fail("stride must be between 1 and window", "stride", c->stride);
  if (c->warmup < 1)
    return fail("warmup must be at least 1", "warmup", c->warmup);
  if (c->total_bits < c->k)
    return fail("total_bits cannot be below components", "total_bits",
                c->total_bits);
  if (c->gs_period < 1)
    return fail("gs_period must be at least 1", "gs_period", c->gs_period);
  if (c->lr <= 0.0) {
    fprintf(stderr, "config: learning_rate must be positive (got %g)\n", c->lr);
    return -1;
  }
  if (c->lambda < 0.0) {
    fprintf(stderr, "config: lambda cannot be negative (got %g)\n", c->lambda);
    return -1;
  }
  return 0;
}

/* ---------------- the reader ---------------- */

int config_load(const char* path, Config* cfg, RunOptions* run) {
  config_defaults(cfg, run);

  /* Settings whose default depends on another one. Left UNSET until the whole
     file has been read, so the order of lines cannot change the outcome. */
  long stride = UNSET, total_bits = UNSET;

  FILE* f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "config: cannot open '%s'\n", path);
    return -1;
  }

  char line[1024];
  int lineno = 0;
  int rc = 0;

  while (fgets(line, sizeof line, f)) {
    lineno++;

    char* hash = strchr(line, '#');
    if (hash) *hash = '\0';

    char* body = trim(line);
    if (*body == '\0') continue;

    char* eq = strchr(body, '=');
    if (!eq) {
      fprintf(stderr, "config: line %d is not 'key = value'\n", lineno);
      rc = -1;
      break;
    }
    *eq = '\0';
    char* key = trim(body);
    char* val = trim(eq + 1);

    long iv;
    double dv;

    if (strcmp(key, "input") == 0) {
      copy_path(run->input, val);
    } else if (strcmp(key, "words") == 0) {
      copy_path(run->words_out, val);
    } else if (strcmp(key, "dictionary") == 0) {
      copy_path(run->dict_out, val);
    } else if (strcmp(key, "pretty") == 0) {
      if (parse_long(val, &iv) != 0) goto bad_value;
      run->pretty = (iv != 0);
    } else if (strcmp(key, "channels") == 0) {
      if (parse_long(val, &iv) != 0) goto bad_value;
      cfg->d = (int)iv;
    } else if (strcmp(key, "window") == 0) {
      if (parse_long(val, &iv) != 0) goto bad_value;
      cfg->m_time = (int)iv;
    } else if (strcmp(key, "stride") == 0) {
      if (parse_long(val, &iv) != 0) goto bad_value;
      stride = iv;
    } else if (strcmp(key, "components") == 0) {
      if (parse_long(val, &iv) != 0) goto bad_value;
      cfg->k = (int)iv;
    } else if (strcmp(key, "total_bits") == 0) {
      if (parse_long(val, &iv) != 0) goto bad_value;
      total_bits = iv;
    } else if (strcmp(key, "warmup") == 0) {
      if (parse_long(val, &iv) != 0) goto bad_value;
      cfg->warmup = (int)iv;
    } else if (strcmp(key, "gs_period") == 0) {
      if (parse_long(val, &iv) != 0) goto bad_value;
      cfg->gs_period = (int)iv;
    } else if (strcmp(key, "learning_rate") == 0) {
      if (parse_double(val, &dv) != 0) goto bad_value;
      cfg->lr = dv;
    } else if (strcmp(key, "lambda") == 0) {
      if (parse_double(val, &dv) != 0) goto bad_value;
      cfg->lambda = dv;
    } else {
      fprintf(stderr, "config: line %d, unknown key '%s'\n", lineno, key);
      rc = -1;
      break;
    }
    continue;

  bad_value:
    fprintf(stderr, "config: line %d, bad value for '%s': '%s'\n", lineno, key,
            val);
    rc = -1;
    break;
  }

  fclose(f);
  if (rc != 0) return rc;

  /* m is derived, never given directly. */
  cfg->m = cfg->d * cfg->m_time;

  /* Now the dependent defaults, once everything else is known. */
  cfg->stride = (stride == UNSET) ? cfg->m_time : (int)stride;
  cfg->total_bits = (total_bits == UNSET) ? 2 * cfg->k : (int)total_bits;

  return validate(cfg);
}

/* ---------------- echo ---------------- */

void config_dump(FILE* f, const Config* cfg, const RunOptions* run) {
  fprintf(f, "# settings in force for this run\n");
  fprintf(f, "input         = %s\n", run->input);
  fprintf(f, "channels      = %d\n", cfg->d);
  fprintf(f, "window        = %d\n", cfg->m_time);
  fprintf(f, "stride        = %d\n", cfg->stride);
  fprintf(f, "components    = %d\n", cfg->k);
  fprintf(f, "total_bits    = %d\n", cfg->total_bits);
  fprintf(f, "lambda        = %g\n", cfg->lambda);
  fprintf(f, "learning_rate = %g\n", cfg->lr);
  fprintf(f, "gs_period     = %d\n", cfg->gs_period);
  fprintf(f, "warmup        = %d\n", cfg->warmup);
  fprintf(f, "words         = %s\n", run->words_out);
  fprintf(f, "dictionary    = %s\n", run->dict_out[0] ? run->dict_out : "");
  fprintf(f, "pretty        = %d\n", run->pretty);
}
