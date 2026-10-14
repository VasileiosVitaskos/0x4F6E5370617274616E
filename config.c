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
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
/* A value that means "the user did not say". Resolved after parsing, because
 * some defaults depend on other settings and the file may list them in any
 * order. */

#define UNSET LONG_MIN

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
  if (c->ring_w < 2)
    return fail("ring_window must be at least 2", "ring_window", c->ring_w);
  if (c->total_bits > c->k * MAX_BITS)
    return fail("total_bits cannot be above components * MAX_BITS",
                "total_bits", c->total_bits);
  if (c->gs_period < 1)
    return fail("gs_period must be at least 1", "gs_period", c->gs_period);
  if (c->angle_period < 1)
    return fail("angle_period must be at least 1", "angle_period",
                c->angle_period);
  if (c->angle_period % c->gs_period != 0)
    return fail(
        "angle_period must be a multiple of gs_period, otherwise the "
        "angle is read on a basis that is not orthonormal and the "
        "numbers are not angles",
        "angle_period", c->angle_period);
  if (c->step <= 0.0) {
    fprintf(stderr, "config: step must be positive (got %g)\n", c->step);
    return -1;
  }
  if (c->beta < 0.0 || c->beta > 1.0) {
    fprintf(stderr, "config: basis_average must be between 0 and 1 (got %g)\n",
            c->beta);
    return -1;
  }
  if (!(c->theta_threshold_deg > 0.0) || !(c->theta_threshold_deg < 90.0)) {
    fprintf(stderr,
            "config: theta_threshold_deg must be above 0 and below 90 "
            "(got %g). At 90 the two subspaces are orthogonal, the old words "
            "carry nothing about the new one and there is nothing to "
            "translate: that is the degenerate case, not a tight setting\n",
            c->theta_threshold_deg);
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
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      run->pretty = (iv != 0);
    } else if (strcmp(key, "channels") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      cfg->d = (int)iv;
    } else if (strcmp(key, "window") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      cfg->m_time = (int)iv;
    } else if (strcmp(key, "stride") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      stride = iv;
    } else if (strcmp(key, "components") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      cfg->k = (int)iv;
    } else if (strcmp(key, "total_bits") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      total_bits = iv;
    } else if (strcmp(key, "warmup") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      cfg->warmup = (int)iv;
    } else if (strcmp(key, "gs_period") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      cfg->gs_period = (int)iv;
    } else if (strcmp(key, "angle_period") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      cfg->angle_period = (int)iv;
    } else if (strcmp(key, "step") == 0) {
      if (parse_double(val, &dv) != 0) goto bad_value;
      cfg->step = dv;
    } else if (strcmp(key, "basis_average") == 0) {
      if (parse_double(val, &dv) != 0) goto bad_value;
      cfg->beta = dv;
    } else if (strcmp(key, "theta_threshold_deg") == 0) {
      if (parse_double(val, &dv) != 0) goto bad_value;
      cfg->theta_threshold_deg = dv;
    } else if (strcmp(key, "learning_rate") == 0) {
      fprintf(stderr,
              "config: line %d, 'learning_rate' no longer exists. Use 'step', "
              "which is divided by the window length internally: the old "
              "constant silently meant a different thing at every window "
              "size. 'step = 0.24' is the measured default.\n",
              lineno);
      rc = -1;
      break;
    } else if (strcmp(key, "lambda") == 0) {
      if (parse_double(val, &dv) != 0) goto bad_value;
      cfg->lambda = dv;
    } else if (strcmp(key, "ring_window") == 0) {
      if (parse_long(val, &iv) != 0 || iv < INT_MIN || iv > INT_MAX)
        goto bad_value;
      cfg->ring_w = (int)iv;
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

  /* And so is the Oja step. After the per window z-norm every window has
     length exactly sqrt(m), so the projections grow with m and the step the
     rule actually takes is step * m, not step. Dividing here is what makes one
     setting mean the same thing at every window size.

     IT IS NO ACCIDENT that the old fixed learning_rate = 0.01 looked fine for
     so long: the built-in demo has window = 8, where it works out to an
     effective step of 0.08, the quiet end of the range. At window 32 the same
     number means 0.32, at window 128 it means 1.28, which is far past where
     the directions stop settling at all. The demo was the one size that
     flattered it, and every measurement taken there was the easiest case we
     own. Measured subspace error at window 16 / 32 / 64 / 128 with the old
     fixed step: 0.135, 0.241, 0.440, 1.136. With step / m and the averaged
     basis: 0.089, 0.108, 0.161, 0.219. */
  cfg->lr = cfg->step / cfg->m;
  cfg->theta_threshold = cfg->theta_threshold_deg * SPARTAN_DEG_TO_RAD;

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
  fprintf(f, "lambda        = %.*g\n", DBL_DECIMAL_DIG, cfg->lambda);
  fprintf(f, "step          = %.*g\n", DBL_DECIMAL_DIG, cfg->step);
  fprintf(f, "basis_average = %.*g\n", DBL_DECIMAL_DIG, cfg->beta);
  fprintf(f, "theta_threshold_deg = %.*g\n", DBL_DECIMAL_DIG,
          cfg->theta_threshold_deg);
  fprintf(f, "# effective Oja step, step / window = %g\n", cfg->lr);
  fprintf(f, "gs_period     = %d\n", cfg->gs_period);
  fprintf(f, "angle_period  = %d\n", cfg->angle_period);
  fprintf(f, "warmup        = %d\n", cfg->warmup);
  fprintf(f, "words         = %s\n", run->words_out);
  fprintf(f, "dictionary    = %s\n", run->dict_out[0] ? run->dict_out : "");
  fprintf(f, "pretty        = %d\n", run->pretty);
  fprintf(f, "ring_window   = %d\n", cfg->ring_w);
  /* The block is self delimiting: the header above and this line below bracket
   it, so a log can be sliced back into a settings file without guessing
   which lines belong. Both are comments, so the slice is valid as it stands */
  fprintf(f, "# end of settings\n");
}
