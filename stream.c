#define _POSIX_C_SOURCE 200809L

#include "stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declaration of internal helper
static int fetch_frame(WindowStream *s, double *dst);

// Initializes a stream sourced from a text file
int stream_init_file(WindowStream *s, const char *path, int d, int m_time,
                     int stride) {
  if (!s || !path || d < 1 || m_time < 1 || stride < 1 || stride > m_time) {
    return -1;
  }

  // Reset struct state to zero
  memset(s, 0, sizeof(*s));

  FILE *fp = fopen(path, "r");
  if (!fp) {
    return -1;
  }

  // Allocate continuous buffer for the ring
  s->ring = calloc((size_t)m_time * (size_t)d, sizeof(double));
  if (!s->ring) {
    fclose(fp);
    return -1;
  }

  s->d = d;
  s->m_time = m_time;
  s->stride = stride;
  s->src_kind = STREAM_SRC_FILE;
  s->fp = fp;

  return 0;
}

// Calculates total extractable sliding windows for in-memory streams
long stream_window_count(long n_src, int d, int m_time, int stride) {
  long n_frames = n_src / d;
  return (n_frames >= m_time) * (1 + (n_frames - m_time) / stride);
}

// Initializes a stream sourced from an in-memory array
int stream_init_array(WindowStream *s, const double *src, long n_src, int d,
                      int m_time, int stride) {
  if (!s || !src || d < 1 || m_time < 1 || stride < 1 || stride > m_time ||
      (n_src % d) != 0) {
    return -1;
  }

  // Reset struct state to zero
  memset(s, 0, sizeof(*s));

  // Allocate continuous buffer for the ring
  s->ring = calloc((size_t)m_time * (size_t)d, sizeof(double));
  if (!s->ring) {
    return -1;
  }

  s->d = d;
  s->m_time = m_time;
  s->stride = stride;
  s->src_kind = STREAM_SRC_ARRAY;
  s->src = src;
  s->n_src = n_src;

  return 0;
}

// Releases all allocated memory and open file descriptors
void stream_free(WindowStream *s) {
  if (!s)
    return;

  free(s->ring);

  if (s->src_kind == STREAM_SRC_FILE) {
    if (s->fp) {
      fclose(s->fp);
    }
    free(s->line_buf);
  }

  // Reset whole structure to prevent dangling references
  memset(s, 0, sizeof(*s));
}

// Fetches a single frame (d values) from the active stream source
static int fetch_frame(WindowStream *s, double *dst) {
  if (!s || !dst)
    return 0;

  // Route 1: Memory array source
  if (s->src_kind == STREAM_SRC_ARRAY) {
    if (s->pos + s->d > s->n_src) {
      return 0;
    }
    memcpy(dst, &s->src[s->pos], (size_t)s->d * sizeof(double));
    s->pos += s->d;
    return 1;
  }

  // Route 2: Line-by-line text file source
  if (s->src_kind == STREAM_SRC_FILE) {
    ssize_t nread;
    while ((nread = getline(&s->line_buf, &s->line_cap, s->fp)) != -1) {
      char *p = s->line_buf;

      // Skip leading whitespace characters
      while (*p == ' ' || *p == '\t')
        p++;

      // Ignore blank lines and comment lines
      if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') {
        continue;
      }

      int got = 0;
      char *end;

      // Parse numeric values sequentially
      for (;;) {
        double val = strtod(p, &end);
        if (end == p) {
          break;
        }

        if (got < s->d) {
          dst[got] = val;
        }
        got++;

        p = end;
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == ';')
          p++;
      }

      // Reject frame if channel count does not match exactly
      if (got != s->d) {
        s->err = 1;
        return 0;
      }
      // Trailing junk check: consume spaces, then require end-of-line or
      // comment
      while (*p == ' ' || *p == '\t') {
        p++;
      }
      if (*p != '\0' && *p != '\n' && *p != '\r' && *p != '#') {
        s->err = 1;
        return 0;
      }

      return 1;
    }

    // Flag potential stream read failures
    if (ferror(s->fp)) {
      s->err = 1;
    }
    return 0;
  }

  return 0;
}

// Advances the stream and exports the unrolled window matrix
int stream_next(WindowStream *s, double *out) {
  if (!s || !out)
    return 0;

  int need = s->primed ? s->stride : s->m_time;

  // Pull incoming frames into the circular ring buffer
  for (int f = 0; f < need; f++) {
    if (!fetch_frame(s, &s->ring[s->cur * s->d])) {
      return 0;
    }
    if (++s->cur == s->m_time) {
      s->cur = 0;
    }
  }

  s->primed = 1;
  s->n_frames += need;

  // Split ring indices into contiguous blocks
  int n1 = s->m_time - s->cur;
  int n2 = s->cur;

  // Export with Mode-1 unfolding (channels first, time contiguous)
  for (int c = 0; c < s->d; c++) {
    double *dst = out + c * s->m_time;

    for (int i = 0; i < n1; i++) {
      dst[i] = s->ring[(s->cur + i) * s->d + c];
    }
    for (int i = 0; i < n2; i++) {
      dst[n1 + i] = s->ring[i * s->d + c];
    }
  }

  return 1;
}
