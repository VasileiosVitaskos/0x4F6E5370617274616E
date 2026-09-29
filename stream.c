#include "stream.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

long stream_window_count(long n_src, int d, int m_time, int stride) {
  long n_frames = n_src / d;
  return (n_frames >= m_time) * (1 + (n_frames - m_time) / stride);
}

int stream_init_array(WindowStream* s, const double* src, long n_src, int d,
                      int m_time, int stride) {
  if (!src || !s || d < 1 || m_time < 1 || stride < 1 || stride > m_time ||
      (n_src % d) != 0)
    return -1;

  size_t total_elements = (size_t)m_time * d;

  s->ring = (double*)aligned_alloc(64, total_elements * sizeof(double));

  if (!s->ring) {
    return -1;
  }
  s->d = d;
  s->m_time = m_time;
  s->stride = stride;
  s->cur = 0;
  s->primed = 0;
  s->pos = 0;
  s->n_frames = 0;
  s->src = src;
  s->n_src = n_src;
  s->src_kind = STREAM_SRC_ARRAY;
  return 0;
}

void stream_free(WindowStream* s) {
  free(s->ring);
  s->ring = NULL;
  s->cur = 0;
  s->primed = 0;
  s->pos = 0;
  s->n_frames = 0;
}
int stream_next(WindowStream* s, double* out) {
  int need = s->primed ? s->stride : s->m_time;
  if (s->pos + (long)need * s->d > s->n_src) {
    return 0;
  }

  size_t frame_bytes = (size_t)s->d * sizeof(double);
  for (int f = 0; f < need; f++) {
    memcpy(&s->ring[s->cur * s->d], &s->src[s->pos], frame_bytes);
    s->pos += s->d;
    if (++s->cur == s->m_time) {
      s->cur = 0;
    }
  }

  s->primed = 1;
  s->n_frames += need;

  int n1 = s->m_time - s->cur;
  int n2 = s->cur;

  for (int c = 0; c < s->d; c++) {
    double* dst = out + c * s->m_time;

    for (int i = 0; i < n1; i++) {
      dst[i] = s->ring[(s->cur + i) * s->d + c];
    }
    for (int i = 0; i < n2; i++) {
      dst[n1 + i] = s->ring[i * s->d + c];
    }
  }

  return 1;
}
