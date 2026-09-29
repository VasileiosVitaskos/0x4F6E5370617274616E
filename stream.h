#ifndef STREAM_H
#define STREAM_H
#include <stdio.h>

typedef enum { STREAM_SRC_ARRAY = 0, STREAM_SRC_FILE = 1 } StreamSourceKind;

typedef struct {
  // Existing ring buffer fields
  double* ring;
  int m_time;
  int d;
  int stride;
  int cur;
  int primed;
  long n_frames;

  // Source selection
  StreamSourceKind src_kind;

  // Array source fields (used only when src_kind == STREAM_SRC_ARRAY)
  const double* src;
  long n_src;
  long pos;

  // File source fields (used only when src_kind == STREAM_SRC_FILE)
  FILE* fp;
  int err;
  char* line_buf;
  size_t line_cap;
} WindowStream;

int stream_init_array(WindowStream* s, const double* src, long n_src, int d,
                      int m_time, int stride);
int stream_init_file(WindowStream* s, const char* path, int d, int m_time,
                     int stride);
void stream_free(WindowStream* s);
int stream_next(WindowStream* s, double* out);
long stream_window_count(long n_src, int d, int m_time, int stride);

#endif
