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
/**
 * @brief Initializes a sliding window stream backed by a contiguous in-memory
 * array.
 *
 * Allocates an internal circular ring buffer of size (m_time * d) to stage
 * incoming multivariate frames and binds the stream reader to a static memory
 * buffer.
 *
 * @param s Pointer to the WindowStream instance to initialize.
 * @param src Pointer to the source array of interleaved multivariate frames.
 * @param n_src Total number of double elements in the source array (must be a
 * multiple of d).
 * @param d Number of channels (variables) per time frame.
 * @param m_time Number of time frames per sliding window.
 * @param stride Step size in frames between consecutive windows (must be in [1,
 * m_time]).
 * @return 0 on success, or -1 on invalid arguments or memory allocation
 * failure.
 */
int stream_init_array(WindowStream* s, const double* src, long n_src, int d,
                      int m_time, int stride);
/**
 * @brief Initializes a sliding window stream backed by a delimited text or CSV
 * file.
 *
 * Opens the target file stream and allocates an internal circular ring buffer
 * of size (m_time * d) to incrementally ingest and unfold multivariate frames
 * without loading the entire dataset into memory.
 *
 * @param s Pointer to the WindowStream instance to initialize.
 * @param path File system path to the input text/CSV data stream.
 * @param d Number of channels (columns) expected per line.
 * @param m_time Number of time frames per sliding window.
 * @param stride Step size in frames between consecutive windows (must be in [1,
 * m_time]).
 * @return 0 on success, or -1 on invalid arguments, file open failure, or
 * allocation failure.
 */
int stream_init_file(WindowStream* s, const char* path, int d, int m_time,
                     int stride);
/**
 * @brief Releases all heap buffers and file descriptors owned by a
 * WindowStream.
 *
 * Frees the internal circular ring buffer, closes the underlying file handle
 * and line buffer if sourced from disk, and zeroes the structure to prevent
 * dangling pointers.
 *
 * @param s Pointer to the WindowStream instance to clean up.
 * @return void
 */
void stream_free(WindowStream* s);
/**
 * @brief Advances the sliding window by one stride and exports the unfolded
 * window matrix.
 *
 * Pulls `m_time` frames on the initial call and `stride` frames on subsequent
 * calls into the circular buffer, then exports the active window into `out`
 * using Mode-1 unfolding (channels contiguous, time contiguous within each
 * channel).
 *
 * @param s Pointer to the active WindowStream instance.
 * @param out Output buffer of size (d * m_time) to receive the Mode-1 unfolded
 * window.
 * @return 1 if a full window was produced, or 0 on EOF or stream parsing error
 * (check s->err).
 */
int stream_next(WindowStream* s, double* out);
/**
 * @brief Computes the exact number of sliding windows extractable from a
 * fixed-length array.
 *
 * Evaluates the closed-form window count over `n_src / d` multivariate frames
 * given the window length and stride without allocating or scanning memory.
 *
 * @param n_src Total number of scalar elements in the source buffer.
 * @param d Number of channels per frame.
 * @param m_time Number of time frames per window.
 * @param stride Step size in frames between consecutive windows.
 * @return Total number of complete windows, or 0 if total frames are fewer than
 * m_time.
 */
long stream_window_count(long n_src, int d, int m_time, int stride);

#endif
