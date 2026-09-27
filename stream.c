long stream_window_count(long n_src, int d, int m_time, int stride) {
  long n_frames = n_src / d;
  if (n_frames < m_time)
    return 0;
  long windows = 1 + ((n_frames - m_time) / stride);
  return windows;
}
