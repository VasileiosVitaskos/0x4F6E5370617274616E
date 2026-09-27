#ifndef STREAM_H
#define STREAM_H

typedef enum { STREAM_SRC_ARRAY, STREAM_SRC_FILE } StreamSourceKind;

typedef struct {
  int d;        // πόσα κανάλια
  int m_time;   // μήκος παραθύρου σε καρέ
  int stride;   // βήμα σε καρέ
  double *ring; // m_time · d αριθμοί
  int cur; // πού γράφεται το επόμενο καρέ, στο [0, m_time)
  int primed; // έχει σχηματιστεί το πρώτο παράθυρο;
  StreamSourceKind src_kind;
  const double *src; // η πηγή (προς το παρόν πίνακας)
  long n_src; // πόσοι αριθμοί συνολικά στην πηγή
  long pos; // δείκτης του επόμενου αριθμού που θα διαβαστεί
  long n_frames; // πόσα καρέ καταναλώθηκαν
} WindowStream;

int stream_init_array(WindowStream *s, const double *src, long n_src, int d,
                      int m_time, int stride);
void stream_free(WindowStream *s);
int stream_next(WindowStream *s, double *out);
long stream_window_count(long n_src, int d, int m_time, int stride);

#endif
