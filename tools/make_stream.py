#!/usr/bin/env python3
"""A CONTINUOUS stationary series, for overlapping windows.

make_datasets.py draws each window independently, so overlapping windows of
it would be meaningless: two consecutive draws share nothing, and a window
straddling them sees two unrelated pieces glued together.

Here the whole series is built at once, in the frequency domain: white noise
shaped so the power sits at a few low frequencies. Every window of it then
has the same k-dimensional structure, and consecutive windows are correlated
the way real data is.

Each frequency contributes a cosine AND a sine of equal power, so the planted
directions come in pairs. That is not a choice, it is what a stationary
series does, and it is why real time series have paired eigenvalues.

    python3 tools/make_stream.py
"""
import json, os, numpy as np

M       = 256
N_SAMP  = 1_024_000
SIGMA   = 0.20
SEED    = 2001
FREQS   = [3, 7, 13, 21]          # cycles per window -> 8 modes with the pairs
WEIGHT  = [1.00, 0.62, 0.38, 0.22]
OUT     = os.path.join("tests", "synthetic", "data")

rng = np.random.default_rng(SEED)
F   = np.fft.rfftfreq(N_SAMP, d=1.0/M)
psd = np.full(len(F), 1e-6)
for f, w in zip(FREQS, WEIGHT):
    psd += w * np.exp(-0.5*((F-f)/0.05)**2)
spec = np.sqrt(psd)*(rng.standard_normal(len(F)) + 1j*rng.standard_normal(len(F)))
x = np.fft.irfft(spec, N_SAMP)
x = x/x.std() + SIGMA*rng.standard_normal(N_SAMP)

os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "stationary_stream.frames")
with open(p, "w") as f:
    f.write("\n".join("%.17g" % v for v in x)); f.write("\n")

# the planted shapes: cos and sin at each frequency, on the window grid
t = np.arange(M)/M
rows = []
for fr in FREQS:
    rows += [np.cos(2*np.pi*fr*t), np.sin(2*np.pi*fr*t)]
W0 = np.array([r/np.linalg.norm(r) for r in rows])
with open(os.path.join(OUT, "stationary_stream.meta.json"), "w") as f:
    json.dump(dict(name="stationary_stream", seed=SEED,
                   generator="tools/make_stream.py",
                   geometry=dict(m_time=M, channels=1, k=8, n_samples=N_SAMP),
                   noise=dict(sigma=SIGMA),
                   shapes=dict(kind="fourier pairs", freqs=FREQS,
                               weights=WEIGHT, directions=W0.tolist()),
                   rotation=None, angle_true_rad=[0.0],
                   angle_true_max_deg=0.0), f, indent=1)
print("wrote %s   %.1f MB   %d samples" % (p, os.path.getsize(p)/1e6, N_SAMP))
print("windows at stride 64 : %d" % (1 + (N_SAMP-M)//64))
