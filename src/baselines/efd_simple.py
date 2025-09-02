# src/baselines/efd_simple.py
import numpy as np

def resample_contour(contour, n=256):
    """
    Resample a (N,2) contour to n points uniformly along its arc length.
    """
    P = np.asarray(contour, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        return None
    # cumulative arc length
    d = np.sqrt(np.sum(np.diff(P, axis=0)**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return None
    t = np.linspace(0.0, s[-1], num=n, endpoint=False)
    Q = np.empty((n,2), dtype=np.float64)
    for i, ti in enumerate(t):
        j = np.searchsorted(s, ti, side="right") - 1
        j = np.clip(j, 0, len(P)-2)
        alpha = (ti - s[j]) / max(s[j+1]-s[j], 1e-12)
        Q[i] = (1-alpha)*P[j] + alpha*P[j+1]
    return Q

def efd_simple(contour, K=20):
    """
    Simple elliptic Fourier-like descriptor:
    - Resample contour to fixed n
    - Convert to complex signal z = x + i y (centered)
    - Take DFT, drop DC, keep first K magnitudes (scale-invariant)
    Returns K-dim vector.
    """
    Q = resample_contour(contour, n=256)
    if Q is None:
        return np.zeros(K, dtype=np.float64)
    z = Q[:,0] + 1j*Q[:,1]
    z = z - z.mean()              # translation invariance
    norm = np.linalg.norm(z) + 1e-12
    z = z / norm                  # scale invariance
    F = np.fft.rfft(z)            # half-spectrum
    mag = np.abs(F)[1:K+1]        # drop DC term
    if mag.size < K:
        mag = np.pad(mag, (0, K - mag.size), mode='constant')
    return mag.astype(np.float64)

