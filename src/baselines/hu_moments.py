# src/baselines/hu_moments.py
import numpy as np
from skimage.measure import moments_hu, moments_central, moments

def hu_descriptor(binary_image):
    """
    Compute 7 Hu invariant moments from a binary silhouette.
    Returns a 7-dim float vector (log-scaled, signed convention).
    """
    # normalize to {0,1}
    img = (binary_image > 0).astype(np.float64)
    if img.sum() == 0:
        return np.zeros(7, dtype=np.float64)
    m = moments(img, order=3)
    centroid = (m[1,0]/m[0,0], m[0,1]/m[0,0])
    mu = moments_central(img, centroid[0], centroid[1], order=3)
    hu = moments_hu(mu)
    # common practice: signed log transform to compress dynamic range
    # log(|hu|) with sign to preserve invariance
    out = np.zeros_like(hu)
    for i, v in enumerate(hu):
        s = 0.0 if v == 0 else np.sign(v)
        out[i] = s * np.log10(abs(v) + 1e-30)
    return out

