# src/mof/learn_medoids.py
import numpy as np

def fft_align_shift(a, b):
    """
    Find circular shift s (0..M-1) that maximizes correlation <a, roll(b,s)>
    using FFT-based cross-correlation. Returns integer shift s.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    xcorr = np.fft.irfft(fa * np.conj(fb), n=a.size)
    return int(np.argmax(xcorr))

def aligned_l2(a, b):
    """
    Rotation-aligned L2 distance between 1D periodic supports a,b (length M).
    """
    s = fft_align_shift(a, b)
    b_shift = np.roll(b, s)
    return np.linalg.norm(a - b_shift)

def k_medoids_supports(H_list, k, max_iter=20, rng=None):
    """
    Simple k-medoids on rotation-aligned L2 for 1D supports (same length).
    Returns (medoid_indices, clusters dict).
    """
    rng = np.random.default_rng(rng)
    N = len(H_list)
    if N == 0 or k <= 0:
        return [], {}
    k = min(k, N)
    medoids = sorted(set(rng.choice(N, size=k, replace=False).tolist()))
    for it in range(max_iter):
        # assign points to nearest medoid
        clusters = {m: [] for m in medoids}
        for i in range(N):
            m_best = min(medoids, key=lambda m: aligned_l2(H_list[i], H_list[m]))
            clusters[m_best].append(i)
        # update medoids
        new_meds = []
        for m, idxs in clusters.items():
            if not idxs:
                continue
            # Pick j in idxs minimizing sum of aligned L2
            best_j = min(idxs, key=lambda j: sum(aligned_l2(H_list[j], H_list[p]) for p in idxs))
            new_meds.append(best_j)
        new_meds = sorted(set(new_meds))
        if new_meds == medoids:
            break
        medoids = new_meds
    # rebuild clusters one last time
    clusters = {m: [] for m in medoids}
    for i in range(N):
        m_best = min(medoids, key=lambda m: aligned_l2(H_list[i], H_list[m]))
        clusters[m_best].append(i)
    return medoids, clusters

def medoids_to_refs(H_list, medoid_ids):
    """
    Convert medoid support vectors into reference dicts of kind 'medoid'.
    Assumes H_list[j] is (M0,) sampled on theta0=linspace(0,2pi,M0).
    """
    refs = []
    for j in medoid_ids:
        h = np.asarray(H_list[j], dtype=np.float64)
        if h.ndim != 1:
            continue
        refs.append({"kind":"medoid", "h": h})
    return refs

