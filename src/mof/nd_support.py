import numpy as np

def sample_sphere(n, M, rng=None):
    """Uniform random directions on S^{n-1} by normalizing Gaussians."""
    rng = np.random.default_rng(rng)
    U = rng.normal(size=(M, n))
    return U / np.linalg.norm(U, axis=1, keepdims=True)

def haar_orthogonal(n, rng=None):
    """Random orthogonal matrix (Haar) via QR."""
    rng = np.random.default_rng(rng)
    A = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0: Q[:,0] *= -1
    return Q

# Closed-form supports

def support_cube(U, a=1.0):
    """Hypercube (ℓ∞ ball dual): h(u) = a * ||u||_1."""
    return a * np.sum(np.abs(U), axis=1)

def support_crosspolytope(U, r=1.0):
    """Cross-polytope (ℓ1 ball dual): h(u) = r * ||u||_∞."""
    return r * np.max(np.abs(U), axis=1)

def support_ellipsoid(U, axes):
    """Ellipsoid with axis lengths = axes (len n)."""
    # h(u) = sqrt(u^T diag(axes^2) u)
    A = np.diag(np.square(axes))
    vals = np.einsum('ij,jk,ik->i', U, A, U)
    return np.sqrt(vals)

def smooth_support(h, rho=0.1):
    """
    Minkowski smoothing: K ⊕ ρB.
    In support-function space this is just h(u) -> h(u) + rho.
    """
    return h + rho

def jitter_support(h, sigma=0.05, rng=None):
    """
    Add Gaussian jitter to support values to mimic spiky perturbations.
    Does NOT break convexity if sigma is small relative to h.
    """
    rng = np.random.default_rng(rng)
    return h + rng.normal(scale=sigma, size=h.shape)

