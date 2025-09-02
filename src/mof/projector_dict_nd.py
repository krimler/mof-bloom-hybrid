from .nd_support import haar_orthogonal, support_cube, support_crosspolytope, support_ellipsoid
import numpy as np

def build_orbit_dictionary(U, ref_kind, params, d=8, rng=None):
    """
    Build orbit dictionary columns for reference shape.
    U: (M,n) directions
    ref_kind: 'cube', 'xpoly', 'ellip'
    params: dict of reference params
    d: dictionary size (# rotations/axes)
    """
    rng = np.random.default_rng(rng)
    M, n = U.shape
    cols = []
    for _ in range(d):
        Q = haar_orthogonal(n, rng)
        Urot = U @ Q
        if ref_kind == 'cube':
            cols.append(support_cube(Urot, a=params.get('a',1.0)))
        elif ref_kind == 'xpoly':
            cols.append(support_crosspolytope(Urot, r=params.get('r',1.0)))
        elif ref_kind == 'ellip':
            axes = params.get('axes', np.ones(n))
            cols.append(support_ellipsoid(Urot, axes))
        else:
            raise ValueError("Unknown ref_kind")
    return np.column_stack(cols)  # (M,d)

