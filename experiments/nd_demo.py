import os, time
import numpy as np, pandas as pd

from src.mof.nd_support import (
    sample_sphere,
    support_cube, support_crosspolytope, support_ellipsoid,
    smooth_support, jitter_support
)
from src.mof.projector_dict_nd import build_orbit_dictionary
from src.mof.opoi import op_oi  # assumes you already have op_oi(y,phi)


def run_nd_demo():
    results = []
    rng = np.random.default_rng(0)
    dims = [2, 3, 4, 6, 8, 10, 12]

    for n in dims:
        print(f"Running {n}D...")
        M = 5000  # number of sampled directions
        U = sample_sphere(n, M, rng)

        # Build dictionary for cube reference (example, d=8)
        H_cube = build_orbit_dictionary(U, "cube", {"a": 1.0}, d=8, rng=rng)

        # Shape families
        shapes = {
            "cube": support_cube(U, a=1.0),
            "xpoly": support_crosspolytope(U, r=1.0),
            "ellip_3to1": support_ellipsoid(U, axes=[3.0] + [1.0] * (n - 1)),
            "ellip_2to1": support_ellipsoid(U, axes=[2.0] + [1.0] * (n - 1)),
        }

        for name, hK in shapes.items():
            phi, *_ = np.linalg.lstsq(H_cube, hK, rcond=None)
            h_proj = H_cube @ phi
            op, oi, res = op_oi(hK, h_proj)
            results.append({"n": n, "shape": name, "OP": op, "OI": oi, "residual": res})

        # Robustness experiments: base = 3:1 ellipsoid
        base = support_ellipsoid(U, axes=[3.0] + [1.0] * (n - 1))

        # Smoothing
        for rho in [0.0, 0.1, 0.2, 0.3, 0.5]:
            hK = smooth_support(base, rho)
            phi, *_ = np.linalg.lstsq(H_cube, hK, rcond=None)
            h_proj = H_cube @ phi
            op, oi, res = op_oi(hK, h_proj)
            results.append(
                {"n": n, "shape": f"ellip3to1+smooth{rho}", "OP": op, "OI": oi, "residual": res}
            )

        # Jitter
        for sigma in [0.0, 0.05, 0.1, 0.2]:
            hK = jitter_support(base, sigma, rng=rng)
            phi, *_ = np.linalg.lstsq(H_cube, hK, rcond=None)
            h_proj = H_cube @ phi
            op, oi, res = op_oi(hK, h_proj)
            results.append(
                {"n": n, "shape": f"ellip3to1+jitter{sigma}", "OP": op, "OI": oi, "residual": res}
            )

        # Efficiency test
        hK = support_ellipsoid(U, axes=[3.0] + [1.0] * (n - 1))
        t0 = time.time()
        phi, *_ = np.linalg.lstsq(H_cube, hK, rcond=None)
        dt = time.time() - t0
        results.append({"n": n, "shape": "ellip3to1_time", "OP": 0, "OI": dt, "residual": 0})

    df = pd.DataFrame(results)
    os.makedirs("tables", exist_ok=True)
    df.to_csv("tables/nd_demo.csv", index=False)
    print("Saved results to tables/nd_demo.csv")


if __name__ == "__main__":
    run_nd_demo()

