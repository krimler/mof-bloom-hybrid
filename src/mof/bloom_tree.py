# src/mof/bloom_tree.py
import numpy as np
from .refs2d import eval_ref

def pac_radius(B, kappa, d, m, delta):
    """
    Simple PAC-style radius for OP estimation with LS projection:
    c * B * kappa * sqrt((d + log(1/delta)) / m).
    We use a conservative constant c=2.
    """
    c = 2.0
    m = max(int(m), 1)
    return c * B * kappa * np.sqrt((d + np.log(1.0/max(delta,1e-12))) / m)

class Node:
    """
    A Bloom-style routing node.
    refs: list of reference dicts at this node
    children: mapping from ref index -> child Node
    tau: early-reject threshold on OP (calibrated elsewhere)
    name: debug/trace
    """
    def __init__(self, refs, children=None, tau=0.15, name=""):
        self.refs = refs
        self.children = children or {}
        self.tau = tau
        self.name = name
        # Diagnostics for PAC radius; if you have measured values, set per-node.
        self.B = 1.0
        self.kappa = 5.0
        self.d = 8  # dictionary size per ref (conceptual)

    def _per_ref_op(self, hK_m, U_m):
        """
        Approximate per-ref OP by projecting onto each single reference column separately.
        Returns ops: (k,) with OP estimates per ref (k=len(refs)).
        """
        cols = [eval_ref(U_m, R) for R in self.refs]  # list of (m,)
        ops = []
        denom = np.linalg.norm(hK_m) + 1e-12
        for col in cols:
            H = col.reshape(-1,1)
            c, *_ = np.linalg.lstsq(H, hK_m, rcond=None)
            hproj = H @ c
            ops.append(np.linalg.norm(hproj)/denom)
        return np.array(ops)

    def score(self, hK, U, m, delta=0.01):
        """Return per-ref OP estimates and a common PAC radius."""
        ops = self._per_ref_op(hK[:m], U[:m])
        rad = pac_radius(self.B, self.kappa, self.d, m, delta/len(self.refs))
        return ops, rad

    def route(self, hK, U, m0=1024, delta=0.01, max_m=8192):
        """
        CI-gated routing: increase m until top-1 OP margin exceeds 2*rad,
        otherwise early-reject or return ambiguous after max_m.
        Returns (leaf_node, best_ref_index_or_None, path_info).
        """
        t = self
        m = m0
        path = []
        while True:
            ops, rad = t.score(hK, U, m, delta)
            order = np.argsort(ops)
            a = order[-1]
            b = order[-2] if len(order) >= 2 else a
            path.append((t.name, a, ops[a], rad, m))
            # confident winner?
            if ops[a] - ops[b] > 2*rad:
                # route to child or stop at leaf
                if a in t.children:
                    t = t.children[a]
                    m = max(m0, m//2)  # reset budget after descending
                    continue
                else:
                    return t, a, path
            # early reject?
            if ops[a] + rad < t.tau:
                return t, None, path
            # need more directions
            if m >= max_m:
                return t, a, path  # ambiguous but stop
            m *= 2

