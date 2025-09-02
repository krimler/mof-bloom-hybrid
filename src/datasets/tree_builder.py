# src/mof/tree_builder.py
import numpy as np
from .refs2d import analytic_reference_library, eval_ref, sample_circle
from .learn_medoids import k_medoids_supports, medoids_to_refs
from .bloom_tree import Node

def build_analytic_tree():
    lib = analytic_reference_library()
    root = Node(lib["root"], name="root", tau=0.15)
    for idx, child_refs in enumerate(lib["children"]):
        root.children[idx] = Node(child_refs, name=f"child{idx}", tau=0.15)
    return root

def build_learned_tree_from_supports(H_list, U, k_root=6, k_child=4):
    """
    Build a 2-level tree mixing analytic references and learned medoids:
    - root: analytic + k_root medoids from all supports
    - each child: analytic child refs + k_child medoids learned from supports "closest" to that child
    """
    lib = analytic_reference_library()
    M = U.shape[0]
    # root node
    root_refs = list(lib["root"])  # copy
    # learn global medoids
    med_root, _ = k_medoids_supports(H_list, k=k_root, max_iter=20, rng=0)
    root_refs.extend(medoids_to_refs(H_list, med_root))
    root = Node(root_refs, name="root", tau=0.15)

    # assign supports to best root analytic ref (not medoids) for child learning
    # (coarse assignment: nearest analytic root reference)
    root_analytic = lib["root"]
    assign = [[] for _ in root_analytic]
    for idx_h, h in enumerate(H_list):
        # score each analytic ref by L2 between h and eval_ref(U, ref)
        def score_ref(ref):
            col = eval_ref(U, ref)
            return np.linalg.norm(h - col)
        j_best = int(np.argmin([score_ref(r) for r in root_analytic]))
        assign[j_best].append(idx_h)

    # build children
    for child_idx, child_analytic in enumerate(lib["children"]):
        # collect supports routed to the corresponding analytic root ref
        if child_idx < len(assign) and len(assign[child_idx]) > 0:
            sub = [H_list[j] for j in assign[child_idx]]
            med_child, _ = k_medoids_supports(sub, k=min(k_child, len(sub)), max_iter=20, rng=child_idx+1)
            # med_child indices refer to 'sub', map back to original H_list indices
            # but for refs we only need the support vectors
            child_refs = list(child_analytic)
            child_refs.extend(medoids_to_refs(sub, med_child))
        else:
            # no data assigned; just analytic
            child_refs = list(child_analytic)
        root.children[child_idx] = Node(child_refs, name=f"child{child_idx}", tau=0.15)
    return root

def build_learned_tree_from_dataset(loader_func, root_dir, M=1024, k_root=6, k_child=4):
    """
    Convenience: load dataset, compute supports on a deterministic grid U, learn medoids, build tree.
    loader_func should yield (label, image) pairs.
    """
    # Deterministic directions for all shapes
    U, _ = sample_circle(M, deterministic=True)
    H_list = []
    labels = []
    for lbl, img in loader_func(root_dir):
        # largest contour
        from skimage import measure
        contours = measure.find_contours(img, 0.5)
        if not contours:
            continue
        contour = max(contours, key=lambda x: x.shape[0])
        h = (U @ contour.T).max(axis=1)
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        H_list.append(h)
        labels.append(lbl)
    return build_learned_tree_from_supports(H_list, U, k_root=k_root, k_child=k_child)

