import time, numpy as np
from src.datasets.synthetic_shapes import cube_vertices
from src.mof.support import fibonacci_sphere, support_polytope
from src.mof.projector_dict import build_dictionary, project_onto_dictionary

def run():
    for M,d in [(1000,4),(4000,8)]:
        U=fibonacci_sphere(M)
        V_ref=cube_vertices()
        rots=np.array([np.eye(3) for _ in range(d)])
        H=build_dictionary(U,V_ref,rots)
        y=support_polytope(V_ref,U)
        t0=time.time(); phi,_=project_onto_dictionary(y,H); dt=time.time()-t0
        print(f"M={M}, d={d}, time={dt:.4f}s")

if __name__=='__main__':
    run()
