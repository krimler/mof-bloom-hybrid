"""
Robustness experiments for MOF descriptors on synthetic shapes.
- Minkowski smoothing (K ⊕ ρB): simulate by adding constant rho to support values.
- Jittered vertices: perturb cube vertices.
- Partial directions: randomly drop directions.
"""
import numpy as np, pandas as pd, os
from src.datasets.synthetic_shapes import cube_vertices, jitter_vertices
from src.mof.support import fibonacci_sphere, support_polytope
from src.mof.projector_dict import build_dictionary, project_onto_dictionary
from src.mof.opoi import op_oi

def run():
    M=1000; d=4
    U=fibonacci_sphere(M)
    V_ref=cube_vertices()
    rots=np.array([np.eye(3) for _ in range(d)])
    H=build_dictionary(U,V_ref,rots)
    # Base cube
    V=cube_vertices()
    y0=support_polytope(V,U)
    # 1) Minkowski smoothing
    rhos=np.linspace(0,0.5,6)
    rows=[]
    for rho in rhos:
        y=y0+rho
        phi,_=project_onto_dictionary(y,H)
        op,oi,res=op_oi(y,phi)
        rows.append({'type':'smoothing','rho':rho,'OP':op,'OI':oi})
    # 2) Jitter
    sigmas=[0,0.05,0.1,0.2]
    for sigma in sigmas:
        Vj=jitter_vertices(V,sigma)
        y=support_polytope(Vj,U)
        phi,_=project_onto_dictionary(y,H)
        op,oi,res=op_oi(y,phi)
        rows.append({'type':'jitter','sigma':sigma,'OP':op,'OI':oi})
    # 3) Partial directions
    drops=[0,0.25,0.5,0.75]
    for frac in drops:
        keep=int(M*(1-frac))
        idx=np.random.choice(M,keep,replace=False)
        y=y0[idx]; Hsub=H[idx,:]
        phi,_=project_onto_dictionary(y,Hsub)
        op,oi,res=op_oi(y,phi)
        rows.append({'type':'partial','frac_drop':frac,'OP':op,'OI':oi})
    df=pd.DataFrame(rows)
    os.makedirs('tables',exist_ok=True)
    df.to_csv('tables/robustness.csv',index=False)
    print(df)

if __name__=='__main__':
    run()
