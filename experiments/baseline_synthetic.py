import numpy as np, pandas as pd, os
from src.datasets.synthetic_shapes import cube_vertices, icosahedron_vertices, random_points, anisotropic_points, jitter_vertices
from src.mof.support import fibonacci_sphere, support_polytope, convex_hull_vertices
from src.mof.projector_dict import build_dictionary, project_onto_dictionary
from src.mof.opoi import op_oi

def run():
    M=1000; d=4
    U=fibonacci_sphere(M)
    V_ref=cube_vertices()
    rots=np.array([np.eye(3) for _ in range(d)])
    H=build_dictionary(U,V_ref,rots)
    results=[]
    families={
        'cube': cube_vertices(),
        'ico': icosahedron_vertices(),
        'random': convex_hull_vertices(random_points(32)),
        'aniso': convex_hull_vertices(anisotropic_points(32)),
        'jittered': jitter_vertices(cube_vertices(),0.1)
    }
    for name,V in families.items():
        y=support_polytope(V,U)
        phi,_=project_onto_dictionary(y,H)
        op,oi,res=op_oi(y,phi)
        results.append({'shape':name,'OP':op,'OI':oi,'residual':res})
    df=pd.DataFrame(results)
    os.makedirs('tables',exist_ok=True)
    df.to_csv('tables/baseline_synthetic.csv',index=False)
    print(df)

if __name__=='__main__':
    run()
