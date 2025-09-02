"""
Retrieval and clustering experiments for synthetic families.
Each family: cube, ico, random, anisotropic, jittered.
Generate multiple instances per class, compute descriptors, then:
- Retrieval: cosine similarity top-1 accuracy.
- Clustering: k-means, compute purity.
"""
import numpy as np, pandas as pd, os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from src.datasets.synthetic_shapes import cube_vertices, icosahedron_vertices, random_points, anisotropic_points, jitter_vertices
from src.mof.support import fibonacci_sphere, support_polytope, convex_hull_vertices
from src.mof.projector_dict import build_dictionary, project_onto_dictionary
from src.mof.opoi import op_oi

def generate_instance(family,seed):
    if family=='cube': return cube_vertices()
    if family=='ico': return icosahedron_vertices()
    if family=='random': return convex_hull_vertices(random_points(32,seed))
    if family=='aniso': return convex_hull_vertices(anisotropic_points(32,seed=seed))
    if family=='jittered': return jitter_vertices(cube_vertices(),0.1,seed=seed)
    return None

def run():
    M=500; d=4
    U=fibonacci_sphere(M)
    V_ref=cube_vertices()
    rots=np.array([np.eye(3) for _ in range(d)])
    H=build_dictionary(U,V_ref,rots)
    families=['cube','ico','random','aniso','jittered']
    data=[]; labels=[]
    for i,fam in enumerate(families):
        for s in range(10):
            V=generate_instance(fam,s)
            y=support_polytope(V,U)
            phi,_=project_onto_dictionary(y,H)
            op,oi,res=op_oi(y,phi)
            data.append([op,oi]); labels.append(i)
    X=np.array(data); ytrue=np.array(labels)
    # Retrieval top-1
    from sklearn.metrics.pairwise import cosine_similarity
    sim=cosine_similarity(X)
    correct=0; total=0
    for i in range(len(X)):
        sim[i,i]=-1 # exclude self
        j=np.argmax(sim[i])
        if ytrue[j]==ytrue[i]: correct+=1
        total+=1
    retrieval_acc=correct/total
    # Clustering purity
    km=KMeans(n_clusters=len(families),n_init=10,random_state=0).fit(X)
    ypred=km.labels_
    purity=sum([np.max(np.bincount(ytrue[ypred==c])) for c in np.unique(ypred)])/len(ytrue)
    results={'retrieval_top1':retrieval_acc,'clustering_purity':purity}
    os.makedirs('tables',exist_ok=True)
    pd.DataFrame([results]).to_csv('tables/retrieval.csv',index=False)
    print(results)

if __name__=='__main__':
    run()
