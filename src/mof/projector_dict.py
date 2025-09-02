import numpy as np

def build_dictionary(U,R_vertices,rotations):
    M,d = U.shape[0], rotations.shape[0]
    H = np.zeros((M,d))
    for j,R in enumerate(rotations):
        V_rot = (R @ R_vertices.T).T
        H[:,j] = np.max(V_rot @ U.T,axis=0)
    return H

def project_onto_dictionary(y,H,ridge=0.0):
    if ridge>0:
        A = H.T@H + ridge*np.eye(H.shape[1])
        c = np.linalg.solve(A,H.T@y)
    else:
        c, *_ = np.linalg.lstsq(H,y,rcond=None)
    return H@c,c
