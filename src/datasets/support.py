import numpy as np
from scipy.spatial import ConvexHull

def fibonacci_sphere(M, seed=0):
    ga = np.pi*(3-np.sqrt(5))
    z = np.linspace(1-1.0/M, -1+1.0/M, M)
    r = np.sqrt(1-z*z)
    theta = np.arange(M)*ga
    x,y = r*np.cos(theta), r*np.sin(theta)
    dirs = np.vstack([x,y,z]).T
    dirs /= np.linalg.norm(dirs,axis=1,keepdims=True)
    return dirs

def support_polytope(V,U):
    return np.max(V @ U.T,axis=0)

def convex_hull_vertices(pts):
    hull = ConvexHull(pts)
    return pts[hull.vertices]
