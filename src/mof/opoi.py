import numpy as np
def op_oi(y,phi):
    ny = np.linalg.norm(y)
    if ny==0: return 0,0,0
    op = np.linalg.norm(phi)/ny
    oi = np.linalg.norm(y-phi)/ny
    return op, oi, op**2+oi**2-1
