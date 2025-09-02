import numpy as np
def fourier_descriptor(contour,K=10):
    z = contour[:,0] + 1j*contour[:,1]
    F = np.fft.fft(z)
    return np.abs(F[:K])/np.abs(F[0])
