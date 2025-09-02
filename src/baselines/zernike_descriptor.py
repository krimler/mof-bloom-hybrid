import mahotas
def zernike_descriptor(image,radius=21,degree=8):
    return mahotas.features.zernike_moments(image,radius,degree)
