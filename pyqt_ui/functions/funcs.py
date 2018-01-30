import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage.interpolation import rotate
from scipy.fftpack import dct, idct

def ifftnc(arr):
    arr = np.array(arr)
    scaling = np.sqrt(arr.size)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(arr))) * scaling
    
def fftnc(arr):
    arr = np.array(arr)
    scaling = np.sqrt(arr.size)
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr))) / scaling

def dct3(arr):
    return dct(dct(dct(arr,
                       axis = 0, norm = 'ortho'),
                   axis = 1, norm = 'ortho'),
               axis = 2, norm = 'ortho')

def idct3(arr):
    return idct(idct(idct(arr,
                          axis = 2, norm = 'ortho'),
                     axis = 1, norm = 'ortho'),
                axis = 0, norm = 'ortho')

##def interp3(xx, yy, zz, V, xxi, yyi, zzi):
##    interp = rgi((xx, yy, zz), V)
##    xxi, yyi, zzi = xxi.flatten(), yyi.flatten(), zzi.flatten()
##    points = [[xxi[i], yyi[i], zzi[i]] for i in range(len(xxi))]
##    return np.nan_to_num(interp(points))

def interp3(xx, yy, zz, V, points, shape):
    interp = rgi((xx, yy, zz), V, bounds_error = False, fill_value = 0)
    values = interp(points)
    #shape = points[-1]+1
    values = values.reshape((shape[1],shape[0],shape[2]))
    values = rotate(values, -90)
    values = np.fliplr(values)
    #grid_points = np.meshgrid(xx,yy,zz)
    #grid_points = np.vstack(map(np.ravel, grid_points)).T
    #values = interpn(tuple([xx,yy,zz]), V, points)
    return values

def append_zeros(arr, axis, num_zeros = 1):
    arr = np.array(arr)
    pad_size = [(0,0) for i in range(len(arr.shape))]
    pad_size[axis] = (0, num_zeros)
    pad_size = tuple(pad_size)
    return np.pad(arr, pad_size, 'constant')

def calc_d2_matrix(shape, spatial_resolution, B0_dir):
    shape, spatial_resolution, B0_dir = np.array(shape), np.array(spatial_resolution), np.array(B0_dir)
    field_of_view = shape * spatial_resolution
    ry, rx, rz = np.meshgrid(np.arange(-shape[1]//2, shape[1]//2),
                             np.arange(-shape[0]//2, shape[0]//2),
                             np.arange(-shape[2]//2, shape[2]//2))
    rx, ry, rz = rx/field_of_view[0], ry/field_of_view[1], rz/field_of_view[2]
    sq_dist = rx**2 + ry**2 + rz**2
    sq_dist[sq_dist==0] = 1e-6
    d2 = ((B0_dir[0]*rx + B0_dir[1]*ry + B0_dir[2]*rz)**2)/sq_dist
    d2 = (1/3 - d2)
    #d2[np.array(d2.shape)//2] = 0
    return d2
