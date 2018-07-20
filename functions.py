import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage.interpolation import rotate

def ifftnc(arr):
    scaling = np.sqrt(arr.size)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(arr))) * scaling
    
def fftnc(arr):
    scaling = np.sqrt(arr.size)
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr))) / scaling

def interp3(xx, yy, zz, V, points, shape):
    interp = rgi((xx, yy, zz), V, bounds_error = False, fill_value = 0)
    values = interp(points)
    values = values.reshape((int(shape[1]),int(shape[0]),int(shape[2])))
    values = rotate(values, -90)
    values = np.fliplr(values)
    return values

def imscale(image):
    max_v = np.max(image)
    min_v = np.min(image)
    return (image - min_v) / (max_v - min_v)

def ball_3d(radius, voxel_size):
    xx, yy, zz = np.meshgrid(np.arange(-radius, radius + 1),
                             np.arange(-radius, radius + 1),
                             np.arange(-radius, radius + 1))
    xx = xx*voxel_size[1]
    yy = yy*voxel_size[0]
    zz = zz*voxel_size[2]
    sq_dist = np.square(xx) + np.square(yy) + np.square(zz)
    index = np.logical_and(sq_dist <= (radius**2 + radius / 3), sq_dist >= 0.1)
    return index

def total_field(shape, radius, voxel_size):
    kernel_3d = ball_3d(radius, voxel_size)
    kernel_3d = kernel_3d / np.sum(kernel_3d)
    f0 = np.zeros(shape)
    f0[f0.shape[0]//2 - radius : f0.shape[0]//2 + radius + 1,
       f0.shape[1]//2 - radius : f0.shape[1]//2 + radius + 1,
       f0.shape[2]//2 - radius : f0.shape[2]//2 + radius + 1] = kernel_3d
    f0 = fftnc(f0 * np.sqrt(f0.size))
    return f0

def smv_filter(phase, radius, voxel_size):
    f0 = total_field(phase.shape, radius, voxel_size)
    out_phase = ifftnc(f0 * fftnc(phase))
    return np.real(out_phase)

def pad(arr, pad_size):
    pad_size = tuple(pad_size)
    while len(pad_size) < len(arr.shape):
        pad_size = pad_size + (0,)
    pad_size = tuple([(size, size) for size in pad_size])
    return np.pad(arr, pad_size, 'constant')

def left_pad(arr, axis):
    pad_size = tuple([(int(i == axis), 0) for i, a in enumerate(arr.shape)])
    return np.pad(arr, pad_size, 'constant')

def right_pad(arr, axis):
    pad_size = tuple([(0, int(i == axis)) for i, a in enumerate(arr.shape)])
    return np.pad(arr, pad_size, 'constant')

def unpad(arr, pad_size):
    pad_size = tuple(pad_size)
    while len(pad_size) < len(arr.shape):
        pad_size = pad_size + (0,)
    pad_size = tuple([(size, size) for size in pad_size])
    slices = []
    for p in pad_size:
        if p[1] == 0:
            slices.append(slice(p[0], None))
        else:
            slices.append(slice(p[0], -p[1]))
    return arr[slices]

def bbox_slice(bbox):
    front, back = bbox[:len(bbox)//2], bbox[len(bbox)//2:]
    slices = []
    for i in range(len(front)):
        slices.append(slice(front[i], back[i]))
    return slices