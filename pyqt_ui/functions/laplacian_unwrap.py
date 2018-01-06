import numpy as np
from functions.funcs import ifftnc, fftnc

def laplacian_unwrap(raw_phase, voxel_size = None, pad_size = None):
    """Does Laplacian-based phase unwrapping
    Inputs:
        raw_phase - raw phase (must be 3d or 4d - x, y, slice number, (echo time))
        voxel_size: resolution of x, y, and slice (length 3, default [1,1,1])
        pad_size: amount to pad phase x, y, and slice (length 3, default [12,12,12])
    Returns:
        phi - unwrapped phase
        laplacian
    Steven Cao
    Wei Li, PhD
    Chunlei Liu, PhD
    University of California, Berkeley
    """
    if voxel_size is None: voxel_size = [1, 1, 1]
    if pad_size is None: pad_size = [12, 12, 12]

    pad_size = pad_size + [0]
    pad_size = tuple([(size, size) for size in pad_size])

    #add 4th singleton dimension if phase is 3d
    if len(raw_phase.shape) == 3:
        raw_phase = raw_phase[:,:,:,None]
    assert len(raw_phase.shape) == 4, 'Phase must be 3d or 4d'

    raw_phase = np.pad(raw_phase, pad_size, 'constant')

    #make sure the 3rd dim is even
    if raw_phase.shape[2] % 2 == 1:
        np.concatenate((raw_phase, np.zeros(tuple([raw_phase.shape[0], raw_phase.shape[1], 1, raw_phase.shape[3]]))),axis=2)

    yy, xx, zz = np.meshgrid(np.arange(1, raw_phase.shape[1]+1),
                             np.arange(1, raw_phase.shape[0]+1),
                             np.arange(1, raw_phase.shape[2]+1))
    field_of_view = np.multiply(voxel_size, raw_phase.shape[0:3])
    xx, yy, zz = ((xx - raw_phase.shape[0]/2 - 1) / field_of_view[0],
                  (yy - raw_phase.shape[1]/2 - 1) / field_of_view[1],
                  (zz - raw_phase.shape[2]/2 - 1) / field_of_view[2])
    k2 = np.square(xx) + np.square(yy) + np.square(zz)

    #reduce memory usage
    xx, yy, zz, field_of_view = None, None, None, None

    laplacian = np.zeros(raw_phase.shape, dtype=np.complex)
    phi = np.zeros(raw_phase.shape, dtype=np.complex)
    for i in range(0, raw_phase.shape[3]):
        laplacian[:,:,:,i] = (np.cos(raw_phase[:,:,:,i]) * ifftnc(k2 * fftnc(np.sin(raw_phase[:,:,:,i]))) -
                              np.sin(raw_phase[:,:,:,i]) * ifftnc(k2 * fftnc(np.cos(raw_phase[:,:,:,i]))))

        phi[:,:,:,i] = fftnc(laplacian[:,:,:,i]) / k2
        phi[raw_phase.shape[0]//2, raw_phase.shape[1]//2, raw_phase.shape[2]//2, i] = 0
        phi[:,:,:,i] = ifftnc(phi[:,:,:,i])

    #remove padding
    laplacian = laplacian[pad_size[0][0]:-pad_size[0][1],pad_size[1][0]:-pad_size[1][1],pad_size[2][0]:-pad_size[2][1],:]
    phi = phi[pad_size[0][0]:-pad_size[0][1],pad_size[1][0]:-pad_size[1][1],pad_size[2][0]:-pad_size[2][1],:]

    #real
    phi = np.real(phi)
        
    return phi, laplacian
