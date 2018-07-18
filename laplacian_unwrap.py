import numpy as np

from functions import ifftnc, fftnc, pad, left_pad, unpad

def laplacian_unwrap(phase, voxel_size, pad_size = (12,12,12)):
    """
    Implements Laplacian-based unwrapping 
    (http://onlinelibrary.wiley.com/doi/10.1002/nbm.3056/full)
    
    Steven Cao, Hongjiang Wei, Wei Li, Chunlei Liu
    University of California, Berkeley
    """
    # add 4th singleton dimension if phase is 3d
    if len(phase.shape) == 3:
        phase = phase[:,:,:,None]
    assert len(phase.shape) == 4, 'Phase must be 3d or 4d'
    phase = pad(phase, pad_size)
    
    if phase.shape[2] % 2 == 1:
        phase = left_pad(phase, 2)
    
    yy, xx, zz = np.meshgrid(np.arange(1, phase.shape[1]+1),
                             np.arange(1, phase.shape[0]+1),
                             np.arange(1, phase.shape[2]+1))
    field_of_view = np.multiply(voxel_size, phase.shape[0:3])
    xx, yy, zz = ((xx - phase.shape[0]/2 - 1) / field_of_view[0],
                  (yy - phase.shape[1]/2 - 1) / field_of_view[1],
                  (zz - phase.shape[2]/2 - 1) / field_of_view[2])
    k2 = np.square(xx) + np.square(yy) + np.square(zz)
    xx, yy, zz, field_of_view = None, None, None, None
    
    laplacian = np.zeros(phase.shape, dtype = np.complex)
    phi = np.zeros(phase.shape, dtype = np.complex)
    for i in range(0, phase.shape[3]):
        laplacian[:,:,:,i] = (np.cos(phase[:,:,:,i]) * 
                 ifftnc(k2 * fftnc(np.sin(phase[:,:,:,i]))) - 
                 np.sin(phase[:,:,:,i]) * 
                 ifftnc(k2 * fftnc(np.cos(phase[:,:,:,i]))))
        phi[:,:,:,i] = fftnc(laplacian[:,:,:,i]) / k2
        phi[phase.shape[0]//2,phase.shape[1]//2,phase.shape[2]//2,i] = 0
        phi[:,:,:,i] = ifftnc(phi[:,:,:,i])
    laplacian = unpad(laplacian, pad_size)
    phi = unpad(phi, pad_size)
    phi = np.real(phi)
    return phi, laplacian