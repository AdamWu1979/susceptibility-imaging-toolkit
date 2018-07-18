import numpy as np
from functions import ifftnc, fftnc, total_field, bbox_slice, pad, interp3, unpad, left_pad
from skimage.measure import regionprops

def remove_background(phase, mask, radius):
    f0 = total_field(phase.shape, radius, [1,1,1]) # include voxelsize?
    local_phase = phase - ifftnc(f0 * fftnc(phase))
    valid_points = ifftnc(f0 * fftnc(mask))
    return np.real(local_phase), np.real(valid_points)

def v_sharp(phase, mask, voxel_size, pad_size = (12,12,12), smv_r = 12):
    """3D background phase removal for 3D GRE data.
    Schweser et al. Quantitative imaging of intrinsic magnetic tissue 
    properties using MRI signal phase... NeuroImage. 2010
    
    phase: unwrapped phase (sum of all echoes)
    smv_r: radius of spherical mean value filtering
    
    Steven Cao, Hongjiang Wei, Chunlei Liu
    University of California, Berkeley
    
    TODO: improve variable names
    """
    if len(phase.shape) == 4:
        phase = np.sum(phase, axis = 3)
    phase_wo_deconv = np.zeros(phase.shape)
    final_mask = np.zeros(mask.shape)
    bbox = regionprops(mask.astype(int), cache = False)[0].bbox
    phase, mask = phase[bbox_slice(bbox)], mask[bbox_slice(bbox)]
    phase, mask = pad(phase, pad_size), pad(mask, pad_size)
    
    fov = np.multiply(voxel_size, phase.shape).astype(int)
    xx = np.arange(0, phase.shape[0]) * voxel_size[0]
    yy = np.arange(0, phase.shape[1]) * voxel_size[1]
    zz = np.arange(0, phase.shape[2]) * voxel_size[2]
    xf = np.arange(0, fov[0])
    yf = np.arange(0, fov[1])
    zf = np.arange(0, fov[2])
    
    points = np.meshgrid(xf, yf, zf)
    points = np.vstack(map(np.ravel, points)).T
    phase_upsampled = interp3(xx, yy, zz, phase, points, fov)
    mask_upsampled = interp3(xx, yy, zz, mask, points, fov)
    mask_upsampled = mask_upsampled > 0.5
    
    shape_old = mask_upsampled.shape
    for i in range(len(shape_old)):
        if shape_old[i] % 2 == 1:
            phase_upsampled = left_pad(phase_upsampled, i)
            mask_upsampled = left_pad(mask_upsampled, i)
    
    phi_filtered = np.zeros(phase_upsampled.shape)
    print('Iterating radius from 1 mm to', smv_r, 'mm from center...')
    for i in range(1, smv_r + 1):
        local_phase, valid_points = None, None
        local_phase, valid_points = remove_background(phase_upsampled,
                                                      mask_upsampled, i)
        index = np.abs(valid_points - 1) < 1e-6
        phi_filtered[index] = local_phase[index]
        if i == 2:
            final_mask_p = index
        print(i, end = ' ')
    print()
    
    for i in range(len(shape_old)):
        if shape_old[i] % 2 == 1:
            final_mask_p = np.delete(final_mask_p, 0, axis = i)
            phi_filtered = np.delete(phi_filtered, 0, axis = i)
            phase_upsampled = np.delete(phase_upsampled, 0, axis = i)
    
    points2 = np.meshgrid(xx, yy, zz)
    points2 = np.vstack(map(np.ravel, points2)).T
    phi_filtered *= final_mask_p
    resid_phase = interp3(xf, yf, zf, phase_upsampled - phi_filtered, points2,
                          phase.shape)
    final_mask_p = interp3(xf, yf, zf, final_mask_p, points2, phase.shape)
    final_mask_p = final_mask_p > 0.5
    
    v_sharp_phase = phase - resid_phase
    v_sharp_phase = unpad(v_sharp_phase, pad_size)
    phase_wo_deconv[bbox_slice(bbox)] = v_sharp_phase
    final_mask_p = unpad(final_mask_p, pad_size)
    final_mask[bbox_slice(bbox)] = final_mask_p
    phase_wo_deconv *= final_mask
    
    return np.real(phase_wo_deconv), final_mask