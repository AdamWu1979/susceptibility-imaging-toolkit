import numpy as np
from functions.funcs import ifftnc, fftnc, interp3, append_zeros

def bounding_box_3D(mask):
    dims = len(mask.shape)
    assert dims == 3, 'Mask should be 3D'
    assert np.sum(mask) > 0, 'Mask should have nonzero values'
    
    bbox = np.zeros((dims,2), dtype=np.int)
    bbox[:,0] = mask.shape
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i,j,k]:
                    if i < bbox[0,0]:#leftmost non-zero value
                        bbox[0,0] = i
                    if j < bbox[1,0]:
                        bbox[1,0] = j
                    if k < bbox[2,0]:
                        bbox[2,0] = k
                    if i > bbox[0,1]:#rightmost non-zero value
                        bbox[0,1] = i
                    if j > bbox[1,1]:
                        bbox[1,1] = j
                    if k > bbox[2,1]:
                        bbox[2,1] = k
    bbox[:,1] += 1 #exclusive right bound
    return bbox

def mask_bounding_box(mask, voxel_size):
    """
    Steven Cao, 26 October 2017
    Wei Li, March 3, 2014.
    """
    pad_length = 20
    pad_size = [int(pad_length // size) for size in voxel_size]
    pad_size = tuple([(size, size) for size in pad_size])
    #bbox = regionprops(mask, cache=False)[0].bbox
    bbox = bounding_box_3D(mask)
    #bbox = [[bbox[0], bbox[3]],
    #        [bbox[1], bbox[4]],
    #        [bbox[2], bbox[5]]]
    return np.array(bbox), pad_size

def ball_3d(radius):
    xx, yy, zz = np.meshgrid(np.arange(-radius, radius + 1),
                             np.arange(-radius, radius + 1),
                             np.arange(-radius, radius + 1))
    sq_dist = np.square(xx) + np.square(yy) + np.square(zz)
    index = np.logical_and(sq_dist <= (radius**2 + radius / 3), sq_dist >= 0.1)
    return index

def my_removal(phase, mask, radius):
    kernel_3d = ball_3d(radius)
    kernel_3d = kernel_3d / np.sum(kernel_3d)
    f0 = np.zeros(phase.shape)
    f0[f0.shape[0]//2 - radius : f0.shape[0]//2 + radius + 1,
       f0.shape[1]//2 - radius : f0.shape[1]//2 + radius + 1,
       f0.shape[2]//2 - radius : f0.shape[2]//2 + radius + 1] = kernel_3d
    f0 = fftnc(f0 * np.sqrt(f0.size))

    local_phase = phase - ifftnc(f0 * fftnc(phase))
    valid_point = ifftnc(f0 * fftnc(mask))
    return np.real(local_phase), np.real(valid_point)

def v_sharp(unwrapped_phase, brain_mask, voxel_size = None, pad_size = None, smv_size = 12):
    """3D background phase removal for 3D GRE data.
    Inputs:
    unwrapped_phase: 3D unwrapped phase using Laplacian unwrapping
    brain_mask: brain mask 
    voxel_size: spatial resoluiton 
    pad_size: size for padarray to increase numerical accuracy
    smv_size: filtering size, default value = 12
    
    Output:
    phase_wo_deconv: 3D Filtered TissuePhase
    final_mask: eroded mask
    
    Steven Cao
    Hongjiang Wei, PhD
    Chunlei Liu, PhD
    University of California, Berkeley
    """
    if voxel_size is None: voxel_size = [1, 1, 1]
    if pad_size is None: pad_size = [12, 12, 12]
    pad_size = tuple([(size, size) for size in pad_size])
    assert unwrapped_phase.shape == brain_mask.shape, 'Phase and mask should be same shape'
    if len(unwrapped_phase.shape) == 3:
        unwrapped_phase = unwrapped_phase.reshape(tuple(list(unwrapped_phase.shape)+[1]))
        brain_mask = brain_mask.reshape(tuple(list(brain_mask.shape)+[1]))
    else:
        unwrapped_phase = np.array(unwrapped_phase)
        brain_mask = np.array(brain_mask)

    phase_holder = []
    mask_holder = []
    for echo in range(unwrapped_phase.shape[3]):
        phase = unwrapped_phase[:,:,:,echo]
        mask = brain_mask[:,:,:,echo]
        #Preprocessing
        phase_wo_deconv = np.zeros(phase.shape, dtype = np.float)
        final_mask = np.zeros(phase.shape, dtype = np.bool)
        bounding_box, _ = mask_bounding_box(mask.astype(int), voxel_size) #pad_size decided here?
        if mask.shape[2] % 2 == 0:
            mask = append_zeros(mask, 2, 2)
        else:
            mask = append_zeros(mask, 2)

        phase = phase[bounding_box[0,0]:bounding_box[0,1],
                      bounding_box[1,0]:bounding_box[1,1],
                      bounding_box[2,0]:bounding_box[2,1]]
        mask = mask[bounding_box[0,0]:bounding_box[0,1],
                    bounding_box[1,0]:bounding_box[1,1],
                    bounding_box[2,0]:bounding_box[2,1]]
        phase = np.pad(phase, pad_size, 'constant')
        mask = np.pad(mask, pad_size, 'constant')

        #Iterative Filtering
        print('Start 3D V-SHARP, preparation...')

        xx, yy, zz = np.arange(0, phase.shape[0]), np.arange(0, phase.shape[1]), np.arange(0, phase.shape[2])
        xx, yy, zz = xx * voxel_size[0], yy * voxel_size[1], zz * voxel_size[2]
        field_of_view = np.multiply(voxel_size, phase.shape[0:3]).astype(int)
        points = np.meshgrid(np.arange(0, field_of_view[0]),
                             np.arange(0, field_of_view[1]),
                             np.arange(0, field_of_view[2]))
        points = np.vstack(map(np.ravel, points)).T
        phase_uwp_upsampled = interp3(xx, yy, zz, phase, points, field_of_view)
        mask_upsampled = interp3(xx, yy, zz, mask, points, field_of_view)
        mask_upsampled = mask_upsampled > 0.5

        mask_shape_old = np.array(mask_upsampled.shape)
        for i in range(len(mask_shape_old)):
            if mask_shape_old[i] % 2 == 1:
                phase_uwp_upsampled = append_zeros(phase_uwp_upsampled, i)
                mask_upsampled = append_zeros(mask_upsampled, i)
        phi_filtered = np.zeros(mask_upsampled.shape, dtype = np.float)

        print('Iterating R from 1 mm to', smv_size, 'mm towards the center...')
        for i in range(1, smv_size + 1):
            local_phase, valid_point = my_removal(phase_uwp_upsampled, mask_upsampled, i)
            index = np.absolute(valid_point-1) < 1e-6
            phi_filtered[index] = local_phase[index]
            if i == 2:
                final_mask_pre = index
            print(i, end = ' ')
            if i % 20 == 0 and i != 0:
                print()

        for i in range(len(mask_shape_old)):
            if mask_shape_old[i] % 2 == 1:
                final_mask_pre = np.delete(final_mask_pre, 1, i)
                phi_filtered = np.delete(phi_filtered, 1, i)
                phase_uwp_upsampled = np.delete(phase_uwp_upsampled, 1, i)

        xx, yy, zz = np.arange(0, field_of_view[0]), np.arange(0, field_of_view[1]), np.arange(0, field_of_view[2])
        points = np.meshgrid(np.arange(0, phase.shape[0]) * voxel_size[0],
                             np.arange(0, phase.shape[1]) * voxel_size[1],
                             np.arange(0, phase.shape[2]) * voxel_size[2])
        points = np.vstack(map(np.ravel, points)).T

        phi_filtered *= final_mask_pre
        residual_phase = interp3(xx, yy, zz, phase_uwp_upsampled - phi_filtered, points, phase.shape)
        v_sharp_phase = phase - residual_phase
        final_mask_pre = interp3(xx, yy, zz, final_mask_pre, points, phase.shape)
        final_mask_pre = final_mask_pre > 0.5

        v_sharp_phase = v_sharp_phase[pad_size[0][0]:v_sharp_phase.shape[0]-pad_size[0][1],
                                      pad_size[1][0]:v_sharp_phase.shape[1]-pad_size[1][1],
                                      pad_size[2][0]:v_sharp_phase.shape[2]-pad_size[2][1]]
        phase_wo_deconv[bounding_box[0,0]:bounding_box[0,1],
                        bounding_box[1,0]:bounding_box[1,1],
                        bounding_box[2,0]:bounding_box[2,1]] = v_sharp_phase
        final_mask_pre = final_mask_pre[pad_size[0][0]:final_mask_pre.shape[0]-pad_size[0][1],
                                        pad_size[1][0]:final_mask_pre.shape[1]-pad_size[1][1],
                                        pad_size[2][0]:final_mask_pre.shape[2]-pad_size[2][1]]
        final_mask[bounding_box[0,0]:bounding_box[0,1],
                   bounding_box[1,0]:bounding_box[1,1],
                   bounding_box[2,0]:bounding_box[2,1]] = final_mask_pre
        phase_wo_deconv = phase_wo_deconv * final_mask

        phase_holder.append(phase_wo_deconv)
        mask_holder.append(final_mask)

    return np.real(np.stack(phase_holder, axis=3)), np.stack(mask_holder, axis=3)
    
    
