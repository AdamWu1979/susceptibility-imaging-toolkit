import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

from read_dicom import read_dicom
from create_mask import create_mask_qsm
from laplacian_unwrap import laplacian_unwrap
from v_sharp import v_sharp
from qsm_star import qsm_star

data, params = read_dicom('N041/dicom')
mag, phase = np.abs(data), np.angle(data)
voxel_size = params['voxel_size']
TE = sum(params['TE'])
mask = create_mask_qsm(mag[:,:,:,7], voxel_size)
unwrapped, _ = laplacian_unwrap(phase, voxel_size)
unwrapped = np.sum(unwrapped, axis = 3)
tissue, mask2 = v_sharp(unwrapped, mask, voxel_size)
susc = qsm_star(tissue, mask2, voxel_size, TE)
# plt.gray()

def save(file, img):
    img = np.rot90(img)
    img = np.flip(img, 0)
    image = nib.Nifti1Image(img, np.eye(4))
    nib.save(image, file)
    
save('N041/tissue.nii', tissue)
save('N041/susc.nii', susc)
save('N041/mask.nii', mask.astype(int))
save('N041/mask2.nii', mask2.astype(int))
save('N041/phase.nii', phase)
save('N041/unwrapped.nii', unwrapped)