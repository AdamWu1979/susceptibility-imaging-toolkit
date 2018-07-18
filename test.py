import numpy as np
import nibabel as nib
# from matplotlib import pyplot as plt

from create_mask import create_mask_qsm
from laplacian_unwrap import laplacian_unwrap
from v_sharp import v_sharp
from qsm_star import qsm_star

mag = nib.load('N041/eswan_mag.nii.gz').get_data()
phase = nib.load('N041/eswan_phase.nii.gz').get_data()
voxel_size = np.array([0.4688, 0.4688, 2])
mask = create_mask_qsm(mag[:,:,:,7], voxel_size)
unwrapped, _ = laplacian_unwrap(phase, voxel_size)
unwrapped = np.sum(unwrapped, axis = 3)
tissue, mask2 = v_sharp(unwrapped, mask, voxel_size)
TE = 138
susc = qsm_star(tissue, mask2, voxel_size, TE)
# plt.gray()