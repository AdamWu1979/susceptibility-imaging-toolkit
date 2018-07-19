import numpy as np
import nibabel as nib
# from matplotlib import pyplot as plt

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