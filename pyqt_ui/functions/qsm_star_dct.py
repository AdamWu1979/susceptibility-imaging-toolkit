import numpy as np
from functions.funcs import fftnc, ifftnc, calc_d2_matrix, dct3, idct3
from functions.sparsa import sparsa

def scaling_factor(B0, TE_ms):
    gamma = 42.575 * 2 * np.pi
    TE = TE_ms * 1e-3
    return 1 / (gamma * B0 * TE)

def qsm_star(phase_all, mask_all,
             voxel_size = [1,1,1], pad_size = [0,0,0], B0_dir = [0,0,1],
             B0 = 3, TE = 1, tau = 1e-6, d2_threshold = .065):
    pad_size = tuple([(size,size) for size in pad_size])
    assert phase_all.shape == mask_all.shape, "Phase and mask should be same shape."
    if len(phase_all.shape) == 3:
        phase_all, mask_all = phase_all[:,:,:,None], mask_all[:,:,:,None] #add echo dim

    #calculate thresholded dipole kernel
    #d2 values below threshold are set to threshold
    #phase_k = d2 * susc_k
    d2 = calc_d2_matrix(phase_all.shape[0:3], voxel_size, B0_dir)
    undersampling = np.abs(d2) > d2_threshold
    below = d2[np.logical_not(undersampling)]
    below[below >= 0] = d2_threshold
    below[below < 0] = -d2_threshold
    d2[np.logical_not(undersampling)], below = below, None
    
    susc_all = []
    #debug = []
    for echo in range(phase_all.shape[3]): #calculate susc for each echo
        #solve the following problem:
        #min_x || y - Ax ||_2^2 + tau || x ||_1 where
        #   x = susceptibility in dct basis
        #   y = undersampled susceptibility estimate in k-space
        #   A = undersampled DFT * idct
        #
        phase, mask = (np.pad(phase_all[:,:,:,echo], pad_size, 'constant'),
                       np.pad(mask_all[:,:,:,echo], pad_size, 'constant'))
        #x_initial = dct3(ifftnc(fftnc(phase) / d2 * undersampling))
        x_initial = np.zeros(phase.shape)
        y = fftnc(phase) / d2 * undersampling #i think this line is incorrect
        A = lambda x: undersampling * fftnc(idct3(x))
        AT = lambda x: dct3(ifftnc(undersampling * x))
        #A = lambda x: undersampling * fftnc(x)
        #AT = lambda x: ifftnc(undersampling * x)

        susc = sparsa(y, x_initial, A, AT, tau, verbose = True,
                      min_iter = 1)
        #don't forget to scale

        susc = idct3(np.real(susc))
        susc *= mask
        susc = susc[pad_size[0][0]:susc.shape[0]-pad_size[0][1],
                    pad_size[1][0]:susc.shape[0]-pad_size[1][1],
                    pad_size[2][0]:susc.shape[0]-pad_size[2][1]]
        susc *= scaling_factor(B0, TE)
        susc_all.append(np.real(susc))
        
    return np.stack(susc_all, axis = 3)
                
