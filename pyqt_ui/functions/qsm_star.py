import numpy as np
from functions.funcs import fftnc, ifftnc, calc_d2_matrix
from functions.sparsa import sparsa

def scaling_factor(B0, TE_ms):
    gamma = 42.575 * 2 * np.pi
    TE = TE_ms * 1e-3
    return 1 / (gamma * B0 * TE)

def qsm_star(phase_all, mask_all,
             voxel_size = [1,1,1], pad_size = [0,0,0], B0_dir = [0,0,1],
             B0 = 3, TE = 1, tau = 1e-6, d2_threshold = .065):
    """
    [5] fast STAR-QSM (~30 sec): Quantative Susceptibility Mapping
    Inputs:
    vsharp_phase: tissue phase after V_SHARP (3d: x, y, slice number)
    brain_mask: mask after V_SHARP
    voxel_size: resolution (length 3, default [1,1,1])
    pad_size: amount to pad phase and mask (length 3, default [1,1,1])
    B0_dir: direction of magnetic field (default [0,0,1])
    B0: magnetic field strength in Tesla (default 3)
    TE: echo time (default 1)
    tau: SpaRSA parameter (default 0.000001)

    Steven Cao
    Hongjiang Wei, PhD
    Chunlei Liu, PhD
    University of California, Berkeley
    """
    pad_size = tuple([(size,size) for size in pad_size])
    assert phase_all.shape == mask_all.shape, "Phase and mask should be same shape."
    if len(phase_all.shape) == 3:
        phase_all, mask_all = phase_all[:,:,:,None], mask_all[:,:,:,None] #add echo dim

    #calculate thresholded dipole kernel
    #   d2 values below threshold are set to threshold
    #   phase_k = d2 * susc_k
    d2 = calc_d2_matrix(phase_all.shape[0:3], voxel_size, B0_dir)
    d2_t = np.array(d2)
    undersampling = np.abs(d2) > d2_threshold
    below = d2[np.logical_not(undersampling)]
    below[below >= 0] = d2_threshold
    below[below < 0] = -d2_threshold
    d2_t[np.logical_not(undersampling)], below = below, None
    d2, d2_t = (np.pad(d2[:,:,:], pad_size, 'constant'),
                np.pad(d2_t[:,:,:], pad_size, 'constant'))
    
    susc_all = []
    #debug = []
    for echo in range(phase_all.shape[3]): #calculate susc for each echo
        #solve the following problem:
        #min_x || y - Ax ||_2^2 + tau || x ||_1 where
        #   x = susceptibility in image basis
        #   y = phase
        #   Ax = ifftnc ( d2 * fftnc ( susc * mask_i ) )
        #
        phase, mask = (np.pad(phase_all[:,:,:,echo], pad_size, 'constant'),
                       np.pad(mask_all[:,:,:,echo], pad_size, 'constant'))
        x_initial = np.zeros(phase.shape)
        y = phase
        A = lambda x: ifftnc(d2 * fftnc(x * mask))
        AT = A
        
        susc = sparsa(y, x_initial, A, AT, tau, verbose = True)
        susc = susc.reshape(phase.shape)
        susc *= mask
        susc = susc[pad_size[0][0]:susc.shape[0]-pad_size[0][1],
                    pad_size[1][0]:susc.shape[0]-pad_size[1][1],
                    pad_size[2][0]:susc.shape[0]-pad_size[2][1]]
        susc *= scaling_factor(B0, TE)
        susc_all.append(np.real(susc))
        
    return np.stack(susc_all, axis = 3)
                
