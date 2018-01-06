import numpy as np
from functions.funcs import fftnc, ifftnc, calc_d2_matrix
from functions.sparsa import SpaRSA_HW

class iFDF:
    def __init__(self, mask, adj = 0):
        self.mask = np.array(mask)
        self.adjoint = adj
    def ctranspose(self):
        return iFDF(self.mask, self.adjoint^1)
    def mtimes(self, b):
        return ifftnc(self.mask * fftnc(b))
        
class SusceptibilityMap:
    def __init__(self, mask, d2, adj = 0):
        self.mask = np.array(mask)
        self.d2 = np.array(d2)
        self.adjoint = adj
    def ctranspose(self):
        return SusceptibilityMap(self.mask, self.d2, self.adjoint^1)
    def mtimes(self, b):
        if self.adjoint:
            return ifftnc(b * self.d2) * self.mask
        else:
            return self.d2 * fftnc(b * self.mask)

def scaling_factor(B0, TE_ms):
    gamma = 42.575 * 2 * np.pi
    TE = TE_ms * 1e-3
    return 1 / (gamma * B0 * TE)
    
def qsm_star(vsharp_phase, brain_mask, voxel_size = None, pad_size = None, B0_dir = None, B0 = 3, TE = 1, tau = 0.000001):
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
    if voxel_size is None: voxel_size = [1, 1, 1]
    if pad_size is None: pad_size = [0, 0, 0]
    if B0_dir is None: B0_dir = [0, 0, 1]
    pad_size = tuple([(size, size) for size in pad_size])
    assert vsharp_phase.shape == brain_mask.shape, 'Phase and mask should be same shape'
    if len(vsharp_phase.shape) == 3:
        vsharp_phase = vsharp_phase.reshape(tuple(list(vsharp_phase.shape)+[1]))
        brain_mask = brain_mask.reshape(tuple(list(brain_mask.shape)+[1]))
    else:
        vsharp_phase = np.array(vsharp_phase)
        brain_mask = np.array(brain_mask)
    vsharp_phase *= brain_mask

    susc_holder = []
    for echo in range(vsharp_phase.shape[3]):
        phase = vsharp_phase[:,:,:,echo]
        mask = brain_mask[:,:,:,echo]
        
        print('Initialization ...')
        phase = np.pad(phase, pad_size, 'constant')
        mask = np.pad(mask, pad_size, 'constant')

        d2 = calc_d2_matrix(phase.shape, voxel_size, B0_dir)
        d2_iFDF = iFDF(d2)
        
        threshold = .065
        d3 = np.array(d2)
        temp = d2[np.abs(d2)<threshold]
        temp[temp>=0] = threshold
        temp[temp<0] = -threshold
        d3[np.abs(d2)<threshold], temp = temp, None

        phase_k = fftnc(phase)
        x_k_divided = fftnc(ifftnc(phase_k / d3) * mask)
        k_mask = np.zeros(x_k_divided.shape)
        k_mask[np.abs(d2)>=threshold] = 1
        x_divided_threshold = ifftnc(x_k_divided * k_mask) * mask
        data = fftnc(x_divided_threshold) * k_mask
        ft = SusceptibilityMap(mask, k_mask)
        im_dc = ft.ctranspose().mtimes(data)
        scale_factor = np.max(np.abs(im_dc))
        data /= scale_factor
        im_dc /= scale_factor

        d2, threshold, d3, phase_k, x_k_divided, k_mask, x_divided_threshold, data, ft, scale_factor = None, None, None, None, None, None, None, None, None, None

        print('Starting solver.... ')
        A = lambda x: d2_iFDF.mtimes(x * mask)
        AT = lambda x: d2_iFDF.ctranspose().mtimes(x * mask)
        susceptibility = SpaRSA_HW(phase, im_dc, A, AT, tau)
        susceptibility *= mask
        susceptibility *= scaling_factor(B0, TE)
        susceptibility = np.real(susceptibility)

        susceptibility = susceptibility[pad_size[0][0]:susceptibility.shape[0]-pad_size[0][1],
                                        pad_size[1][0]:susceptibility.shape[0]-pad_size[1][1],
                                        pad_size[2][0]:susceptibility.shape[0]-pad_size[2][1]]

        susc_holder.append(susceptibility)

    return np.stack(susc_holder, axis=3)
