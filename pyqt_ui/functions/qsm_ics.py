import numpy as np
from functions.funcs import calc_d2_matrix

def qsm_ics(vsharp_phase, brain_mask, magnitude, voxel_size = None, B0_dir = None,
            B0 = 3, TE = 1, alpha = 5e-5, beta = 3e-4, max_iter = 50,
            mu2 = 1, tol_update = 1, delta_tol = 1e-6):
    """
    """
    if voxel_size is None: voxel_size = [1, 1, 1]
    if B0_dir is None: B0_dir = [0, 0, 1]
    assert vsharp_phase.shape == brain_mask.shape and brain_mask.shape == magnitude.shape, 'Phase, magnitude, and mask should be same shape'
    if len(vsharp_phase.shape) == 3:
        vsharp_phase = vsharp_phase.reshape(tuple(list(vsharp_phase.shape)+[1]))
        brain_mask = brain_mask.reshape(tuple(list(brain_mask.shape)+[1]))
        magnitude = magnitude.reshape(tuple(list(magnitude.shape)+[1]))
    else:
        vsharp_phase = np.array(vsharp_phase)
        brain_mask = np.array(brain_mask)
        magnitude = np.array(magnitude)
    vsharp_phase *= brain_mask
    magnitude *= brain_mask

    susc_holder = []

    for echo in range(vsharp_phase.shape[3]):
        phase, mask, mag = vsharp_phase[:,:,:,echo], brain_mask[:,:,:,echo], magnitude[:,:,:,echo]
    
        tissue_phase = phase / (TE * B0 * 2 * np.pi * 42.58)

        mag = mag / np.max(np.abs(mag))
        d2 = calc_d2_matrix(tissue_phase.shape, voxel_size, B0_dir)

        susc_holder.append(-1 * qsm_ics_solver(d2, tissue_phase, mag, alpha, beta, max_iter,
                                               mu2, tol_update, delta_tol))

    return np.stack(susc_holder, axis=3)

def qsm_ics_solver(K, phase, W, alpha1, mu1, max_iter,
                   mu2 = 1, tol_update = 1, delta_tol = 1e-6):
    
    mu0, alpha0 = mu1 * 2, alpha1 * 2
    W *= W

    N = phase.shape

    #precompute gradient-related matrices
    k1, k2, k3 = np.meshgrid(np.arange(0, N[0]),
                             np.arange(0, N[1]),
                             np.arange(0, N[2]))
    E1 = 1 - np.exp(2j * np.pi * k1 / N[0])
    E2 = 1 - np.exp(2j * np.pi * k2 / N[1])
    E3 = 1 - np.exp(2j * np.pi * k3 / N[2])

    Kt = np.conj(K)
    Et1 = np.conj(E1)
    Et2 = np.conj(E2)
    Et3 = np.conj(E3)

    E1tE1 = Et1*E1
    E2tE2 = Et2*E2
    E3tE3 = Et3*E3
    
    mu0_over_2_E1tE2 = mu0/2*Et1*E2
    mu0_over_2_E1tE3 = mu0/2*Et1*E3
    mu0_over_2_E2tE3 = mu0/2*Et2*E3

    a0 = mu2*Kt*K
    a0_mu1_E_sos = a0 + mu1*(E1tE1 + E2tE2 + E3tE3)
    mu1I_mu0_E_wsos1 = mu1 + mu0*(E1tE1 + (E2tE2 + E3tE3)/2)
    mu1I_mu0_E_wsos2 = mu1 + mu0*(E1tE1/2 + E2tE2 + E3tE3/2)
    mu1I_mu0_E_wsos3 = mu1 + mu0*((E1tE1 + E2tE2)/2 + E3tE3)

    #precomputation for Cramer's Rule

    a1 = a0_mu1_E_sos
    a2 = mu1I_mu0_E_wsos1
    a3 = mu1I_mu0_E_wsos2
    a4 = mu1I_mu0_E_wsos3
    a5 = -mu1*E1
    a6 = -mu1*E2
    a7 = mu0_over_2_E1tE2
    a8 = -mu1*E3
    a9 = mu0_over_2_E1tE3
    a10 = mu0_over_2_E2tE3
    a5t = np.conj(a5)
    a6t = np.conj(a6)
    a7t = np.conj(a7)
    a8t = np.conj(a8)
    a9t = np.conj(a9)
    a10t = np.conj(a10)

    # For x
    D11 = a2*a3*a4    + a7t*a9*a10t + a7*a9t*a10  - a3*a9*a9t   - a2*a10*a10t     - a4*a7*a7t
    D21 = a3*a4*a5t   + a6t*a9*a10t + a7*a8t*a10  - a3*a8t*a9   - a5t*a10*a10t    - a4*a6t*a7
    D31 = a4*a5t*a7t  + a6t*a9*a9t  + a2*a8t*a10  - a7t*a8t*a9  - a5t*a9t*a10     - a2*a4*a6t
    D41 = a5t*a7t*a10t + a6t*a7*a9t + a2*a3*a8t   - a7*a7t*a8t  - a3*a5t*a9t      - a2*a6t*a10t

    # For vx
    D12 = a3*a4*a5    + a7t*a8*a10t + a6*a9t*a10  - a3*a8*a9t   - a5*a10*a10t - a4*a6*a7t
    D22 = a1*a3*a4    + a6t*a8*a10t + a6*a8t*a10  - a3*a8*a8t   - a1*a10*a10t - a4*a6*a6t
    D32 = a1*a4*a7t   + a6t*a8*a9t  + a5*a8t*a10  - a7t*a8*a8t  - a1*a9t*a10  - a4*a5*a6t
    D42 = a1*a7t*a10t + a6*a6t*a9t  + a3*a5*a8t   - a6*a7t*a8t  - a1*a3*a9t   - a5*a6t*a10t

    # For vy
    D13 = a4*a5*a7 + a2*a8*a10t + a6*a9*a9t - a7*a8*a9t - a5*a9*a10t - a2*a4*a6
    D23 = a1*a4*a7 + a5t*a8*a10t +a6*a8t*a9 - a7*a8*a8t - a1*a9*a10t - a4*a5t*a6
    D33 = a1*a2*a4 + a5t*a8*a9t + a5*a8t*a9 - a2*a8*a8t - a1*a9*a9t - a4*a5*a5t
    D43 = a1*a2*a10t + a5t*a6*a9t + a5*a7*a8t - a2*a6*a8t - a1*a7*a9t - a5*a5t*a10t

    # For vz
    D14 = a5*a7*a10 + a2*a3*a8 + a6*a7t*a9 - a7*a7t*a8 - a3*a5*a9 -a2*a6*a10
    D24 = a1*a7*a10 + a3*a5t*a8 + a6*a6t*a9 - a6t*a7*a8 - a1*a3*a9 - a5t*a6*a10
    D34 = a1*a2*a10 + a5t*a7t*a8 + a5*a6t*a9 - a2*a6t*a8 - a1*a7t*a9 - a5*a5t*a10
    D44 = a1*a2*a3 + a5t*a6*a7t + a5*a6t*a7 - a2*a6*a6t - a1*a7*a7t - a3*a5*a5t

    det_A = a1*D11 - a5*D21 + a6*D31 - a8*D41
    det_Ainv = 1 / (np.finfo(float).eps+det_A)

    #precond = True
    z2 = W * phase /(W + mu2);

    s2 = np.zeros(N) 

    # Allocate memory for first order gradient
    s1_1 = np.zeros(N)
    z1_1 = np.zeros(N)
    s1_2 = np.zeros(N)
    z1_2 = np.zeros(N)
    s1_3 = np.zeros(N)
    z1_3 = np.zeros(N)

    # Allocate memory for symmetrized gradient
    s0_1 = np.zeros(N)
    z0_1 = np.zeros(N) 
    s0_2 = np.zeros(N)
    z0_2 = np.zeros(N)
    s0_3 = np.zeros(N)
    z0_3 = np.zeros(N) 
    s0_4 = np.zeros(N)
    z0_4 = np.zeros(N)
    s0_5 = np.zeros(N)
    z0_5 = np.zeros(N)
    s0_6 = np.zeros(N)
    z0_6 = np.zeros(N)

    x_prev = np.zeros(N)

    for i in range(max_iter):
        # Update x and v
        F_z0_minus_s0_1 = np.fft.fftn(z0_1 - s0_1)
        F_z0_minus_s0_2 = np.fft.fftn(z0_2 - s0_2)
        F_z0_minus_s0_3 = np.fft.fftn(z0_3 - s0_3)
        F_z0_minus_s0_4 = np.fft.fftn(z0_4 - s0_4)
        F_z0_minus_s0_5 = np.fft.fftn(z0_5 - s0_5)
        F_z0_minus_s0_6 = np.fft.fftn(z0_6 - s0_6)


        F_z1_minus_s1_1 = np.fft.fftn(z1_1 - s1_1)
        F_z1_minus_s1_2 = np.fft.fftn(z1_2 - s1_2)
        F_z1_minus_s1_3 = np.fft.fftn(z1_3 - s1_3)

        rhs0 = mu2*(Kt*np.fft.fftn( z2-s2 )) #np.fft.fftn(mu2*real(np.fft.ifftn(Kt*( z2-s2 ))) 
        rhs1    =  rhs0                + mu1*(Et1*F_z1_minus_s1_1 + Et2*F_z1_minus_s1_2 + Et3*F_z1_minus_s1_3)
        rhs2    = -mu1*F_z1_minus_s1_1 + mu0*(Et1*F_z0_minus_s0_1 + Et2*F_z0_minus_s0_4 + Et3*F_z0_minus_s0_5)
        rhs3    = -mu1*F_z1_minus_s1_2 + mu0*(Et2*F_z0_minus_s0_2 + Et1*F_z0_minus_s0_4 + Et3*F_z0_minus_s0_6)
        rhs4    = -mu1*F_z1_minus_s1_3 + mu0*(Et3*F_z0_minus_s0_3 + Et1*F_z0_minus_s0_5 + Et2*F_z0_minus_s0_6)

        # Cramer's rule
        Fx = (rhs1*D11 - rhs2*D21 + rhs3*D31 - rhs4*D41) * det_Ainv
        Fv1 = (-rhs1*D12 + rhs2*D22 - rhs3*D32 + rhs4*D42) * det_Ainv
        Fv2 = (rhs1*D13 - rhs2*D23 + rhs3*D33 - rhs4*D43) * det_Ainv
        Fv3 = (-rhs1*D14 +rhs2*D24 - rhs3*D34 + rhs4*D44) * det_Ainv

        x = np.real(np.fft.ifftn(Fx))
        v1 = np.real(np.fft.ifftn(Fv1))
        v2 = np.real(np.fft.ifftn(Fv2))
        v3 = np.real(np.fft.ifftn(Fv3))
        
        x_update = 100 * np.linalg.norm(x.flatten()-x_prev.flatten()) / np.linalg.norm(x.flatten())
        print('Iter:', i + 1, ', Update:', x_update)

        if x_update < tol_update:
            return x

        # Compute gradients for z0 and z1 update
        Dx1 = np.real(np.fft.ifftn(E1*Fx))
        Dx2 = np.real(np.fft.ifftn(E2*Fx))
        Dx3 = np.real(np.fft.ifftn(E3*Fx))

        E_v1 = np.real(np.fft.ifftn(E1*Fv1))
        E_v2 = np.real(np.fft.ifftn(E2*Fv2))
        E_v3 = np.real(np.fft.ifftn(E3*Fv3))
        E_v4 = np.real(np.fft.ifftn(E1*Fv2 + E2*Fv1))/2
        E_v5 = np.real(np.fft.ifftn(E1*Fv3 + E3*Fv1))/2
        E_v6 = np.real(np.fft.ifftn(E2*Fv3 + E3*Fv2))/2
        
        # Update z0: Symm grad
        z0_1 = max(np.max(np.abs(E_v1 + s0_1)-alpha0/mu0),0)*np.sign(E_v1 + s0_1)
        z0_2 = max(np.max(np.abs(E_v2 + s0_2)-alpha0/mu0),0)*np.sign(E_v2 + s0_2)
        z0_3 = max(np.max(np.abs(E_v3 + s0_3)-alpha0/mu0),0)*np.sign(E_v3 + s0_3)
        z0_4 = max(np.max(np.abs(E_v4 + s0_4)-alpha0/mu0),0)*np.sign(E_v4 + s0_4)
        z0_5 = max(np.max(np.abs(E_v5 + s0_5)-alpha0/mu0),0)*np.sign(E_v5 + s0_5)
        z0_6 = max(np.max(np.abs(E_v6 + s0_6)-alpha0/mu0),0)*np.sign(E_v6 + s0_6)
        
        # Update z1: Grad
        #regweight omitted
        z1_1 = max(np.max(np.abs(Dx1-v1+s1_1)-alpha1/mu1),0)*np.sign(Dx1-v1+s1_1)
        z1_2 = max(np.max(np.abs(Dx2-v2+s1_2)-alpha1/mu1),0)*np.sign(Dx2-v2+s1_2)
        z1_3 = max(np.max(np.abs(Dx3-v3+s1_3)-alpha1/mu1),0)*np.sign(Dx3-v3+s1_3)

        rhs_z2 = mu2*np.real(np.fft.ifftn(K*Fx)+s2)        
        z2 =  rhs_z2 / mu2

        # Newton-Raphson method
        delta = np.inf
        inn = 0
        while delta > delta_tol and inn < 50:
            inn += 1
            norm_old = np.linalg.norm(z2.flatten())
            
            update = ( W * np.sin(z2 - phase) + mu2*z2 - rhs_z2 ) / ( W * np.cos(z2 - phase) + mu2 )
            
            z2 = z2 - update     
            delta = np.linalg.norm(update.flatten()) / norm_old

        # Update s0 and s1
        s0_1 = s0_1 + E_v1-z0_1
        s0_2 = s0_2 + E_v2-z0_2
        s0_3 = s0_3 + E_v3-z0_3
        s0_4 = s0_4 + E_v4-z0_4
        s0_5 = s0_5 + E_v5-z0_5
        s0_6 = s0_6 + E_v6-z0_6
        
        s1_1 = s1_1 + Dx1-v1-z1_1
        s1_2 = s1_2 + Dx2-v2-z1_2
        s1_3 = s1_3 + Dx3-v3-z1_3
        
        s2 = s2 + np.real(np.fft.ifftn(K*Fx)) - z2
        
        x_prev = x
    return x

        
    
