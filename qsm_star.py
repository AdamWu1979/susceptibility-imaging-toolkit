import numpy as np
from functions import ifftnc, fftnc, pad, unpad

def qsm_star(phase, mask, voxel_size, TE, pad_size = (8,8,20), 
             B0 = 3, B0_dir = (0,0,1), tau = 1e-6, d2_thresh = .065):
    d2 = calc_d2_matrix(phase.shape, voxel_size, B0_dir)
    # sample = np.abs(d2) > d2_thresh
    d2 = pad(d2, pad_size)
    phase, mask = pad(phase, pad_size), pad(mask, pad_size)
    A = lambda x: ifftnc(d2 * fftnc(x * mask))
    AT = A
    susc = sparsa(phase, np.zeros(phase.shape), A, AT, tau)
    susc *= mask
    susc = unpad(susc, pad_size)
    susc *= scaling_factor(B0, TE)
    return np.real(susc)

def scaling_factor(B0, TE_ms):
    gamma = 42.575 * 2 * np.pi
    TE = TE_ms * 1e-3
    return 1 / (gamma * B0 * TE)

def calc_d2_matrix(shape, voxel_size, B0_dir):
    shape = np.array(shape)
    voxel_size = np.array(voxel_size)
    B0_dir = np.array(B0_dir)
    field_of_view = shape * voxel_size
    ry, rx, rz = np.meshgrid(np.arange(-shape[1]//2, shape[1]//2),
                             np.arange(-shape[0]//2, shape[0]//2),
                             np.arange(-shape[2]//2, shape[2]//2))
    rx, ry, rz = rx/field_of_view[0], ry/field_of_view[1], rz/field_of_view[2]
    sq_dist = rx**2 + ry**2 + rz**2
    sq_dist[sq_dist==0] = 1e-6
    d2 = ((B0_dir[0]*rx + B0_dir[1]*ry + B0_dir[2]*rz)**2)/sq_dist
    d2 = (1/3 - d2)
    return d2

def soft(x, tau):
    if np.sum(np.abs(tau)) == 0:
        return x
    else:
        y = np.abs(x) - tau
        y[y < 0] = 0
        y = y / (y + tau) * x
        return y

def sparsa(y, x, A, AT, tau, verbose = False, min_iter = 5, alpha = 1):
    """
    SpaRSA solver modified from the Matlab version by Mario Figueiredo, 
    Robert Nowak, and Stephen Wright (http://www.lx.it.pt/~mtf/SpaRSA/)
    
    Inputs:
    y: 1D vector or 2D array (image) of observations
    x: initial guess for the solution x
    A: matrix to be applied to x (in function form)
    AT: transpose of A (in function form)
    tau: regularization parameter (scalar)
    
    Compared to the Matlab code, it uses the following options:
    psi = soft
    phi = l1 norm
    StopCriterion = 0 - algorithm stops when the relative 
                        change in the number of non-zero 
                        components of the estimate falls 
                        below 'tolerance'
    debias = 0 (no)
    BB_variant = 1 - standard BB choice  s'r/s's
    monotone = 0 (don't enforce monotonic decrease of f)
    safeguard = 0 (don't enforce a "sufficient decrease" over the largest
                   objective value of the past M iterations.)
    continuation = 0 (don't do the continuation procedure)
    
    Steven Cao
    University of California, Berkeley
    
    ==========================================================================    
    SpaRSA version 2.0, December 31, 2007
     
    This function solves the convex problem 
    
    arg min_x = 0.5*|| y - A x ||_2^2 + tau phi(x)
    
    d'*A'*(A x - y) + 0.5*alpha*d'*d 
    
    where alpha is obtained from a BB formula. In a monotone variant, alpha is
    increased until we see a decreasein the original objective function over
    this step. 
    
    -----------------------------------------------------------------------
    Copyright (2007): Mario Figueiredo, Robert Nowak, Stephen Wright
    ----------------------------------------------------------------------
    """
    tolerance = 0.01 #min amount of changes needed to keep going
    max_iter = 100
    min_alpha = 1e-30
    max_alpha = 1e30
    
    psi_function = soft
    phi_function = lambda x: np.sum(np.abs(x)) #phi is l1
    
    nonzero_x = x != 0
    
    resid = A(x) - y
    gradient = AT(resid)
    alpha = 1
    f = (0.5 * np.vdot(resid.flatten(), resid.flatten())
        + tau * phi_function(x))
    
    iterations = 0
    while True:
        gradient = AT(resid)
        prev_x = x
        prev_resid = resid
        
        x = psi_function(prev_x - gradient * (1/alpha),
                         tau/alpha)
        dx = x - prev_x
        Adx = A(dx)
        resid = prev_resid + Adx
        f = (0.5 * np.vdot(resid.flatten(), resid.flatten())
            + tau * phi_function(x))
        
        dd = np.vdot(dx.flatten(), dx.flatten())
        dGd = np.vdot(Adx.flatten(), Adx.flatten())
        alpha = dGd / (np.finfo(np.double).tiny + dd)
        alpha = min(max_alpha, max(min_alpha, alpha))
        iterations += 1
        
        # compute the stopping criterion based on the change
        # in the number of non-zero components of the estimate
        prev_nonzero_x = nonzero_x
        nonzero_x = x != 0
        changes = np.sum(nonzero_x != prev_nonzero_x) / np.sum(nonzero_x)
        if verbose:
            print('-----------------------',
                  '\nObjective f:', f,
                  '\nAlpha:', alpha,
                  '\nPercent change in x:', changes,
                  '\nChange tolerance:', tolerance, 
                  '\nIterations:', iterations)
        if (changes <= tolerance and
           iterations >= min_iter and
           iterations < max_iter):
           return x