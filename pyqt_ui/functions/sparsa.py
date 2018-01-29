import numpy as np

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
    SpaRSA function modified and shortened from the same function written in
    Matlab by Mario Figueiredo, Robert Nowak, Stephen Wright
    (http://www.lx.it.pt/~mtf/SpaRSA/)
    
    Inputs:
    y: 1D vector or 2D array (image) of observations
    x: initial guess for the solution x
    A: matrix to be applied to x (in function form)
    AT: transpose of A (in function form)
    tau: regularization parameter (scalar)
    
    Assumes the following variant of SpaRSA (options can be seen in the Matlab version):
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
                     continuation procedure - start at a large value of tau and decrease it
    
    Also only returns x, does not return objective func values, taus, mses, etc...
    
    Steven Cao
    University of California, Berkeley
    
    ===========================================================================    
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
    
    Aty = AT(y) #precompute Aty
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
        prev_f = f
        
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
        
        #stop criterion = 0:
        #compute the stopping criterion based on the change
        #in the number of non-zero components of the estimate
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
        
