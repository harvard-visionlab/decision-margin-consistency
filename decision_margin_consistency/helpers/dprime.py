import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermgauss
from functools import partial

from pdb import set_trace

def adjusted_pc_edge_cases(Pc, N):
    """
    Compute the Wald-adjusted proportion correct only for edge cases (Pc = 0 or Pc = 1).

    Parameters:
    Pc (float or array-like): Proportion correct.
    N (int or array-like): Number of trials.

    Returns:
    float or ndarray: Adjusted proportion correct.
    """
    Pc = np.asarray(Pc).astype(np.float64)
    N = np.asarray(N)
    
    # Apply adjustment only to edge cases (Pc = 0 or Pc = 1)
    X = Pc * N
    Pc_adjusted = np.where((Pc == 0) | (Pc==1), (X + .5) / (N + 1), Pc)

    return Pc_adjusted

def adjusted_pc(Pc, N):
    """
    Compute the Wald-adjusted proportion correct.

    Parameters:
    Pc (float or array-like): Proportion correct.
    N (int or array-like): Number of trials.

    Returns:
    float or ndarray: Adjusted proportion correct.
    
    Reference:
    The Wald adjustment (also known as the logit transformation or adjusted Wald correction) is a 
    method used to handle extreme values of proportion correct (Pc), such as 0 or 1, when calculating 
    sensitivity (d'). This adjustment is particularly useful when sample sizes are small, and 
    Pc values are close to the bounds of [0, 1]. Without adjustment, values like 0 and 1 can lead 
    to issues with the d' calculation, as they correspond to infinite or undefined d' values.
    
    Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on 
    estimated values of d'. Behavior Research Methods, Instruments, & Computers, 27(1), 46-51.
    
    """
    Pc = np.asarray(Pc).astype(np.float64)
    N = np.asarray(N)
    
    # Wald-adjusted Pc
    X = Pc * N
    Pc_adjusted = (X + 0.5) / (N + 1)

    return Pc_adjusted

def dprime_mAFC(Pc, m, N_points = 50, N=None):
    """
    Compute the sensitivity index (d') for an M-alternative forced-choice (M-AFC) task.

    Parameters:
    Pc (float or array-like): Proportion correct (between 0 and 1).
    m (int): Number of alternatives in the forced-choice task (integer > 1).
    N (int or array-like, optional): Number of trials for each Pc score. If provided, Wald adjustment is applied.
    N_points (int): Number of quadrature points; adjust as needed for accuracy
    
    Returns:
    float or ndarray: Estimated d' value(s).
    
    References:
    Green, D. M. and Dai, H. (1991) Probability of being correct with 1 of M orthogonal signals. Perception & Psychophysics, 49, 100–101.
    Green, D. M. and Swets, J. A. (1966) Signal Detection Theory and Psychophysics Robert E. Krieger Publishing Company

    """
    Pc = np.asarray(Pc).astype(np.float64)  # Ensure Pc is a NumPy array
    
    # Apply Wald adjustment if N is provided
    if N is not None:
        Pc = adjusted_pc_edge_cases(Pc, N)
        
    # Input validation
    if not isinstance(m, int):
        raise ValueError("m must be an integer")
    if m < 2:
        raise ValueError("m must be an integer greater than 1")

    # Generate grid of dp values
    dp_grid = np.linspace(-10, 10, num=1000)

    # Use Gauss-Hermite quadrature for efficient integration
    N_points = 50  
    gh_x, gh_w = hermgauss(N_points)

    # Adjust nodes and weights for standard normal distribution
    Z_samples = gh_x * np.sqrt(2)
    weights = gh_w / np.sqrt(np.pi)

    # Compute the cumulative distribution function values
    cdf_values = norm.cdf(Z_samples[:, np.newaxis] + dp_grid[np.newaxis, :])

    # Compute Pc_dp using weighted sum (vectorized over dp_grid)
    Pc_dp = np.dot(weights, cdf_values ** (m - 1))

    # Check for monotonicity and sort if necessary
    if not (np.all(np.diff(Pc_dp) >= 0) or np.all(np.diff(Pc_dp) <= 0)):
        sort_idx = np.argsort(Pc_dp)
        Pc_dp = Pc_dp[sort_idx]
        dp_grid = dp_grid[sort_idx]

    # Create interpolation function to map Pc to dp
    interp_dp = interp1d(Pc_dp, dp_grid, bounds_error=False, fill_value='extrapolate')

    # Compute dp values for given Pc values
    dp_values = interp_dp(Pc)

    return dp_values