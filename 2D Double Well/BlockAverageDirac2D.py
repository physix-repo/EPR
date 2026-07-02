import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
from utils import EPR_optimized_knn_histogram
import time
import warnings
import os
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. Core Physics & Theoretical Expectations
# ==========================================

def simulate_2d_double_well_dirac(A, K, D, dt, nsteps, ntraj):
    """
    Generates overdamped Langevin trajectories in a 2D landscape from a Dirac start (0,0).
    X: Quartic Double Well | Y: Harmonic Well
    """
    X = np.zeros((ntraj, nsteps))
    Y = np.zeros((ntraj, nsteps))
    
    sqrt_2Ddt = np.sqrt(2.0 * D * dt)
    
    for t in range(1, nsteps):
        fx = -4.0 * A * X[:, t-1] * (X[:, t-1]**2 - 1.0)
        fy = -K * Y[:, t-1]
        
        noise_x = np.random.normal(0, sqrt_2Ddt, ntraj)
        noise_y = np.random.normal(0, sqrt_2Ddt, ntraj)
        
        X[:, t] = X[:, t-1] + D * fx * dt + noise_x
        Y[:, t] = Y[:, t-1] + D * fy * dt + noise_y
        
    return X, Y

def exact_theoretical_entropies(A, K, D, dt):
    """Computes the exact thermodynamic free energy differences (Delta F) for the 2D Dirac start."""
    # 1. X-Dimension (Double Well Relaxation)
    S_harm_x = A + np.log(1.0 / np.sqrt(4.0 * A * D * dt))
    anharmonic_x = 3.0 / (16.0 * A)
    Exact_Delta_F_X = S_harm_x + anharmonic_x
    
    # 2. Y-Dimension (Harmonic Expansion from Dirac spike)
    rho_start_y = 1.0 / np.sqrt(4.0 * np.pi * D * dt)
    rho_eq_y = np.sqrt(K / (2.0 * np.pi * D))
    Exact_Delta_F_Y = np.log(rho_start_y / rho_eq_y)
    
    # 3. Full 2D Universe
    Exact_Delta_F_2D = Exact_Delta_F_X + Exact_Delta_F_Y
    
    return Exact_Delta_F_X, Exact_Delta_F_Y, Exact_Delta_F_2D

def get_exact_analytical_Stot_2D(A, K, D, dt):
    """
    Computes the absolute mathematical truth for the free energy relaxation 
    along X, Y, and the full 2D space from a Dirac start (evaluated at t=dt).
    """
    sigma_start = np.sqrt(2.0 * D * dt)
    
    # --- 1. X-Axis (Quartic Double Well) ---
    def boltzmann_x(x): 
        return np.exp(-A * (x**2 - 1.0)**2 / D)
    
    Z_X, _ = integrate.quad(boltzmann_x, -np.inf, np.inf)
    V_mean_start_X = A * (3.0*sigma_start**4 - 2.0*sigma_start**2 + 1.0)
    S_start_mean_X = -0.5 * np.log(2.0 * np.pi * np.e * sigma_start**2)
    Analytical_Stot_X = (V_mean_start_X/D + S_start_mean_X) - (-np.log(Z_X))
    
    # --- 2. Y-Axis (Harmonic Well) ---
    # The exact analytical partition function for a harmonic well is known
    Z_Y = np.sqrt(2.0 * np.pi * D / K)
    V_mean_start_Y = 0.5 * K * sigma_start**2
    S_start_mean_Y = -0.5 * np.log(2.0 * np.pi * np.e * sigma_start**2)
    Analytical_Stot_Y = (V_mean_start_Y/D + S_start_mean_Y) - (-np.log(Z_Y))
    
    # --- 3. Full 2D Space ---
    Analytical_Stot_2D = Analytical_Stot_X + Analytical_Stot_Y
    
    return Analytical_Stot_X, Analytical_Stot_Y, Analytical_Stot_2D

# ==========================================
# 2. Thermodynamic Estimators
# ==========================================

def EPR_optimized(Z, dt, bins=100, stridet=1):
    """Computes the 1D Entropy Production Rate for the projected coordinate using grids."""
    ntraj, nsteps, _ = Z.shape
    xmin, xmax = np.min(Z[:, :, 1]), np.max(Z[:, :, 1])
    dx = (xmax - xmin) / bins
    
    displacements = Z[:, 1:, 1] - Z[:, :-1, 1]
    starts = Z[:, :-1, 1]
    
    ix_all = np.clip(np.floor((starts - xmin) / dx).astype(int), 0, bins - 1).ravel()
    disp_flat = displacements.ravel()
    
    counts = np.bincount(ix_all, minlength=bins)
    mask_D = counts > 1 
    
    sum_disp = np.bincount(ix_all, weights=disp_flat, minlength=bins)
    sum_disp_sq = np.bincount(ix_all, weights=disp_flat**2, minlength=bins)
    
    mean_disp = np.zeros(bins)
    mean_disp_sq = np.zeros(bins)
    
    mean_disp[mask_D] = sum_disp[mask_D] / counts[mask_D]
    mean_disp_sq[mask_D] = sum_disp_sq[mask_D] / counts[mask_D]
    
    var_disp = mean_disp_sq - mean_disp**2
    D_array = np.zeros(bins)
    D_array[mask_D] = var_disp[mask_D] / (2.0 * dt)
    
    if np.any(~mask_D):
        global_D = np.var(disp_flat) / (2.0 * dt)
        D_array[~mask_D] = global_D
            
    it_indices = np.arange(stridet, nsteps - 1, stridet)
    Sdots = np.zeros(len(it_indices))
    taus = np.zeros(len(it_indices))

    for i, it in enumerate(it_indices):
        z_curr = Z[:, it, 1]
        v_curr = (Z[:, it + 1, 1] - Z[:, it - 1, 1]) / (2.0 * dt)
        ix = np.clip(np.floor((z_curr - xmin) / dx).astype(int), 0, bins - 1)
        
        rho_counts = np.bincount(ix, minlength=bins)
        vel_sum = np.bincount(ix, weights=v_curr, minlength=bins)
        
        mask = rho_counts > 0
        vel_mean = np.zeros(bins)
        vel_mean[mask] = vel_sum[mask] / rho_counts[mask]
        
        rho_prob = rho_counts / ntraj
        Sdots[i] = np.sum(rho_prob[mask] * (vel_mean[mask] ** 2) / D_array[mask])
        taus[i] = it * dt

    return Sdots, taus

def EPR_optimized_unbiased(Z, dt, bins=50, stridet=1):
    """Computes the 1D EPR using the Split-Sample Unbiased Estimator."""
    ntraj, nsteps, _ = Z.shape
    xmin, xmax = np.min(Z[:, :, 1]), np.max(Z[:, :, 1])
    dx = (xmax - xmin) / bins
    
    # 1. Compute Local Diffusion D(Z) using the full ensemble
    displacements = Z[:, 1:, 1] - Z[:, :-1, 1]
    starts = Z[:, :-1, 1]
    ix_all = np.clip(np.floor((starts - xmin) / dx).astype(int), 0, bins - 1).ravel()
    disp_flat = displacements.ravel()
    
    counts = np.bincount(ix_all, minlength=bins)
    mask_D = counts > 1 
    
    sum_disp = np.bincount(ix_all, weights=disp_flat, minlength=bins)
    sum_disp_sq = np.bincount(ix_all, weights=disp_flat**2, minlength=bins)
    
    mean_disp = np.zeros(bins)
    mean_disp_sq = np.zeros(bins)
    mean_disp[mask_D] = sum_disp[mask_D] / counts[mask_D]
    mean_disp_sq[mask_D] = sum_disp_sq[mask_D] / counts[mask_D]
    
    var_disp = mean_disp_sq - mean_disp**2
    D_array = np.zeros(bins)
    D_array[mask_D] = var_disp[mask_D] / (2.0 * dt)
    
    if np.any(~mask_D):
        D_array[~mask_D] = np.var(disp_flat) / (2.0 * dt)
            
    # 2. Compute Unbiased Velocity Flow
    it_indices = np.arange(stridet, nsteps - 1, stridet)
    Sdots = np.zeros(len(it_indices))
    taus = np.zeros(len(it_indices))

    for i, it in enumerate(it_indices):
        z_curr = Z[:, it, 1]
        v_curr = (Z[:, it + 1, 1] - Z[:, it - 1, 1]) / (2.0 * dt)
        ix = np.clip(np.floor((z_curr - xmin) / dx).astype(int), 0, bins - 1)
        
        # Split-Sample Fix
        is_A = np.random.rand(ntraj) > 0.5
        is_B = ~is_A
        
        counts_A = np.bincount(ix[is_A], minlength=bins)
        counts_B = np.bincount(ix[is_B], minlength=bins)
        
        vel_sum_A = np.bincount(ix[is_A], weights=v_curr[is_A], minlength=bins)
        vel_sum_B = np.bincount(ix[is_B], weights=v_curr[is_B], minlength=bins)
        
        mask_AB = (counts_A > 0) & (counts_B > 0)
        
        vel_mean_A = np.zeros(bins)
        vel_mean_B = np.zeros(bins)
        vel_mean_A[mask_AB] = vel_sum_A[mask_AB] / counts_A[mask_AB]
        vel_mean_B[mask_AB] = vel_sum_B[mask_AB] / counts_B[mask_AB]
        
        rho_prob = (counts_A + counts_B) / ntraj
        Sdots[i] = np.sum(rho_prob[mask_AB] * (vel_mean_A[mask_AB] * vel_mean_B[mask_AB]) / D_array[mask_AB])
        taus[i] = it * dt
        
    return Sdots, taus


def EPR_2D_meshless(X, Y, D_global, dt, k_neighbors=50, stridet=1):
    """Computes the full 2D Entropy Production Rate using the Meshless KNN approach."""
    ntraj, nsteps = X.shape
    it_indices = np.arange(stridet, nsteps - 1, stridet)
    Sdots = np.zeros(len(it_indices))
    taus = np.zeros(len(it_indices))
    
    for i, it in enumerate(it_indices):
        x_curr = X[:, it]
        y_curr = Y[:, it]
        positions = np.column_stack((x_curr, y_curr))
        
        vx_curr = (X[:, it + 1] - X[:, it - 1]) / (2.0 * dt)
        vy_curr = (Y[:, it + 1] - Y[:, it - 1]) / (2.0 * dt)
        velocities = np.column_stack((vx_curr, vy_curr))
        
        tree = cKDTree(positions)
        distances, indices = tree.query(positions, k=k_neighbors, workers=-1)
        
        neighbor_vels = velocities[indices]
        v_mean = np.mean(neighbor_vels, axis=1) 
        v_sq_sum = np.sum(v_mean**2, axis=1)
        Sdots[i] = np.mean(v_sq_sum / D_global)
        taus[i] = it * dt
        
    return Sdots, taus

def EPR_2D_meshless_spatial_D(X, Y, dt, k_neighbors=50, alpha_prior=20.0, stridet=1):
    """
    Computes 2D EPR using Split-Sample velocity AND a Bayesian Shrinkage 
    estimator for the local spatial diffusion profile D(x,y).
    """
    ntraj, nsteps = X.shape
    
    # 1. Establish the Global Prior
    dx_all = X[:, 1:] - X[:, :-1]
    dy_all = Y[:, 1:] - Y[:, :-1]
    D_global = 0.5 * (np.var(dx_all) + np.var(dy_all)) / (2.0 * dt)
    
    it_indices = np.arange(stridet, nsteps - 1, stridet)
    Sdots, taus = np.zeros(len(it_indices)), np.zeros(len(it_indices))
    k_half = k_neighbors // 2

    for i, it in enumerate(it_indices):
        positions = np.column_stack((X[:, it], Y[:, it]))
        
        # Velocity components (central difference)
        vx_curr = (X[:, it + 1] - X[:, it - 1]) / (2.0 * dt)
        vy_curr = (Y[:, it + 1] - Y[:, it - 1]) / (2.0 * dt)
        velocities = np.column_stack((vx_curr, vy_curr))
        
        # Diffusion components (forward difference to isolate noise variance)
        dx_fwd = X[:, it + 1] - X[:, it]
        dy_fwd = Y[:, it + 1] - Y[:, it]
        
        tree = cKDTree(positions)
        _, indices = tree.query(positions, k=k_neighbors, workers=-1)
        
        # --- A. Unbiased Split-Sample Velocity ---
        indices_A = indices[:, :k_half]
        indices_B = indices[:, k_half:]
        v_mean_A = np.mean(velocities[indices_A], axis=1) 
        v_mean_B = np.mean(velocities[indices_B], axis=1)
        v_sq_unbiased = np.sum(v_mean_A * v_mean_B, axis=1)
        
        # --- B. Local Spatial Diffusion Profile D(x,y) ---
        dx_neigh = dx_fwd[indices]
        dy_neigh = dy_fwd[indices]
        
        var_x = np.var(dx_neigh, axis=1)
        var_y = np.var(dy_neigh, axis=1)
        D_local = (var_x + var_y) / (4.0 * dt)
        
        # Bayesian Shrinkage: Blends local variance with global prior 
        # to prevent division-by-zero explosions in sparse regions
        D_robust = (k_neighbors * D_local + alpha_prior * D_global) / (k_neighbors + alpha_prior)
        
        # Compute local entropy production
        Sdots[i] = np.mean(v_sq_unbiased / D_robust)
        taus[i] = it * dt
        
    return Sdots, taus

def epr_plain(Z, dt, bins=None, stridet=1, D_grid_bins=80, min_count_D=10, seed=0):
    """
    Time-resolved EPR estimator.
 
    Parameters
    ----------
    Z : array, shape (ntraj, nsteps)
        1D trajectories sampled at uniform step dt.
    dt : float
        Sampling interval.
    bins : int or None
        Number of quantile bins per time slice. If None, ntraj//40 clipped
        to [8, 50].
    stridet : int
        Compute Sdot every `stridet` time steps (1 = every step).
    D_grid_bins : int
        Grid resolution for the pooled D(x).
    min_count_D : int
        Bins with fewer samples fall back to the global D.
    seed : int
        For the A/B trajectory split.
 
    Returns
    -------
    taus : (T,) array of times
    Sdots : (T,) array of S_dot(t)
    """
    rng = np.random.default_rng(seed)
    ntraj, nsteps = Z.shape
    if bins is None:
        bins = int(np.clip(ntraj // 40, 8, 50))
 
    # ------------------------------------------------------------------
    # 1. Pooled D(x) on a fixed fine grid (over all traj * all times)
    # ------------------------------------------------------------------
    starts = Z[:, :-1].ravel()
    disp   = (Z[:, 1:] - Z[:, :-1]).ravel()
 
    xmin_D, xmax_D = starts.min(), starts.max()
    pad = 1e-8 * (xmax_D - xmin_D + 1.0)
    xmin_D -= pad; xmax_D += pad
    dx_D = (xmax_D - xmin_D) / D_grid_bins
    D_centers = xmin_D + (np.arange(D_grid_bins) + 0.5) * dx_D
 
    ix = np.clip(((starts - xmin_D) / dx_D).astype(int), 0, D_grid_bins - 1)
    counts_D = np.bincount(ix, minlength=D_grid_bins).astype(float)
    sum_d  = np.bincount(ix, weights=disp,    minlength=D_grid_bins)
    sum_d2 = np.bincount(ix, weights=disp**2, minlength=D_grid_bins)
 
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_d  = np.where(counts_D > 0, sum_d  / counts_D, 0.0)
        mean_d2 = np.where(counts_D > 0, sum_d2 / counts_D, 0.0)
    D_grid = (mean_d2 - mean_d**2) / (2.0 * dt)
 
    global_D = np.var(disp) / (2.0 * dt)
    D_grid = np.where(counts_D >= min_count_D, D_grid, global_D)
    D_grid = np.clip(D_grid, 1e-12, None)
 
    # ------------------------------------------------------------------
    # 2. Fixed A/B trajectory split (unbiased v_drift^2)
    # ------------------------------------------------------------------
    perm = rng.permutation(ntraj)
    halfA = perm[: ntraj // 2]
    halfB = perm[ntraj // 2 : 2 * (ntraj // 2)]
 
    # ------------------------------------------------------------------
    # 3. S_dot(t) at each time slice, with quantile bins per slice
    # ------------------------------------------------------------------
    it_indices = np.arange(stridet, nsteps - 1, stridet)
    taus = it_indices * dt
    Sdots = np.zeros(len(it_indices))
 
    for i, it in enumerate(it_indices):
        x = Z[:, it]
        v = (Z[:, it + 1] - Z[:, it - 1]) / (2.0 * dt)   # centered velocity
 
        # Skip degenerate slices (cloud too narrow)
        if x.std() < 1e-8:
            continue
 
        # Quantile bin edges from combined ensemble
        edges_t = np.quantile(x, np.linspace(0, 1, bins + 1))
        edges_t = np.maximum.accumulate(edges_t)
        edges_t[-1] += 1e-12 * (abs(edges_t[-1]) + 1.0)
        if edges_t[-1] - edges_t[0] < 1e-12:
            continue
 
        # Bin assignment for each half
        ix_A = np.clip(np.searchsorted(edges_t, x[halfA], side="right") - 1,
                       0, bins - 1)
        ix_B = np.clip(np.searchsorted(edges_t, x[halfB], side="right") - 1,
                       0, bins - 1)
 
        # Conditional v_mean per half
        cnt_A = np.bincount(ix_A, minlength=bins).astype(float)
        cnt_B = np.bincount(ix_B, minlength=bins).astype(float)
        sum_vA = np.bincount(ix_A, weights=v[halfA], minlength=bins)
        sum_vB = np.bincount(ix_B, weights=v[halfB], minlength=bins)
 
        mask = (cnt_A > 0) & (cnt_B > 0)
        if not mask.any():
            continue
 
        vA_bar = np.zeros(bins); vA_bar[mask] = sum_vA[mask] / cnt_A[mask]
        vB_bar = np.zeros(bins); vB_bar[mask] = sum_vB[mask] / cnt_B[mask]
 
        # D(x) at the bin centers, by linear interpolation of the pooled grid
        bin_centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])
        D_at_bins = np.interp(bin_centers_t, D_centers, D_grid,
                              left=global_D, right=global_D)
 
        # Empirical P(bin) -- this implicitly contains the dx factor
        rho_prob = (cnt_A + cnt_B) / (cnt_A.sum() + cnt_B.sum())
 
        Sdots[i] = np.sum(
            rho_prob[mask] * vA_bar[mask] * vB_bar[mask] / D_at_bins[mask]
        )
 
    return Sdots, taus

def compute_thermodynamics_2d(X_b, Y_b, A, K, D, dt):
    """Computes empirical Stot = Q + dS from the 2D trajectory endpoints via KDE."""
    X_start, Y_start = X_b[:, 1], Y_b[:, 1]
    X_end, Y_end = X_b[:, -1], Y_b[:, -1]
    
    sigma_start = np.sqrt(2.0 * D * dt)
    
    # EXACT Initial Entropy (Evaluated at t=dt for a 2D Isotropic Gaussian)
    ln_rho_start = -np.log(2.0 * np.pi * sigma_start**2) - (X_start**2 + Y_start**2) / (2.0 * sigma_start**2)
    
    try:
        kde_end = gaussian_kde(np.vstack([X_end, Y_end]))
        ln_rho_end = np.log(kde_end(np.vstack([X_end, Y_end])) + 1e-12)
    except Exception:
        ln_rho_end = np.zeros_like(X_end)
        
    Delta_S = ln_rho_start - ln_rho_end
    
    V_start = A * (X_start**2 - 1.0)**2 + 0.5 * K * Y_start**2
    V_end = A * (X_end**2 - 1.0)**2 + 0.5 * K * Y_end**2
    Q = (V_start - V_end) / D
    
    return np.mean(Q + Delta_S)

def compute_thermodynamics_1d_X(X_b, A, D, dt):
    """Computes empirical Stot along the X-axis using 1D KDE."""
    X_start, X_end = X_b[:, 1], X_b[:, -1]
    sigma_start = np.sqrt(2.0 * D * dt)
    
    ln_rho_start = -0.5 * np.log(2.0 * np.pi * sigma_start**2) - (X_start**2) / (2.0 * sigma_start**2)
    try:
        kde_end = gaussian_kde(X_end)
        ln_rho_end = np.log(kde_end(X_end) + 1e-12)
    except Exception:
        ln_rho_end = np.zeros_like(X_end)
        
    Delta_S = ln_rho_start - ln_rho_end
    V_start = A * (X_start**2 - 1.0)**2
    V_end = A * (X_end**2 - 1.0)**2
    Q = (V_start - V_end) / D
    
    return np.mean(Q + Delta_S)

def compute_thermodynamics_1d_Y(Y_b, K, D, dt):
    """Computes empirical Stot along the Y-axis using 1D KDE."""
    Y_start, Y_end = Y_b[:, 1], Y_b[:, -1]
    sigma_start = np.sqrt(2.0 * D * dt)
    
    ln_rho_start = -0.5 * np.log(2.0 * np.pi * sigma_start**2) - (Y_start**2) / (2.0 * sigma_start**2)
    try:
        kde_end = gaussian_kde(Y_end)
        ln_rho_end = np.log(kde_end(Y_end) + 1e-12)
    except Exception:
        ln_rho_end = np.zeros_like(Y_end)
        
    Delta_S = ln_rho_start - ln_rho_end
    V_start = 0.5 * K * Y_start**2
    V_end = 0.5 * K * Y_end**2
    Q = (V_start - V_end) / D
    
    return np.mean(Q + Delta_S)

def compute_thermodynamics_2d_GMM(X_b, Y_b, A, K, D, dt):
    """Computes empirical Stot = Q + dS from the 2D trajectory endpoints via KDE."""
    X_start, Y_start = X_b[:, 1], Y_b[:, 1]
    X_end, Y_end = X_b[:, -1], Y_b[:, -1]
    
    sigma_start = np.sqrt(2.0 * D * dt)
    
    # EXACT Initial Entropy (Evaluated at t=dt for a 2D Isotropic Gaussian)
    ln_rho_start = -np.log(2.0 * np.pi * sigma_start**2) - (X_start**2 + Y_start**2) / (2.0 * sigma_start**2)
    
    data = np.column_stack((X_end, Y_end))
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(data)
    ln_rho_end_gmm = gmm.score_samples(data)
        
    Delta_S = ln_rho_start - ln_rho_end_gmm
    
    V_start = A * (X_start**2 - 1.0)**2 + 0.5 * K * Y_start**2
    V_end = A * (X_end**2 - 1.0)**2 + 0.5 * K * Y_end**2
    Q = (V_start - V_end) / D
    
    return np.mean(Q + Delta_S)

def get_2d_ln_rho_gmm(X, Y):
    data = np.column_stack((X, Y))
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(data)
    return gmm.score_samples(data)


# ==========================================
# 3. Execution & Block Averaging
# ==========================================

if __name__ == "__main__":
    method = 'biased'

    # Base Parameters
    A = 8.0       
    K = 40.0       
    D = 1.0       
    dt = 0.001    
    nsteps = 1000
    ntraj = 2000
    bins_1d = 50
    k_neighbors = 40  
    n_blocks = 20
    
    angles_deg = np.linspace(0, 90, 6)
    angles_rad = np.radians(angles_deg)
    
    Exact_Delta_F_X, Exact_Delta_F_Y, Exact_Delta_F_2D = exact_theoretical_entropies(A, K, D, dt)
    Analytical_Stot_X, Analytical_Stot_Y, Analytical_Stot_2D = get_exact_analytical_Stot_2D(A, K, D, dt)
    
    print(f"--- Simulating 2D Free Energy Relaxation (A={A}, K={K}) ---")
    t0 = time.time()
    
    # ----------------------------------------------------
    # 0. Simulate Master Ensemble
    # ----------------------------------------------------
    print(f"Generating Master Ensemble ({ntraj} trajectories)...")
    X_master, Y_master = simulate_2d_double_well_dirac(A, K, D, dt, nsteps, ntraj)

    stride = 1
    X_master = X_master[:, ::stride]
    Y_master = Y_master[:, ::stride]
    dt = dt*stride
    
    # ----------------------------------------------------
    # 1. Block Averaging Engine
    # ----------------------------------------------------
    print(f"Computing Block-Averaged Thermodynamics ({n_blocks} blocks of {ntraj//n_blocks} trajectories)...")
    X_blocks = np.array_split(X_master, n_blocks, axis=0)
    Y_blocks = np.array_split(Y_master, n_blocks, axis=0)
    
    # Data Storage
    Stot_EPR_2D_blocks = []
    Stot_Thermo_2D_blocks = []
    Stot_Thermo_X_blocks = []
    Stot_Thermo_Y_blocks = []
    Stot_EPR_1D_blocks = {deg: [] for deg in angles_deg}
    Sdots_EPR = {deg: [] for deg in angles_deg}
    
    for b in range(n_blocks):
        print(f"  Processing Block {b+1}/{n_blocks}...")
        X_b = X_blocks[b]
        Y_b = Y_blocks[b]
        ntraj_b = X_b.shape[0]
        
        # --- A. Full 2D EPR (Meshless KNN) ---
        Sdots_2d, taus_2d = EPR_2D_meshless_spatial_D(X_b, Y_b, dt, k_neighbors=int(np.sqrt(ntraj//n_blocks)), stridet=1)
        baseline_2d = 0#np.mean(Sdots_2d[-100:])
        Stot_EPR_2D_blocks.append(np.trapezoid(Sdots_2d - baseline_2d, taus_2d))
        
        # --- B. Full 2D Thermo Thermodynamics (KDE) ---
        thermo_stot_2d = compute_thermodynamics_2d_GMM(X_b, Y_b, A, K, D, dt)
        Stot_Thermo_2D_blocks.append(thermo_stot_2d)
        
        # --- C. Projected 1D Thermo Thermodynamics (X and Y axis) ---
        thermo_stot_x = compute_thermodynamics_1d_X(X_b, A, D, dt)
        Stot_Thermo_X_blocks.append(thermo_stot_x)
        
        thermo_stot_y = compute_thermodynamics_1d_Y(Y_b, K, D, dt)
        Stot_Thermo_Y_blocks.append(thermo_stot_y)
        
        # --- D. Projected 1D Angle Sweep (Grid-based EPR) ---
        for deg, rad in zip(angles_deg, angles_rad):
            Z_1d_pos = X_b * np.cos(rad) + Y_b * np.sin(rad)
            
            Z_1d_formatted = np.zeros((ntraj_b, int(nsteps/stride), 2))
            Z_1d_formatted[:, :, 0] = np.arange(int(nsteps/stride)) * dt
            Z_1d_formatted[:, :, 1] = Z_1d_pos
            
            if method == 'biased':
                Sdots_1d, taus_1d = EPR_optimized_knn_histogram(Z_1d_formatted, dt, bins=int(np.sqrt(ntraj//n_blocks)), stridet=1)
                #Sdots_1d, taus_1d = epr_plain(Z_1d_formatted[:, :, 1], dt)
                baseline_1d = np.mean(Sdots_1d[-100:])
                Stot_EPR_1D_blocks[deg].append(np.trapezoid(Sdots_1d - baseline_1d, taus_1d))
                Sdots_EPR[deg].append(Sdots_1d)
            else:
                Sdots_1d, taus_1d = EPR_optimized_unbiased(Z_1d_formatted, dt, bins=int(np.sqrt(ntraj//n_blocks)), stridet=1)
                baseline_1d = 0
                Stot_EPR_1D_blocks[deg].append(np.trapezoid(Sdots_1d - baseline_1d, taus_1d))
                Sdots_EPR[deg].append(Sdots_1d)
            
    # Calculate Statistical Means and Errors
    Integrated_Stot_2D_Mean = np.mean(Stot_EPR_2D_blocks)
    Integrated_Stot_2D_Err = np.std(Stot_EPR_2D_blocks) / np.sqrt(n_blocks)
    
    Thermo_Stot_2D_Mean = np.mean(Stot_Thermo_2D_blocks)
    Thermo_Stot_2D_Err = np.std(Stot_Thermo_2D_blocks) / np.sqrt(n_blocks)
    
    Thermo_Stot_X_Mean = np.mean(Stot_Thermo_X_blocks)
    Thermo_Stot_X_Err = np.std(Stot_Thermo_X_blocks) / np.sqrt(n_blocks)
    
    Thermo_Stot_Y_Mean = np.mean(Stot_Thermo_Y_blocks)
    Thermo_Stot_Y_Err = np.std(Stot_Thermo_Y_blocks) / np.sqrt(n_blocks)
    
    Integrated_Stot_1D_Means = [np.mean(Stot_EPR_1D_blocks[deg]) for deg in angles_deg]
    Integrated_Stot_1D_Errs = [np.std(Stot_EPR_1D_blocks[deg]) / np.sqrt(n_blocks) for deg in angles_deg]

    Sdots = [Sdots_EPR[deg] for deg in angles_deg]

    print(f"\nTime: {time.time()-t0:.1f}s")
    print("\n--- 2D Thermodynamic Trifecta Ledger ---")
    print(f"  Exact Analytical DF (X):     {Exact_Delta_F_X:.4f} kT")
    print(f"  Exact Analytical Stot (X):     {Analytical_Stot_X:.4f} kT")
    print(f"  Empirical Thermo Stot (X): {Thermo_Stot_X_Mean:.4f} +/- {Thermo_Stot_X_Err:.4f} kT")
    print(f"  Exact Analytical DF (Y):     {Exact_Delta_F_Y:.4f} kT")
    print(f"  Exact Analytical Stot (Y):     {Analytical_Stot_Y:.4f} kT")
    print(f"  Empirical Thermo Stot (Y): {Thermo_Stot_Y_Mean:.4f} +/- {Thermo_Stot_Y_Err:.4f} kT")
    print("-----------------------------------")
    print(f"  1. Exact Analytical DF (2D): {Exact_Delta_F_2D:.4f} kT")
    print(f"  2. Empirical Thermo Stot:  {Thermo_Stot_2D_Mean:.4f} +/- {Thermo_Stot_2D_Err:.4f} kT")
    print(f"  3. Meshless EPR Stot (2D):   {Integrated_Stot_2D_Mean:.4f} +/- {Integrated_Stot_2D_Err:.4f} kT")

    # ==========================================
    # 4. Save Data
    # ==========================================
    if method == 'biased':
        save_filename = 'quartic_results_{}k_{}_blocks_GMM_knn_squared_biased.npz'.format(int(ntraj//1000), n_blocks)
        #save_filename = 'test.npz'
    else:
        save_filename = 'trifecta_2D_meshless_results_5k_5_blocks_dt_0.01_GMM_unbiased_full.npz'
    print(f"\nSaving all results to '{save_filename}'...")
    
    np.savez_compressed(
        save_filename,
        A=A, K=K, D=D, dt=dt, nsteps=nsteps, ntraj=ntraj, n_blocks=n_blocks,
        k_neighbors=k_neighbors, bins_1d=bins_1d, angles_deg=angles_deg,
        Analytical_Stot_X=Analytical_Stot_X, Analytical_Stot_Y=Analytical_Stot_Y, Analytical_Stot_2D=Analytical_Stot_2D,
        Exact_Delta_F_X=Exact_Delta_F_X, Exact_Delta_F_Y=Exact_Delta_F_Y, Exact_Delta_F_2D=Exact_Delta_F_2D,
        Stot_EPR_2D_blocks=Stot_EPR_2D_blocks, Integrated_Stot_2D_Mean=Integrated_Stot_2D_Mean, Integrated_Stot_2D_Err=Integrated_Stot_2D_Err,
        Stot_Thermo_2D_blocks=Stot_Thermo_2D_blocks, Thermo_Stot_2D_Mean=Thermo_Stot_2D_Mean, Thermo_Stot_2D_Err=Thermo_Stot_2D_Err,
        Thermo_Stot_X_Mean=Thermo_Stot_X_Mean, Thermo_Stot_X_Err=Thermo_Stot_X_Err,
        Thermo_Stot_Y_Mean=Thermo_Stot_Y_Mean, Thermo_Stot_Y_Err=Thermo_Stot_Y_Err,
        Integrated_Stot_1D_Means=Integrated_Stot_1D_Means, Integrated_Stot_1D_Errs=Integrated_Stot_1D_Errs, Sdots=Sdots
    )
    print("Data successfully saved!")

    # ==========================================
    # 5. Plotting
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the full 2D Trifecta
    ax.axhline(Analytical_Stot_2D, color='black', linestyle='-', linewidth=2, label=r'Exact $\Delta F$ (2D Limit)')
    
    # 2D Thermo KDE
    ax.axhline(Thermo_Stot_2D_Mean, color='tab:green', linestyle='--', linewidth=2, label=r'Empirical Thermo $S_{tot}$ (2D)')
    ax.fill_between(np.linspace(-5, 105, 10), 
                    Thermo_Stot_2D_Mean - Thermo_Stot_2D_Err, 
                    Thermo_Stot_2D_Mean + Thermo_Stot_2D_Err, 
                    color='tab:green', alpha=0.15)

    # 2D Meshless EPR
    ax.axhline(Integrated_Stot_2D_Mean, color='red', linestyle='-.', linewidth=2, label=r'Meshless EPR $S_{tot}$ (2D)')
    ax.fill_between(np.linspace(-5, 105, 10), 
                    Integrated_Stot_2D_Mean - Integrated_Stot_2D_Err, 
                    Integrated_Stot_2D_Mean + Integrated_Stot_2D_Err, 
                    color='red', alpha=0.15)
    
    # Plot the 1D X theoretical and empirical limits
    ax.axhline(Analytical_Stot_X, color='blue', linestyle=':', alpha=0.7, label=r'Exact $\Delta F$ (X-Limit)')
    #ax.axhline(Thermo_Stot_X_Mean, color='blue', linestyle='--', alpha=0.4, label=r'Empirical Thermo $S_{tot}$ (X)')
    
    # Plot the 1D Y theoretical and empirical limits
    ax.axhline(Analytical_Stot_Y, color='tab:orange', linestyle=':', alpha=0.7, label=r'Exact $\Delta F$ (Y-Limit)')
    #ax.axhline(Thermo_Stot_Y_Mean, color='tab:orange', linestyle='--', alpha=0.4, label=r'Empirical Thermo $S_{tot}$ (Y)')
    
    # Plot the measured 1D projection data
    ax.errorbar(angles_deg, Integrated_Stot_1D_Means, yerr=Integrated_Stot_1D_Errs, fmt='o-', capsize=5, 
                 color='purple', markersize=8, linewidth=2, label=r'Apparent $S_{tot}$ (1D Grid Projection)')
    
    ax.set_title('Thermodynamic Trifecta: 1D Projection vs Full 2D Entropy limits', fontsize=14, pad=15)
    ax.set_xlabel(r'Observation Angle $\theta$ (Degrees)', fontsize=12)
    ax.set_ylabel(r'Entropy Production ($k_B T$)', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Organize legend columns to keep it clean
    ax.legend(loc='center right', fontsize=9, ncol=1)
    ax.set_xlim(-5, 95)
    
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.05)
    
    plt.tight_layout()
    plt.show()