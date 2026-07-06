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
    S_harm_x = A + np.log(1.0 / np.sqrt(4.0 * A * D * dt))
    anharmonic_x = 3.0 / (16.0 * A)
    Exact_Delta_F_X = S_harm_x + anharmonic_x
    
    rho_start_y = 1.0 / np.sqrt(4.0 * np.pi * D * dt)
    rho_eq_y = np.sqrt(K / (2.0 * np.pi * D))
    Exact_Delta_F_Y = np.log(rho_start_y / rho_eq_y)
    
    Exact_Delta_F_2D = Exact_Delta_F_X + Exact_Delta_F_Y
    
    return Exact_Delta_F_X, Exact_Delta_F_Y, Exact_Delta_F_2D

def get_exact_analytical_Stot_2D(A, K, D, dt):
    sigma_start = np.sqrt(2.0 * D * dt)
    
    def boltzmann_x(x): 
        return np.exp(-A * (x**2 - 1.0)**2 / D)
    
    Z_X, _ = integrate.quad(boltzmann_x, -np.inf, np.inf)
    V_mean_start_X = A * (3.0*sigma_start**4 - 2.0*sigma_start**2 + 1.0)
    S_start_mean_X = -0.5 * np.log(2.0 * np.pi * np.e * sigma_start**2)
    Analytical_Stot_X = (V_mean_start_X/D + S_start_mean_X) - (-np.log(Z_X))
    
    Z_Y = np.sqrt(2.0 * np.pi * D / K)
    V_mean_start_Y = 0.5 * K * sigma_start**2
    S_start_mean_Y = -0.5 * np.log(2.0 * np.pi * np.e * sigma_start**2)
    Analytical_Stot_Y = (V_mean_start_Y/D + S_start_mean_Y) - (-np.log(Z_Y))
    
    Analytical_Stot_2D = Analytical_Stot_X + Analytical_Stot_Y
    
    return Analytical_Stot_X, Analytical_Stot_Y, Analytical_Stot_2D

# ==========================================
# 2. Thermodynamic Estimators
# ==========================================

def compute_thermodynamics_2d_GMM(X_b, Y_b, A, K, D, dt):
    X_start, Y_start = X_b[:, 1], Y_b[:, 1]
    X_end, Y_end = X_b[:, -1], Y_b[:, -1]
    sigma_start = np.sqrt(2.0 * D * dt)
    
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

def compute_thermodynamics_1d_X(X_b, A, D, dt):
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

def compute_thermodynamics_1d_projected_KM(Z_pos, dt):
    """
    Computes thermodynamic Stot using Kramers-Moyal for the PMF.
    """
    # 1. Build the exact PMF from the trajectory kinematics
    z_grid, pmf, mask = estimate_pmf_kramers_moyal(Z_pos, dt, bins=100)
    
    Z_start = Z_pos[:, 1]
    Z_end = Z_pos[:, -1]
    
    # 2. Map trajectory endpoints to the PMF
    xmin, xmax = np.min(Z_pos), np.max(Z_pos)
    dx = (xmax - xmin) / 100
    
    ix_start = np.clip(np.floor((Z_start - xmin) / dx).astype(int), 0, 99)
    ix_end = np.clip(np.floor((Z_end - xmin) / dx).astype(int), 0, 99)
    
    # Filter to only evaluate particles that land in valid PMF regions
    valid_endpoints = mask[ix_start] & mask[ix_end]
    
    # Apparent Heat: V_eff(start) - V_eff(end)
    Q_1D = pmf[ix_start[valid_endpoints]] - pmf[ix_end[valid_endpoints]]
    
    # System Entropy Change via standard KDE for P(x)
    try:
        kde_end = gaussian_kde(Z_end)
        ln_rho_end = np.log(kde_end(Z_end[valid_endpoints]) + 1e-12)
    except Exception:
        ln_rho_end = np.zeros(len(valid_endpoints))
        
    sigma_start = np.sqrt(2.0 * 1.0 * dt) # Assuming D=1.0 for the start variance
    ln_rho_start = -0.5 * np.log(2.0 * np.pi * sigma_start**2) - (Z_start[valid_endpoints]**2) / (2.0 * sigma_start**2)
    
    Delta_S_sys = ln_rho_start - ln_rho_end
    
    return np.mean(Q_1D + Delta_S_sys)

def estimate_pmf_kramers_moyal(Z_pos, dt, bins=100):
    """
    Estimates the Potential of Mean Force (PMF) strictly from trajectory kinematics
    using the Kramers-Moyal expansion. Works even if the system is out of equilibrium.
    
    Parameters:
      Z_pos : array of shape (ntraj, nsteps) containing 1D coordinates
      dt : time step
      
    Returns:
      bin_centers : spatial coordinates of the PMF
      pmf : The integrated potential of mean force
      mask : boolean array indicating valid spatial bins
    """
    ntraj, nsteps = Z_pos.shape
    xmin, xmax = np.min(Z_pos), np.max(Z_pos)
    dx = (xmax - xmin) / bins
    
    # 1. Forward Finite Differences (KM requires the future jump)
    displacements = Z_pos[:, 1:] - Z_pos[:, :-1]
    starts = Z_pos[:, :-1]
    
    ix_all = np.clip(np.floor((starts - xmin) / dx).astype(int), 0, bins - 1).ravel()
    disp_flat = displacements.ravel()
    
    # 2. Extract local jump statistics
    counts = np.bincount(ix_all, minlength=bins)
    mask = counts > 20  # Require minimal statistics for stable integration
    
    sum_disp = np.bincount(ix_all, weights=disp_flat, minlength=bins)
    sum_disp_sq = np.bincount(ix_all, weights=disp_flat**2, minlength=bins)
    
    D1 = np.zeros(bins)
    D2 = np.zeros(bins)
    
    # 3. First KM Coefficient (Drift)
    D1[mask] = (sum_disp[mask] / counts[mask]) / dt
    
    # 4. Second KM Coefficient (Diffusion)
    # Using variance rather than raw <x^2> removes finite-dt drift bias
    mean_disp = np.zeros(bins)
    mean_disp[mask] = sum_disp[mask] / counts[mask]
    
    mean_disp_sq = np.zeros(bins)
    mean_disp_sq[mask] = sum_disp_sq[mask] / counts[mask]
    
    var_disp = np.zeros(bins)
    var_disp[mask] = mean_disp_sq[mask] - mean_disp[mask]**2
    D2[mask] = var_disp[mask] / (2.0 * dt)
    
    # Safety fallback for empty bins
    if np.any(~mask):
        global_D2 = np.var(disp_flat) / (2.0 * dt)
        D2[~mask] = global_D2
        
    # 5. Compute the local force vector
    force = np.zeros(bins)
    force[mask] = D1[mask] / D2[mask]
    
    # 6. Cumulative Integration to construct the PMF
    pmf = np.zeros(bins)
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) > 0:
        start_idx = valid_indices[0]
        
        # Trapezoidal rule over the valid spatial grid
        for i in range(start_idx + 1, bins):
            if mask[i] and mask[i-1]:
                pmf[i] = pmf[i-1] - 0.5 * (force[i] + force[i-1]) * dx
            elif mask[i]:
                pmf[i] = pmf[i-1] - force[i] * dx
            else:
                pmf[i] = pmf[i-1] # Carry flat potential over voids
                
        # Shift the PMF so the global minimum rests at 0
        pmf -= np.min(pmf[mask])
        
    bin_centers = xmin + (np.arange(bins) + 0.5) * dx
    
    # Set invalid regions to NaN so they plot correctly
    pmf[~mask] = np.nan 
    
    return bin_centers, pmf, mask

# ==========================================
# 3. Execution & Block Averaging
# ==========================================

if __name__ == "__main__":
    method = 'biased'

    A = 8.0       
    K = 40.0       
    D = 1.0       
    dt = 0.001    
    nsteps = 1000
    ntraj = 5000
    bins_1d = 50
    k_neighbors = 40  
    n_blocks = 5
    
    angles_deg = np.linspace(0, 90, 6)
    angles_rad = np.radians(angles_deg)
    
    Exact_Delta_F_X, Exact_Delta_F_Y, Exact_Delta_F_2D = exact_theoretical_entropies(A, K, D, dt)
    Analytical_Stot_X, Analytical_Stot_Y, Analytical_Stot_2D = get_exact_analytical_Stot_2D(A, K, D, dt)
    
    print(f"--- Simulating 2D Free Energy Relaxation (A={A}, K={K}) ---")
    t0 = time.time()
    
    print(f"Generating Master Ensemble ({ntraj} trajectories)...")
    X_master, Y_master = simulate_2d_double_well_dirac(A, K, D, dt, nsteps, ntraj)

    stride = 1
    X_master = X_master[:, ::stride]
    Y_master = Y_master[:, ::stride]
    dt = dt * stride
    
    print(f"Computing Block-Averaged Thermodynamics ({n_blocks} blocks)...")
    X_blocks = np.array_split(X_master, n_blocks, axis=0)
    Y_blocks = np.array_split(Y_master, n_blocks, axis=0)
    
    # Data Storage
    Stot_EPR_2D_blocks = []
    Stot_Thermo_2D_blocks = []
    Stot_Thermo_X_blocks = []
    Stot_Thermo_Y_blocks = []
    
    Stot_EPR_1D_blocks = {deg: [] for deg in angles_deg}
    Stot_Thermo_1D_blocks = {deg: [] for deg in angles_deg}  # NEW DICT FOR PROJECTED THERMO
    Sdots_EPR = {deg: [] for deg in angles_deg}
    
    for b in range(n_blocks):
        X_b = X_blocks[b]
        Y_b = Y_blocks[b]
        ntraj_b = X_b.shape[0]
        
        # --- A. Full 2D Thermodynamics ---
        thermo_stot_2d = compute_thermodynamics_2d_GMM(X_b, Y_b, A, K, D, dt)
        Stot_Thermo_2D_blocks.append(thermo_stot_2d)
        
        # --- B. Strict X and Y Thermodynamics ---
        Stot_Thermo_X_blocks.append(compute_thermodynamics_1d_X(X_b, A, D, dt))
        Stot_Thermo_Y_blocks.append(compute_thermodynamics_1d_Y(Y_b, K, D, dt))
        
        # --- C. Projected 1D Angle Sweep (EPR vs Thermo) ---
        for deg, rad in zip(angles_deg, angles_rad):
            Z_1d_pos = X_b * np.cos(rad) + Y_b * np.sin(rad)
            
            # 1. Projected Thermodynamics Estimation
            thermo_1d = compute_thermodynamics_1d_projected_KM(Z_1d_pos, dt)
            Stot_Thermo_1D_blocks[deg].append(thermo_1d)
            
            # 2. Projected EPR Estimation
            Z_1d_formatted = np.zeros((ntraj_b, int(nsteps/stride), 2))
            Z_1d_formatted[:, :, 0] = np.arange(int(nsteps/stride)) * dt
            Z_1d_formatted[:, :, 1] = Z_1d_pos
            
            Sdots_1d, taus_1d = EPR_optimized_knn_histogram(Z_1d_formatted, dt, bins=int(np.sqrt(ntraj//n_blocks)), stridet=1)
            baseline_1d = np.mean(Sdots_1d[-100:])
            Stot_EPR_1D_blocks[deg].append(np.trapezoid(Sdots_1d - baseline_1d, taus_1d))
            Sdots_EPR[deg].append(Sdots_1d)
            
    # Calculate Means and Errors
    Thermo_Stot_2D_Mean = np.mean(Stot_Thermo_2D_blocks)
    Thermo_Stot_2D_Err = np.std(Stot_Thermo_2D_blocks) / np.sqrt(n_blocks)
    
    Thermo_Stot_X_Mean = np.mean(Stot_Thermo_X_blocks)
    Thermo_Stot_X_Err = np.std(Stot_Thermo_X_blocks) / np.sqrt(n_blocks)
    
    Thermo_Stot_Y_Mean = np.mean(Stot_Thermo_Y_blocks)
    Thermo_Stot_Y_Err = np.std(Stot_Thermo_Y_blocks) / np.sqrt(n_blocks)
    
    Integrated_Stot_1D_Means = [np.mean(Stot_EPR_1D_blocks[deg]) for deg in angles_deg]
    Integrated_Stot_1D_Errs = [np.std(Stot_EPR_1D_blocks[deg]) / np.sqrt(n_blocks) for deg in angles_deg]
    
    Thermo_Stot_1D_Means = [np.mean(Stot_Thermo_1D_blocks[deg]) for deg in angles_deg]
    Thermo_Stot_1D_Errs = [np.std(Stot_Thermo_1D_blocks[deg]) / np.sqrt(n_blocks) for deg in angles_deg]

    print(f"\nTime: {time.time()-t0:.1f}s")
    
    # ==========================================
    # 5. Plotting
    # ==========================================
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # 2D Limits
    ax.axhline(Analytical_Stot_2D, color='black', linestyle='-', linewidth=2, label=r'Exact $\Delta F$ (2D Limit)')
    ax.axhline(Thermo_Stot_2D_Mean, color='tab:green', linestyle='--', linewidth=2, label=r'Empirical Thermo $S_{tot}$ (2D)')
    
    # X and Y Analytical Limits
    ax.axhline(Analytical_Stot_X, color='blue', linestyle=':', alpha=0.7, label=r'Exact $\Delta F$ (X-Limit)')
    ax.axhline(Analytical_Stot_Y, color='tab:orange', linestyle=':', alpha=0.7, label=r'Exact $\Delta F$ (Y-Limit)')
    
    # Plot the 1D Thermodynamic Data
    ax.errorbar(angles_deg, Thermo_Stot_1D_Means, yerr=Thermo_Stot_1D_Errs, fmt='s--', capsize=5, 
                 color='tab:green', markersize=8, linewidth=2, label=r'Thermo Estimation (PMF via KDE)')

    # Plot the 1D EPR Data
    ax.errorbar(angles_deg, Integrated_Stot_1D_Means, yerr=Integrated_Stot_1D_Errs, fmt='o-', capsize=5, 
                 color='purple', markersize=8, linewidth=2, label=r'Kinematic EPR ($v^2$ integration)')
    
    ax.set_title('Thermodynamic vs Kinematic Entropy Across Projections', fontsize=16, pad=15)
    ax.set_xlabel(r'Observation Angle $\theta$ (Degrees)', fontsize=14)
    ax.set_ylabel(r'Entropy Production ($k_B T$)', fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='center right', fontsize=11, ncol=1)
    ax.set_xlim(-5, 95)
    
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.05)
    
    plt.tight_layout()
    plt.show()