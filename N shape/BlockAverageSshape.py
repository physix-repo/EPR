import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
import time
import warnings
import os
from utils import EPR_optimized_knn_histogram
from scipy.interpolate import griddata, RegularGridInterpolator
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)

# ==========================================
# 1. Functions
# ==========================================

def simulate_2d_sinusoidal_dirac(A, K, D, dt, nsteps, ntraj):
    """
    Generates overdamped Langevin trajectories for a coupled 2D landscape.
    X: Quartic Double Well
    Y: Harmonic Well shifted by sin(pi * (X + 1))
    """
    X = np.zeros((ntraj, nsteps))
    Y = np.zeros((ntraj, nsteps))
    
    sqrt_2Ddt = np.sqrt(2.0 * D * dt)
    
    for t in range(1, nsteps):
        X_prev = X[:, t-1]
        Y_prev = Y[:, t-1]
        
        # Derivatives of the sinusoidal shift
        sin_shift = np.sin(np.pi * (X_prev + 1.0))
        d_sin_shift = np.pi * np.cos(np.pi * (X_prev + 1.0))
        
        # Forces
        fx = -4.0 * A * X_prev * (X_prev**2 - 1.0) + K * (Y_prev - sin_shift) * d_sin_shift
        fy = -K * (Y_prev - sin_shift)
        
        noise_x = np.random.normal(0, sqrt_2Ddt, ntraj)
        noise_y = np.random.normal(0, sqrt_2Ddt, ntraj)
        
        X[:, t] = X_prev + D * fx * dt + noise_x
        Y[:, t] = Y_prev + D * fy * dt + noise_y
        
    return X, Y

def get_exact_analytical_Stot_sinusoidal(A, K, D, dt):
    """Computes the absolute mathematical truth for the 2D sinusoidal potential."""
    sigma = np.sqrt(2.0 * D * dt)
    
    # 1. Partition Function Z
    # Because Y is harmonic with a shift depending entirely on X, the Y-integral factors out perfectly!
    def boltzmann_x(x): 
        return np.exp(-A * (x**2 - 1.0)**2 / D)
    Z_X, _ = integrate.quad(boltzmann_x, -np.inf, np.inf)
    Z_2D = Z_X * np.sqrt(2.0 * np.pi * D / K)
    
    # 2. V_mean_start (Numerical integration of V(X,Y) over the initial t=dt Gaussian)
    def v_start_integrand(x, y):
        V = A * (x**2 - 1.0)**2 + 0.5 * K * (y - np.sin(np.pi * (x + 1.0)))**2
        rho = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2)
        return V * rho
        
    bound = 6.0 * sigma
    V_mean_start, _ = integrate.dblquad(v_start_integrand, -bound, bound, lambda x: -bound, lambda x: bound)
    
    # 3. Initial Entropy
    S_start_mean = -np.log(2.0 * np.pi * np.e * sigma**2)
    
    Analytical_Stot_2D = (V_mean_start / D + S_start_mean) - (-np.log(Z_2D))
    return Analytical_Stot_2D

# ==========================================
# 2. Collective Variables (CVs)
# ==========================================

def compute_path_cv(X, Y, n_nodes):
    """Computes a string-method style 1D Path CV along the sinusoid."""
    # Define the idealized path nodes
    x_nodes = np.linspace(-1.3, 1.3, n_nodes)
    y_nodes = np.sin(np.pi * (x_nodes + 1.0))

    nodes = np.column_stack([x_nodes, y_nodes])
    
    # Dynamic lambda to ensure sharp distinction between nodes
    dR = np.mean(np.linalg.norm(nodes[1:] - nodes[:-1], axis=1))
    lambda_val = 2.3 / (dR**2)  
    
    num = np.zeros_like(X)
    den = np.zeros_like(X)
    
    for i in range(n_nodes):
        dist_sq = (X - x_nodes[i])**2 + (Y - y_nodes[i])**2
        weight = np.exp(-lambda_val * dist_sq)
        num += (i / float(n_nodes - 1)) * weight
        den += weight
        
    s_cv = num / (den + 1e-12)
    return s_cv

def load_committor_grid(filepath):
    """Loads a 3-column (x, y, h) committor file and builds the exact 2D mesh."""
    print(f"Loading (x, y, h) committor data from {filepath}...")
    try:
        data = np.loadtxt(filepath)
        
        x_val = data[:, 0]
        y_val = data[:, 1]
        h_val = data[:, 2]

        # Extract the unique coordinate ticks, automatically sorted
        x_lin = np.unique(x_val)
        y_lin = np.unique(y_val)
        nx, ny = len(x_lin), len(y_lin)
        
        # Reconstruct the 2D array grid_array[i, j] to match (x_lin[i], y_lin[j])
        X_mesh, Y_mesh = np.meshgrid(x_lin, y_lin, indexing='ij')
        
        # griddata perfectly maps the unstructured list back onto the structured 2D grid
        grid_array = griddata((x_val, y_val), h_val, (X_mesh, Y_mesh), method='nearest')
        
        print(f"  -> Successfully reconstructed a {nx}x{ny} committor grid.")
        return grid_array, x_lin, y_lin
        
    except Exception as e:
        print(f"Error loading grid: {e}")
        print("Creating a dummy grid for testing purposes...")
        x_lin = np.linspace(-1.5, 1.5, 100)
        y_lin = np.linspace(-2.0, 2.0, 100)
        X_mesh, _ = np.meshgrid(x_lin, y_lin, indexing='ij')
        dummy_grid = np.tanh(X_mesh * 2.0) # Fake barrier
        return dummy_grid, x_lin, y_lin

def get_committor_cv(X, Y, grid_array, x_lin, y_lin, use_logit_transform=True):
    """Maps continuous 2D matrices to a smooth Committor CV using the exact file axes."""
    
    # 1. CUBIC INTERPOLATION: Prevent grid-edge velocity explosions
    interpolator = RegularGridInterpolator((x_lin, y_lin), grid_array, 
                                           method='cubic', 
                                           bounds_error=False, fill_value=None)
    
    # Stack X and Y into a single array of points
    pts = np.stack((X, Y), axis=-1)
    q_smooth = interpolator(pts)
    
    # 2. BOUNDARY CLIPPING: Prevent extrapolation explosions
    # (Clips slightly inside [-1, 1] to prevent divide-by-zero later)
    q_smooth = np.clip(q_smooth, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Scale from [-1, 1] to standard Committor probability space [0, 1]
    # (Assuming your file stores h from -1 to 1 based on your previous code)
    q_prob = (1.0 + q_smooth) / 2.0
    
    # 3. LOGIT TRANSFORM: Prevents Division-by-Zero in the EPR Integral
    if use_logit_transform:
        return np.log(q_prob / (1.0 - q_prob))
    else:
        return q_prob
# ==========================================
# 3. Thermodynamic Estimators
# ==========================================

import numpy as np

def EPR_1D_macroscopic_flux(Z_pos, dt, bins=100, stridet=1):
    """
    Computes the 1D EPR using the Macroscopic Probability Flux (Current).
    This method is completely immune to the Itô-Hessian curvature artifact 
    caused by projecting curved manifolds, and it ONLY requires the 1D trajectory!
    
    Parameters:
        Z_pos: 2D array of shape (ntraj, nsteps) containing the 1D CV values.
        dt: Time step.
    """
    ntraj, nsteps = Z_pos.shape
    xmin, xmax = np.min(Z_pos), np.max(Z_pos)
    dx = (xmax - xmin) / bins
    if dx == 0: dx = 1e-12

    # =======================================================
    # 1. Global Diffusion Profile D(z)
    # =======================================================
    displacements = Z_pos[:, 1:] - Z_pos[:, :-1]
    starts = Z_pos[:, :-1]
    ix_all = np.clip(np.floor((starts - xmin) / dx).astype(int), 0, bins - 1).ravel()
    disp_flat = displacements.ravel()

    counts = np.bincount(ix_all, minlength=bins)
    mask_D = counts > 10

    sum_disp = np.bincount(ix_all, weights=disp_flat, minlength=bins)
    sum_disp_sq = np.bincount(ix_all, weights=disp_flat**2, minlength=bins)

    mean_disp = np.zeros(bins)
    mean_disp_sq = np.zeros(bins)
    mean_disp[mask_D] = sum_disp[mask_D] / counts[mask_D]
    mean_disp_sq[mask_D] = sum_disp_sq[mask_D] / counts[mask_D]

    var_disp = np.zeros(bins)
    var_disp[mask_D] = mean_disp_sq[mask_D] - mean_disp[mask_D]**2
    
    D_bin = np.zeros(bins)
    D_bin[mask_D] = var_disp[mask_D] / (2.0 * dt)
    if np.any(~mask_D):
        D_bin[~mask_D] = np.var(disp_flat) / (2.0 * dt)

    # Interpolate D(z) to the edges between the bins
    D_edge = (D_bin[:-1] + D_bin[1:]) / 2.0

    # =======================================================
    # 2. Time-resolved Entropy Integration via Net Flux
    # =======================================================
    it_indices = np.arange(stridet, nsteps - 1, stridet)
    Sdots = np.zeros(len(it_indices))
    taus = np.zeros(len(it_indices))

    for i, it in enumerate(it_indices):
        z_curr = Z_pos[:, it]
        z_next = Z_pos[:, it + 1]

        ix_curr = np.clip(np.floor((z_curr - xmin) / dx).astype(int), 0, bins - 1)
        ix_next = np.clip(np.floor((z_next - xmin) / dx).astype(int), 0, bins - 1)

        # --- A. Count Boundary Crossings ---
        # An edge 'k' separates bin 'k' and 'k+1'
        N_LR = np.zeros(bins - 1)
        N_RL = np.zeros(bins - 1)

        for k in range(bins - 1):
            N_LR[k] = np.sum((ix_curr <= k) & (ix_next > k))
            N_RL[k] = np.sum((ix_curr > k) & (ix_next <= k))

        # --- B. Unbiased Squared Flux (Current) ---
        # The true net flux is (N_LR - N_RL).
        # By subtracting (N_LR + N_RL), we exactly analytically remove 
        # the thermal Poisson counting noise variance.
        J_sq_unbiased = ((N_LR - N_RL)**2 - (N_LR + N_RL)) / (ntraj * dt)**2

        # --- C. Compute Density at the Edges ---
        counts_curr = np.bincount(ix_curr, minlength=bins)
        counts_next = np.bincount(ix_next, minlength=bins)
        
        # Average density over the jump for symmetry
        rho_bin = (counts_curr + counts_next) / (2.0 * ntraj * dx)
        rho_edge = (rho_bin[:-1] + rho_bin[1:]) / 2.0

        # --- D. Integrate Local Entropy ---
        # Mask out empty regions to prevent division by zero
        mask_edge = (rho_edge > 0) & (D_edge > 0)

        # Seifert's continuous formula: Integral( J^2 / (rho * D) ) dz
        Sdot_t = np.sum( (J_sq_unbiased[mask_edge] / (rho_edge[mask_edge] * D_edge[mask_edge])) * dx )

        Sdots[i] = Sdot_t
        taus[i] = it * dt

    return Sdots, taus

def EPR_2D_meshless(X, Y, dt, k_neighbors=50, stridet=1):
    ntraj, nsteps = X.shape
    it_indices = np.arange(stridet, nsteps - 1, stridet)
    Sdots = np.zeros(len(it_indices))
    taus = np.zeros(len(it_indices))
    
    for i, it in enumerate(it_indices):
        x_curr, y_curr = X[:, it], Y[:, it]
        positions = np.column_stack((x_curr, y_curr))
        
        # Central difference for velocities
        vx_curr = (X[:, it + 1] - X[:, it - 1]) / (2.0 * dt)
        vy_curr = (Y[:, it + 1] - Y[:, it - 1]) / (2.0 * dt)
        velocities = np.column_stack((vx_curr, vy_curr))
        
        # Forward difference for diffusion estimation
        dx_fwd = X[:, it + 1] - X[:, it]
        dy_fwd = Y[:, it + 1] - Y[:, it]
        
        tree = cKDTree(positions)
        distances, indices = tree.query(positions, k=k_neighbors, workers=-1)
        
        # Original biased velocity mean
        v_mean = np.mean(velocities[indices], axis=1) 
        
        # Estimation of the local diffusion profile D(x, y)
        dx_neigh = dx_fwd[indices]
        dy_neigh = dy_fwd[indices]
        D_local = (np.var(dx_neigh, axis=1) + np.var(dy_neigh, axis=1)) / (4.0 * dt)
        
        # Prevent division by zero in perfectly stagnant regions
        D_local = np.clip(D_local, 1e-12, None)
        
        # Original EPR integration using the new local D
        Sdots[i] = np.mean(np.sum(v_mean**2, axis=1) / D_local)
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

def compute_thermodynamics_2d(X_b, Y_b, A, K, D, dt):
    X_start, Y_start = X_b[:, 1], Y_b[:, 1]
    X_end, Y_end = X_b[:, -1], Y_b[:, -1]
    sigma_start = np.sqrt(2.0 * D * dt)
    ln_rho_start = -np.log(2.0 * np.pi * sigma_start**2) - (X_start**2 + Y_start**2) / (2.0 * sigma_start**2)
    
    try:
        kde_end = gaussian_kde(np.vstack([X_end, Y_end]))
        ln_rho_end = np.log(kde_end(np.vstack([X_end, Y_end])) + 1e-12)
    except Exception:
        ln_rho_end = np.zeros_like(X_end)
        
    Delta_S = ln_rho_start - ln_rho_end
    V_start = A * (X_start**2 - 1.0)**2 + 0.5 * K * (Y_start - np.sin(np.pi*(X_start+1)))**2
    V_end = A * (X_end**2 - 1.0)**2 + 0.5 * K * (Y_end - np.sin(np.pi*(X_end+1)))**2
    return np.mean((V_start - V_end) / D + Delta_S)

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
    
    # Updated V_start and V_end calculation for sinusoidal potential
    V_start = A * (X_start**2 - 1.0)**2 + 0.5 * K * (Y_start - np.sin(np.pi * (X_start + 1.0)))**2
    V_end = A * (X_end**2 - 1.0)**2 + 0.5 * K * (Y_end - np.sin(np.pi * (X_end + 1.0)))**2
    Q = (V_start - V_end) / D
    
    return np.mean(Q + Delta_S)


# ==========================================
# 4. Execution & Block Averaging
# ==========================================

if __name__ == "__main__":
    method = 'biased'

    A, K, D = 8.0, 40.0, 1.0       
    dt, nsteps, ntraj = 0.001, 1000, 200000
    bins_1d, k_neighbors, n_blocks = 50, 20, 20
    stride = 5
    
    path_nodes_list = [2, 4, 6, 8, 16]
    
    Analytical_Stot_2D = get_exact_analytical_Stot_sinusoidal(A, K, D, dt)
    grid_array, x_lin, y_lin = load_committor_grid('/home/sorbonne/ProductionEntropy/Spotential/sol_ongrid.dat')

    
    print(f"--- Simulating Coupled Sinusoidal Landscape (A={A}, K={K}) ---")
    t0 = time.time()
    
    X_master, Y_master = simulate_2d_sinusoidal_dirac(A, K, D, dt, nsteps, ntraj)
    X_blocks = np.array_split(X_master, n_blocks, axis=0)
    Y_blocks = np.array_split(Y_master, n_blocks, axis=0)
    
    # Ledgers
    Stot_EPR_2D_blocks = []
    Stot_Thermo_2D_blocks = []
    Stot_Committor_blocks = []
    Sdots_Committor_blocks = []
    Stot_PathCV_blocks = {n: [] for n in path_nodes_list}
    Sdots_PathCV_blocks = {n: [] for n in path_nodes_list}
    
    
    for b in range(n_blocks):
        print(f"  Processing Block {b+1}/{n_blocks}...")
        X_b, Y_b = X_blocks[b], Y_blocks[b]
        ntraj_b = X_b.shape[0]
        
        # 1. 2D Meshless
        Sdots_2d, taus_2d = EPR_2D_meshless(X_b, Y_b, dt, k_neighbors=int(np.sqrt(ntraj//n_blocks)))
        Stot_EPR_2D_blocks.append(np.trapezoid(Sdots_2d - np.mean(Sdots_2d[-100:]), taus_2d))
        #Stot_EPR_2D_blocks.append(np.trapezoid(Sdots_2d, taus_2d))
        
        # 2. 2D Boundary Thermodynamics
        Stot_Thermo_2D_blocks.append(compute_thermodynamics_2d_GMM(X_b, Y_b, A, K, D, dt))
        
        # 3. Committor CV EPR

        CV_comm = get_committor_cv(X_b, Y_b, grid_array, x_lin, y_lin, use_logit_transform=True)
        Z_comm = np.zeros((ntraj_b, nsteps, 2))
        Z_comm[:, :, 0] = np.arange(nsteps) * dt
        Z_comm[:, :, 1] = CV_comm
        if method == 'biased':
            Sdots_c, taus_c = EPR_optimized_knn_histogram(Z_comm[:, ::stride, :], dt*stride, bins=int(np.sqrt(ntraj//n_blocks)))
            #Sdots_c, taus_c = EPR_1D_macroscopic_flux(Z_comm[:, :, 1], dt, bins=int(np.sqrt(ntraj//n_blocks)))
            Stot_Committor_blocks.append(np.trapezoid(Sdots_c - np.mean(Sdots_c[-int(nsteps*0.1/stride):]), taus_c))
        else:
            #Sdots_c, taus_c = EPR_optimized_unbiased(Z_comm, dt, bins=int(np.sqrt(ntraj//n_blocks)))
            Sdots_c, taus_c = EPR_optimized_knn_histogram(Z_comm, dt, bins=int(np.sqrt(ntraj//n_blocks)), method='unbiased')
            Stot_Committor_blocks.append(np.trapezoid(Sdots_c - 0*np.mean(Sdots_c[-100:]), taus_c))
        Sdots_Committor_blocks.append(Sdots_c)
        #Stot_Committor_blocks.append(np.trapezoid(Sdots_c , taus_c))
        
        # 4. Path CV Sweep
        for n_nodes in path_nodes_list:
            CV_path = compute_path_cv(X_b, Y_b, n_nodes)
            Z_path = np.zeros((ntraj_b, nsteps, 2))
            Z_path[:, :, 0] = np.arange(nsteps) * dt
            Z_path[:, :, 1] = CV_path
            if method == 'biased':
                Sdots_p, taus_p = EPR_optimized_knn_histogram(Z_path[:, ::stride, :], dt*stride, bins=int(np.sqrt(ntraj//n_blocks)))
                #Sdots_p, taus_p = EPR_1D_macroscopic_flux(CV_path, dt, bins=int(np.sqrt(ntraj//n_blocks)))
                Stot_PathCV_blocks[n_nodes].append(np.trapezoid(Sdots_p - np.mean(Sdots_p[-int(nsteps*0.1/stride):]), taus_p))
            else:
                #Sdots_p, taus_p = EPR_optimized_unbiased(Z_path, dt, bins=int(np.sqrt(ntraj//n_blocks)))
                Sdots_p, taus_p = EPR_optimized_knn_histogram(Z_path, dt, bins=int(np.sqrt(ntraj//n_blocks)), method='unbiased')
                Stot_PathCV_blocks[n_nodes].append(np.trapezoid(Sdots_p - 0*np.mean(Sdots_p[-100:]), taus_p))
            Sdots_PathCV_blocks[n_nodes].append(Sdots_p)
            #Stot_PathCV_blocks[n_nodes].append(np.trapezoid(Sdots_p, taus_p))


    # Calculate Means and Errors
    EPR_2D_M, EPR_2D_E = np.mean(Stot_EPR_2D_blocks), np.std(Stot_EPR_2D_blocks)/np.sqrt(n_blocks)
    Thermo_2D_M, Thermo_2D_E = np.mean(Stot_Thermo_2D_blocks), np.std(Stot_Thermo_2D_blocks)/np.sqrt(n_blocks)
    Comm_M, Comm_E = np.mean(Stot_Committor_blocks), np.std(Stot_Committor_blocks)/np.sqrt(n_blocks)
    
    Path_M = [np.mean(Stot_PathCV_blocks[n]) for n in path_nodes_list]
    Path_E = [np.std(Stot_PathCV_blocks[n])/np.sqrt(n_blocks) for n in path_nodes_list]

    Sdots_P = [Sdots_PathCV_blocks[n] for n in path_nodes_list]

    print(f"\n--- Results Ledger ({time.time()-t0:.1f}s) ---")
    print(f"  Exact Analytical 2D DF:      {Analytical_Stot_2D:.4f} kT")
    print(f"  Empirical Boundary 2D Stot:  {Thermo_2D_M:.4f} +/- {Thermo_2D_E:.4f} kT")
    print(f"  Meshless 2D EPR:             {EPR_2D_M:.4f} +/- {EPR_2D_E:.4f} kT")
    print(f"  True Committor CV EPR:       {Comm_M:.4f} +/- {Comm_E:.4f} kT")

    # ==========================================
    # 5. Save & Plot
    # ==========================================
    if method == 'biased':
        save_filename = 'sinusoidal_pathCV_{}k_{}_blocks_stride_{}_biased_knn_squared_results.npz'.format(int(ntraj//1000), n_blocks, stride)
    else:
        save_filename = 'sinusoidal_pathCV_{}_{}_blocks_unbiased_bins_squared_results.npz'.format(int(ntraj//1000), n_blocks)
    np.savez_compressed(save_filename, 
        path_nodes_list=path_nodes_list, Sdots_P=Sdots_P, Path_M=Path_M, Path_E=Path_E,
        Analytical_Stot_2D=Analytical_Stot_2D, Thermo_2D_M=Thermo_2D_M, Thermo_2D_E=Thermo_2D_E,
        EPR_2D_M=EPR_2D_M, EPR_2D_E=EPR_2D_E, Sdots_C=Sdots_Committor_blocks, Comm_M=Comm_M, Comm_E=Comm_E)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Theoretical Ceilings
    ax.axhline(Analytical_Stot_2D, color='black', linestyle='-', lw=2, label=r'Exact $\Delta F$ (2D)')
    ax.axhline(Thermo_2D_M, color='tab:green', linestyle='--', lw=2, label=r'Empirical Boundary $S_{tot}$ (2D)')
    ax.axhline(EPR_2D_M, color='red', linestyle='-.', lw=2, label=r'Meshless EPR $S_{tot}$ (2D)')
    
    # Committor Reference
    ax.axhline(Comm_M, color='tab:orange', linestyle=':', lw=2, label=r'True Committor CV $S_{tot}$')
    ax.fill_between([min(path_nodes_list)-10, max(path_nodes_list)+10], Comm_M - Comm_E, Comm_M + Comm_E, color='tab:orange', alpha=0.15)
    
    # Path CV Scatter
    ax.errorbar(path_nodes_list, Path_M, yerr=Path_E, fmt='o-', capsize=5, color='purple', markersize=8, lw=2, label=r'Path CV $S_{tot}$ (Varying Nodes)')
    
    ax.set_title('Path CV vs True Committor in Coupled Sinusoidal Landscape', fontsize=14, pad=15)
    ax.set_xlabel('Number of String Nodes (Path Resolution)', fontsize=12)
    ax.set_ylabel(r'Entropy Production ($k_B T$)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    #ax.set_xscale('log')
    ax.set_xticks(path_nodes_list)
    ax.set_xticklabels(path_nodes_list)
    ax.set_xlim(min(path_nodes_list)-1, max(path_nodes_list)*1.1)
    
    plt.tight_layout()
    plt.show()