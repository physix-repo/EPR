import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# ==========================================
# 1. Committor Grid & Theoretical Setup
# ==========================================

def load_committor_grid(filepath):
    """Loads the committor grid and dynamically infers the grid dimensions."""
    print(f"Loading committor grid from {filepath}...")
    try:
        data = np.loadtxt(filepath)
        ngrid = int(np.sqrt(data.size))
        grid_array = data.reshape((ngrid, ngrid))
        print(f"  -> Successfully loaded {ngrid}x{ngrid} committor grid.")
        return grid_array, ngrid
    except Exception as e:
        print(f"Error loading grid: {e}")
        print("Creating a dummy grid for testing purposes...")
        # Fallback dummy grid if file doesn't exist on local run
        ngrid = 100
        x = np.linspace(-1.5, 1.5, ngrid)
        dummy_grid = np.zeros((ngrid, ngrid))
        for i in range(ngrid):
            dummy_grid[i, :] = np.tanh(x[i] * 2) # Fake committor from -1 to 1
        return dummy_grid, ngrid

def get_nearest_values_array(x_query, y_query, grid_array, x_bounds=(-1.5, 1.5), y_bounds=(-1.0, 1.0)):
    """Maps 2D coordinates to the closest true committor value on the pre-computed grid."""
    nx, ny = grid_array.shape
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    x_q = np.asarray(x_query)
    y_q = np.asarray(y_query)
    
    i = np.round((x_q - x_min) / dx).astype(int)
    j = np.round((y_q - y_min) / dy).astype(int)
    
    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)

    # Convert from [-1, 1] space to [0, 1] probability space
    val = (1.0 + grid_array[i, j]) / 2.0
    return val

def target_function(x, a, b):
    """The ideal 1D committor profile (Tanh)."""
    return 0.5 * (1.0 + np.tanh((x - a) / (2.0 * b)))

# ==========================================
# 2. Trajectory Generation & Sampling
# ==========================================

def generate_trajectory_pool(A, K, D, dt=0.005, nsteps=50000, ntraj=20):
    """Generates a quick pool of uncoupled 2D Langevin trajectories to sample from."""
    print(f"Generating 2D trajectory pool for sampling...")
    X = np.zeros((ntraj, nsteps))
    Y = np.zeros((ntraj, nsteps))
    
    # Start uniformly distributed to ensure good coverage across the barrier
    X[:, 0] = np.random.uniform(-1.5, 1.5, ntraj)
    Y[:, 0] = np.random.uniform(-1.0, 1.0, ntraj)
    
    sqrt_2Ddt = np.sqrt(2.0 * D * dt)
    
    for t in range(1, nsteps):
        fx = -4.0 * A * X[:, t-1] * (X[:, t-1]**2 - 1.0)
        fy = -K * Y[:, t-1]
        noise_x = np.random.normal(0, sqrt_2Ddt, ntraj)
        noise_y = np.random.normal(0, sqrt_2Ddt, ntraj)
        X[:, t] = X[:, t-1] + D * fx * dt + noise_x
        Y[:, t] = Y[:, t-1] + D * fy * dt + noise_y
        
    return X.flatten(), Y.flatten()

def sample_points_from_bins(X_flat, Y_flat, param_1d, bin_edges, n_samples=10):
    """Uniformly samples points along the 1D reaction coordinate to prevent basin bias."""
    sampled_x, sampled_y, sampled_cv = [], [], []
    
    for i in range(1, len(bin_edges)):
        mask = (param_1d < bin_edges[i]) & (param_1d >= bin_edges[i-1])
        available_indices = np.where(mask)[0]
        
        n_available = len(available_indices)
        if n_available == 0:
            continue
            
        # If fewer points available than requested, take what we have
        n_select = min(n_samples, n_available)
        idx = np.random.choice(available_indices, n_select, replace=False)
        
        sampled_x.extend(X_flat[idx])
        sampled_y.extend(Y_flat[idx])
        sampled_cv.extend(param_1d[idx])
        
    return np.array(sampled_x), np.array(sampled_y), np.array(sampled_cv)

# ==========================================
# 3. Main Execution Pipeline
# ==========================================

if __name__ == "__main__":
    # --- A. Load Previous EPR Data ---
    npz_filename = 'trifecta_2D_meshless_results.npz'
    if not os.path.exists(npz_filename):
        print(f"Error: {npz_filename} not found. Please run the 2D Sweep first.")
        exit()
        
    epr_data = np.load(npz_filename)
    A = float(epr_data['A'])
    K = float(epr_data['K'])
    D = float(epr_data['D'])
    angles_deg = epr_data['angles_deg']
    Integrated_Stot_1D_Means = epr_data['Integrated_Stot_1D_Means']
    Integrated_Stot_1D_Errs = epr_data['Integrated_Stot_1D_Errs']
    
    # --- B. Load Committor Grid ---
    grid_path = '/home/sorbonne/ProductionEntropy/QuarticDoubleWell/sol_ongrid.dat'
    grid_array, ngrid = load_committor_grid(grid_path)
    
    # --- C. Generate and Sample Trajectories ---
    X_flat, Y_flat = generate_trajectory_pool(A, K, D)
    bin_edges = np.linspace(-1.5, 1.5, 101)
    
    # Data Storage for export
    rss_list = []
    a_opt_list = []
    b_opt_list = []
    
    print("\n--- Fitting Committor Profiles ---")
    for theta_deg in angles_deg:
        theta_rad = np.radians(theta_deg)
        
        # 1. Project full pool onto 1D Reaction Coordinate
        CV_full = X_flat * np.cos(theta_rad) + Y_flat * np.sin(theta_rad)
        
        # 2. Extract uniform samples across the CV (prevents overfitting to the dense basins)
        X_sub, Y_sub, CV_sub = sample_points_from_bins(X_flat, Y_flat, CV_full, bin_edges, n_samples=15)
        
        # 3. Get true committor value for these sampled 2D points
        true_committors = get_nearest_values_array(X_sub, Y_sub, grid_array)
        
        # 4. Curve Fitting
        initial_guess = [0.0, 0.5] # a=offset (0), b=width (0.5)
        
        try:
            # Fit the target tanh function to the sampled committors
            popt, _ = curve_fit(target_function, CV_sub, true_committors, p0=initial_guess, maxfev=5000)
            a_opt, b_opt = popt
            
            # Compute RSS
            y_fit = target_function(CV_sub, a_opt, b_opt)
            residuals = true_committors - y_fit
            rss = np.sum(residuals**2)
            
        except Exception as e:
            print(f"  Angle {theta_deg:5.1f}° -> Fit Failed! Assigning NaN.")
            a_opt, b_opt = np.nan, np.nan
            rss = np.nan
            
        rss_list.append(rss)
        a_opt_list.append(a_opt)
        b_opt_list.append(b_opt)
        
        if not np.isnan(rss):
            print(f"  Angle {theta_deg:5.1f}° -> RSS: {rss:.4f}  (a={a_opt:.2f}, b={b_opt:.2f})")

    # ==========================================
    # 4. Save RSS Data
    # ==========================================
    save_filename = 'rss_committor_results.npz'
    print(f"\nSaving RSS results to '{save_filename}'...")
    
    np.savez_compressed(
        save_filename,
        angles_deg=angles_deg,
        rss_list=rss_list,
        a_opt_list=a_opt_list,
        b_opt_list=b_opt_list
    )
    print("Data successfully saved!")

    # ==========================================
    # 5. Dual-Axis Plotting (EPR vs RSS)
    # ==========================================
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    # --- Left Axis: Hidden Entropy (from NPZ) ---
    color1 = 'tab:red'
    ax1.set_xlabel(r'Observation Angle $\theta$ (Degrees)', fontsize=12)
    ax1.set_ylabel(r'Apparent Entropy Production ($k_B T$)', color=color1, fontsize=12)
    ax1.errorbar(angles_deg, Integrated_Stot_1D_Means, yerr=Integrated_Stot_1D_Errs, 
                 fmt='o-', capsize=5, color=color1, markersize=8, linewidth=2, 
                 label=r'Apparent $S_{tot}$ (EPR)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Right Axis: Reaction Coordinate Quality (RSS) ---
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel(r'Committor RSS (Fit Error)', color=color2, fontsize=12)
    ax2.plot(angles_deg, rss_list, 's--', color=color2, markersize=8, linewidth=2, 
             label=r'Residual Sum of Squares (RSS)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Titles and Formatting
    plt.title('Reaction Coordinate Quality vs. Entropy Production', fontsize=14, pad=15)
    
    # Combine Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    
    plt.xlim(-5, 95)
    fig.tight_layout()
    plt.show()