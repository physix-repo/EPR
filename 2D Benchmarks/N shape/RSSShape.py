import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, RegularGridInterpolator
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# ==========================================
# 1. Committor Grid Setup (3-Column)
# ==========================================

def load_committor_grid_3col(filepath):
    """Loads a 3-column (x, y, h) committor file and builds the exact 2D mesh."""
    print(f"Loading (x, y, h) committor data from {filepath}...")
    data = np.loadtxt(filepath)
    x_val, y_val, h_val = data[:, 0], data[:, 1], data[:, 2]

    x_lin = np.unique(x_val)
    y_lin = np.unique(y_val)
    nx, ny = len(x_lin), len(y_lin)
    
    X_mesh, Y_mesh = np.meshgrid(x_lin, y_lin, indexing='ij')
    grid_array = griddata((x_val, y_val), h_val, (X_mesh, Y_mesh), method='nearest')
    
    print(f"  -> Successfully reconstructed a {nx}x{ny} committor grid.")
    return grid_array, x_lin, y_lin

def get_true_committor(X, Y, grid_array, x_lin, y_lin):
    """Maps continuous 2D coordinates to the true committor value."""
    interpolator = RegularGridInterpolator((x_lin, y_lin), grid_array, 
                                           method='cubic', bounds_error=False, fill_value=None)
    pts = np.stack((X, Y), axis=-1)
    q_smooth = interpolator(pts)
    q_smooth = np.clip(q_smooth, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Scale from [-1, 1] to [0, 1]
    return (1.0 + q_smooth) / 2.0

def target_function(x, a, b):
    """The ideal 1D committor profile (Tanh mapping)."""
    return 0.5 * (1.0 + np.tanh((x - a) / (2.0 * b)))

# ==========================================
# 2. Physics & Path CV
# ==========================================

def generate_sinusoidal_pool(A, K, D, dt=0.0001, nsteps=10000, ntraj=100):
    """Generates 2D Langevin trajectories on the coupled sinusoidal landscape."""
    print("Generating 2D sinusoidal trajectory pool for sampling...")
    X = np.zeros((ntraj, nsteps))
    Y = np.zeros((ntraj, nsteps))
    
    # Start uniformly distributed across the domain to ensure good coverage
    X[:, 0] = np.random.uniform(-1.3, 1.3, ntraj)
    Y[:, 0] = np.sin(np.pi * (X[:, 0] + 1.0)) + np.random.normal(0, 0.5, ntraj)
    
    sqrt_2Ddt = np.sqrt(2.0 * D * dt)
    
    for t in range(1, nsteps):
        X_prev, Y_prev = X[:, t-1], Y[:, t-1]
        
        sin_shift = np.sin(np.pi * (X_prev + 1.0))
        d_sin_shift = np.pi * np.cos(np.pi * (X_prev + 1.0))
        
        fx = -4.0 * A * X_prev * (X_prev**2 - 1.0) + K * (Y_prev - sin_shift) * d_sin_shift
        fy = -K * (Y_prev - sin_shift)
        
        X[:, t] = X_prev + D * fx * dt + np.random.normal(0, sqrt_2Ddt, ntraj)
        Y[:, t] = Y_prev + D * fy * dt + np.random.normal(0, sqrt_2Ddt, ntraj)
        
    return X.flatten(), Y.flatten()

def compute_path_cv(X, Y, n_nodes):
    """Computes the continuous 1D Path CV along the sinusoid."""
    x_nodes = np.linspace(-1.3, 1.3, n_nodes)
    y_nodes = np.sin(np.pi * (x_nodes + 1.0))
    
    node_dist_sq = (x_nodes[1] - x_nodes[0])**2 + (y_nodes[1] - y_nodes[0])**2
    lambda_val = -np.log(0.5) / (node_dist_sq / 4.0)
    
    num = np.zeros_like(X)
    den = np.zeros_like(X)
    
    for i in range(n_nodes):
        dist_sq = (X - x_nodes[i])**2 + (Y - y_nodes[i])**2
        weight = np.exp(-lambda_val * dist_sq)
        num += (i / float(n_nodes - 1)) * weight
        den += weight
        
    return num / (den + 1e-10)

def sample_points_from_bins(X_flat, Y_flat, param_1d, bin_edges, n_samples=10):
    """Uniformly samples points along the 1D reaction coordinate to prevent basin bias."""
    sampled_x, sampled_y, sampled_cv = [], [], []
    for i in range(1, len(bin_edges)):
        mask = (param_1d < bin_edges[i]) & (param_1d >= bin_edges[i-1])
        available_indices = np.where(mask)[0]
        
        n_available = len(available_indices)
        if n_available == 0:
            continue
            
        n_select = min(n_samples, n_available)
        idx = np.random.choice(available_indices, n_select, replace=False)
        sampled_x.extend(X_flat[idx])
        sampled_y.extend(Y_flat[idx])
        sampled_cv.extend(param_1d[idx])
        
    return np.array(sampled_x), np.array(sampled_y), np.array(sampled_cv)

# ==========================================
# 3. Execution Pipeline
# ==========================================

if __name__ == "__main__":

    SEED = 42
    np.random.seed(SEED)

    A, K, D = 8.0, 40.0, 1.0
    
    # 1. Attempt to load previous Path CV Entropy Data
    npz_filename = './N_shape/sinusoidal_20k_ntraj_20_blocks_biased_results.npz'
    if os.path.exists(npz_filename):
        print(f"Loaded thermodynamic data from {npz_filename}")
        epr_data = np.load(npz_filename)
        path_nodes_list = epr_data['path_nodes_list']
        Path_M = epr_data['Path_M']
        Path_E = epr_data['Path_E']
    else:
        print("Warning: Previous EPR data not found. Using default nodes.")
        path_nodes_list = [2, 4, 6, 16]#[2, 4, 6, 8, 16]
        Path_M = np.zeros(len(path_nodes_list))
        Path_E = np.zeros(len(path_nodes_list))

    # 2. Load Committor & Generate Pool
    grid_path = '/home/sorbonne/ProductionEntropy/Spotential/data/sol_ongrid.dat'
    grid_array, x_lin, y_lin = load_committor_grid_3col(grid_path)
    X_flat, Y_flat = generate_sinusoidal_pool(A, K, D)
    
    # The Path CV outputs values between [0, 1]
    bin_edges = np.linspace(0.0, 1.0, 101) 
    
    rss_list = []
    
    print("\n--- Fitting Committor Profiles ---")
    for n_nodes in path_nodes_list:
        # A. Project onto Path CV
        CV_full = compute_path_cv(X_flat, Y_flat, n_nodes)
        
        # B. Uniform Sampling
        X_sub, Y_sub, CV_sub = sample_points_from_bins(X_flat, Y_flat, CV_full, bin_edges, n_samples=15)
        
        # C. Get True Committor
        true_committors = get_true_committor(X_sub, Y_sub, grid_array, x_lin, y_lin)
        
        # D. Curve Fitting
        # Since CV is [0, 1], the barrier should be near 0.5. Initial guess: center=0.5, width=0.1
        initial_guess = [0.5, 0.1] 
        
        try:
            popt, _ = curve_fit(target_function, CV_sub, true_committors, p0=initial_guess, maxfev=5000)
            a_opt, b_opt = popt
            
            y_fit = target_function(CV_sub, a_opt, b_opt)
            residuals = true_committors - y_fit
            rss = np.sum(residuals**2)
        except Exception:
            print(f"  Nodes {n_nodes:3d} -> Fit Failed!")
            rss = np.nan
            
        rss_list.append(rss)
        print(f"  Nodes {n_nodes:3d} -> RSS: {rss:.4f}")
    
    save_filename = './N_shape/rss_committor_sinus_results.npz'
    print(f"\nSaving RSS results to '{save_filename}'...")
    
    np.savez_compressed(
        save_filename,
        path_nodes_list=path_nodes_list,
        rss_list=rss_list,
    )
    print("Data successfully saved!")

    # ==========================================
    # 4. Dual-Axis Plotting
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    # --- Left Axis: Entropy Production ---
    color1 = 'purple'
    ax1.set_xlabel('Number of String Nodes (Path Resolution)', fontsize=12)
    ax1.set_ylabel(r'Apparent Entropy Production ($k_B T$)', color=color1, fontsize=12)
    ax1.errorbar(path_nodes_list, Path_M, yerr=Path_E, fmt='o-', capsize=5, 
                 color=color1, markersize=8, linewidth=2, label=r'Path CV $S_{tot}$ (EPR)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')
    ax1.set_xticks(path_nodes_list)
    ax1.set_xticklabels(path_nodes_list)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Right Axis: RSS Error ---
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel(r'Committor RSS (Fit Error)', color=color2, fontsize=12)
    ax2.plot(path_nodes_list, rss_list, 's--', color=color2, markersize=8, linewidth=2, 
             label=r'Residual Sum of Squares (RSS)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Path CV Quality (RSS) vs. Entropy Production', fontsize=14, pad=15)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left')
    
    plt.xlim(min(path_nodes_list)-1, max(path_nodes_list)*1.1)
    fig.tight_layout()
    plt.show()