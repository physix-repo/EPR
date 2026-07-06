import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
import time
import warnings
import os
import sys
from scipy.interpolate import griddata, RegularGridInterpolator
from sklearn.mixture import GaussianMixture

# 1. Dynamically add the parent directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# 2. Now use an absolute import instead of a relative one
from utils import EPR, simulate_2d_sinusoidal_dirac, compute_macrostate_free_energy

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
# 3. Execution & Block Averaging
# ==========================================

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    
    method = 'biased'

    A, K, D = 8.0, 40.0, 1.0       
    dt, nsteps, ntraj = 0.001, 1000, 20000
    n_blocks = 20
    stride = 5
    
    path_nodes_list = [2, 4, 6, 16]
    
    grid_array, x_lin, y_lin = load_committor_grid('./sol_ongrid.dat')

    print(f"\n--- Simulating Coupled Sinusoidal Landscape (A={A}, K={K}) ---")
    t0 = time.time()
    
    X_master, Y_master = simulate_2d_sinusoidal_dirac(A, K, D, dt, nsteps, ntraj)
    X_start, X_end = X_master[:, 1], X_master[:, -1]
    Y_start, Y_end = Y_master[:, 1], Y_master[:, -1]

    # ----------------------------------------------------
    # Direct estimation Free energy
    # ----------------------------------------------------
    print("Computing Free Energy Drop (2D GMM)...")
    
    x_min = min(np.min(X_start), np.min(X_end)) - 1.0
    x_max = max(np.max(X_start), np.max(X_end)) + 1.0
    y_min = min(np.min(Y_start), np.min(Y_end)) - 1.0
    y_max = max(np.max(Y_start), np.max(Y_end)) + 1.0

    bins_2d = 150 
    x_edges_2d = np.linspace(x_min, x_max, bins_2d + 1)
    y_edges_2d = np.linspace(y_min, y_max, bins_2d + 1)
    
    dx_2d = x_edges_2d[1] - x_edges_2d[0]
    dy_2d = y_edges_2d[1] - y_edges_2d[0]
    dA = dx_2d * dy_2d 
    
    x_centers_2d = (x_edges_2d[:-1] + x_edges_2d[1:]) / 2.0
    y_centers_2d = (y_edges_2d[:-1] + y_edges_2d[1:]) / 2.0
    
    XX, YY = np.meshgrid(x_centers_2d, y_centers_2d, indexing='ij')
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    
    # Fit Start state (1 Component)
    data_start = np.vstack([X_start, Y_start]).T
    gmm_start = GaussianMixture(n_components=1, covariance_type='full', random_state=SEED)
    gmm_start.fit(data_start)
    rho_2d_start = np.exp(gmm_start.score_samples(grid_points)).reshape(XX.shape)

    # Fit End state (2 Components)
    data_end = np.vstack([X_end, Y_end]).T
    gmm_end = GaussianMixture(n_components=2, covariance_type='full', random_state=SEED)
    gmm_end.fit(data_end)
    rho_2d_end = np.exp(gmm_end.score_samples(grid_points)).reshape(XX.shape)

    # Effective Potential with sinusoidal dependency
    V_2d = (A * (XX**2 - 1.0)**2 + 0.5 * K * (YY - np.sin(np.pi * (XX + 1.0)))**2) / D
    
    F_2d_start, U_2d_start, S_2d_start = compute_macrostate_free_energy(rho_2d_start, V_2d, dA)
    F_2d_end, U_2d_end, S_2d_end = compute_macrostate_free_energy(rho_2d_end, V_2d, dA)
    Delta_F_2D_macro = F_2d_start - F_2d_end

    # ----------------------------------------------------
    # EPR estimation
    # ----------------------------------------------------
    print(f"\nComputing Block-Averaged 1D EPR projections ({n_blocks} blocks of {ntraj//n_blocks} trajectories)...")
    X_blocks = np.array_split(X_master, n_blocks, axis=0)
    Y_blocks = np.array_split(Y_master, n_blocks, axis=0)
    
    # Ledgers
    Stot_Committor_blocks = []
    Sdots_Committor_blocks = []
    Stot_PathCV_blocks = {n: [] for n in path_nodes_list}
    Sdots_PathCV_blocks = {n: [] for n in path_nodes_list}
    
    for b in range(n_blocks):
        X_b, Y_b = X_blocks[b], Y_blocks[b]
        ntraj_b = X_b.shape[0]
        
        # --- 1. Committor CV EPR ---
        CV_comm = get_committor_cv(X_b, Y_b, grid_array, x_lin, y_lin, use_logit_transform=True)
        Z_comm = np.zeros((ntraj_b, nsteps, 2))
        Z_comm[:, :, 0] = np.arange(nsteps) * dt
        Z_comm[:, :, 1] = CV_comm
        
        if method == 'biased':
            Sdots_c, taus_c = EPR(Z_comm[:, ::stride, :], dt*stride, bins=int(np.sqrt(ntraj//n_blocks)))
            Stot_Committor_blocks.append(np.trapezoid(Sdots_c - np.mean(Sdots_c[-int(nsteps*0.1/stride):]), taus_c))
        else:
            Sdots_c, taus_c = EPR(Z_comm[:, ::stride, :], dt*stride, bins=int(np.sqrt(ntraj//n_blocks)), method=method)
            Stot_Committor_blocks.append(np.trapezoid(Sdots_c, taus_c))
            
        Sdots_Committor_blocks.append(Sdots_c)
        
        # --- 2. Path CV Sweep ---
        for n_nodes in path_nodes_list:
            CV_path = compute_path_cv(X_b, Y_b, n_nodes)
            Z_path = np.zeros((ntraj_b, nsteps, 2))
            Z_path[:, :, 0] = np.arange(nsteps) * dt
            Z_path[:, :, 1] = CV_path
            
            if method == 'biased':
                Sdots_p, taus_p = EPR(Z_path[:, ::stride, :], dt*stride, bins=int(np.sqrt(ntraj//n_blocks)))
                Stot_PathCV_blocks[n_nodes].append(np.trapezoid(Sdots_p - np.mean(Sdots_p[-int(nsteps*0.1/stride):]), taus_p))
            else:
                Sdots_p, taus_p = EPR(Z_path[:, ::stride, :], dt*stride, bins=int(np.sqrt(ntraj//n_blocks)), method=method)
                Stot_PathCV_blocks[n_nodes].append(np.trapezoid(Sdots_p, taus_p))
                
            Sdots_PathCV_blocks[n_nodes].append(Sdots_p)

    # Calculate Means and Errors
    Comm_M, Comm_E = np.mean(Stot_Committor_blocks), np.std(Stot_Committor_blocks)/np.sqrt(n_blocks)
    Path_M = [np.mean(Stot_PathCV_blocks[n]) for n in path_nodes_list]
    Path_E = [np.std(Stot_PathCV_blocks[n])/np.sqrt(n_blocks) for n in path_nodes_list]
    Sdots_P = [Sdots_PathCV_blocks[n] for n in path_nodes_list]

    print(f"\n--- Results ({time.time()-t0:.1f}s) ---")
    print(f"  Macrostate Delta F (2D GMM): {Delta_F_2D_macro:.3f} kT")
    print(f"  True Committor CV EPR:       {Comm_M:.3f} +/- {Comm_E:.3f} kT")

    # ==========================================
    # 4. Save & Plot
    # ==========================================
    save_filename = 'sinusoidal_{}k_ntraj_{}_blocks_{}_results.npz'.format(int(ntraj//1000), n_blocks, method)

    np.savez_compressed(
        save_filename, 
        SEED=SEED, path_nodes_list=path_nodes_list, Sdots_P=Sdots_P, Path_M=Path_M, Path_E=Path_E,
        Delta_F_2D_macro=Delta_F_2D_macro,
        Sdots_C=Sdots_Committor_blocks, Comm_M=Comm_M, Comm_E=Comm_E
    )
