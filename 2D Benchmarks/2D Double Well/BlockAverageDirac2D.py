import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time
import warnings
import os
import sys
from sklearn.mixture import GaussianMixture

# 1. Dynamically add the parent directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# 2. Now use an absolute import instead of a relative one
from utils import EPR, simulate_2d_double_well_dirac, compute_macrostate_free_energy

warnings.filterwarnings("ignore", category=RuntimeWarning)




# ==========================================
# 1. Execution & Block Averaging
# ==========================================

if __name__ == "__main__":

    SEED = 42
    np.random.seed(SEED)

    method = 'biased'

    # Base Parameters
    A = 8.0       
    K = 40.0       
    D = 1.0       
    dt = 0.001    
    nsteps = 1000
    ntraj = 20000 
    n_blocks = 20
    
    angles_deg = np.linspace(0, 90, 6)
    angles_rad = np.radians(angles_deg)
    
    print(f"--- Simulating 2D Free Energy Relaxation (A={A}, K={K}) ---")
    t0 = time.time()
    
    # ----------------------------------------------------
    # 0. Simulate Master Ensemble
    # ----------------------------------------------------
    print(f"Generating Master Ensemble ({ntraj} trajectories)...")
    X_master, Y_master = simulate_2d_double_well_dirac(A, K, D, dt, nsteps, ntraj)

    X_start, X_end = X_master[:, 1], X_master[:, -1]
    Y_start, Y_end = Y_master[:, 1], Y_master[:, -1]

    # ----------------------------------------------------
    # 1. Calculate Macrostate Free Energy (Delta F)
    # ----------------------------------------------------
    print("Computing Macrostate Free Energy Drops (X, Y, and 2D)...")
    bins_thermo = int(np.sqrt(ntraj))

    # --- A. 1D X-Axis Macrostate ---
    x_min = min(np.min(X_start), np.min(X_end)) - 1.0
    x_max = max(np.max(X_start), np.max(X_end)) + 1.0
    rho_x_start, x_edges = np.histogram(X_start, bins=bins_thermo, range=(x_min, x_max), density=True)
    rho_x_end, _ = np.histogram(X_end, bins=bins_thermo, range=(x_min, x_max), density=True)
    dx = x_edges[1] - x_edges[0]
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    V_x = A * (x_centers**2 - 1.0)**2 / D

    F_x_start, U_x_start, S_x_start = compute_macrostate_free_energy(rho_x_start, V_x, dx)
    F_x_end, U_x_end, S_x_end = compute_macrostate_free_energy(rho_x_end, V_x, dx)
    Delta_F_X_macro = F_x_start - F_x_end

    # --- B. 1D Y-Axis Macrostate ---
    y_min = min(np.min(Y_start), np.min(Y_end)) - 1.0
    y_max = max(np.max(Y_start), np.max(Y_end)) + 1.0
    rho_y_start, y_edges = np.histogram(Y_start, bins=bins_thermo, range=(y_min, y_max), density=True)
    rho_y_end, _ = np.histogram(Y_end, bins=bins_thermo, range=(y_min, y_max), density=True)
    dy = y_edges[1] - y_edges[0]
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    V_y = 0.5 * K * y_centers**2 / D

    F_y_start, U_y_start, S_y_start = compute_macrostate_free_energy(rho_y_start, V_y, dy)
    F_y_end, U_y_end, S_y_end = compute_macrostate_free_energy(rho_y_end, V_y, dy)
    Delta_F_Y_macro = F_y_start - F_y_end

    # --- C. Full 2D Macrostate ---
    # Establish a high-resolution 2D grid for numerical integration
    bins_2d = 150 
    x_edges_2d = np.linspace(x_min, x_max, bins_2d + 1)
    y_edges_2d = np.linspace(y_min, y_max, bins_2d + 1)
    
    dx_2d = x_edges_2d[1] - x_edges_2d[0]
    dy_2d = y_edges_2d[1] - y_edges_2d[0]
    dA = dx_2d * dy_2d # 2D Integration area element
    
    x_centers_2d = (x_edges_2d[:-1] + x_edges_2d[1:]) / 2.0
    y_centers_2d = (y_edges_2d[:-1] + y_edges_2d[1:]) / 2.0
    
    XX, YY = np.meshgrid(x_centers_2d, y_centers_2d, indexing='ij')
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    
    # Fit Start state (1 Component: harmonic expansion from Dirac point)
    data_start = np.vstack([X_start, Y_start]).T
    gmm_start = GaussianMixture(n_components=1, covariance_type='full', random_state=SEED)
    gmm_start.fit(data_start)
    rho_2d_start = np.exp(gmm_start.score_samples(grid_points)).reshape(XX.shape)

    # Fit End state (2 Components: bi-modal probability at the double well minima)
    data_end = np.vstack([X_end, Y_end]).T
    gmm_end = GaussianMixture(n_components=2, covariance_type='full', random_state=SEED)
    gmm_end.fit(data_end)
    rho_2d_end = np.exp(gmm_end.score_samples(grid_points)).reshape(XX.shape)

    V_2d = (A * (XX**2 - 1.0)**2 + 0.5 * K * YY**2) / D
    
    F_2d_start, U_2d_start, S_2d_start = compute_macrostate_free_energy(rho_2d_start, V_2d, dA)
    F_2d_end, U_2d_end, S_2d_end = compute_macrostate_free_energy(rho_2d_end, V_2d, dA)
    Delta_F_2D_macro = F_2d_start - F_2d_end

    # ----------------------------------------------------
    # 2. Block Averaging Engine (1D Projections only)
    # ----------------------------------------------------
    print(f"Computing Block-Averaged 1D EPR projections ({n_blocks} blocks of {ntraj//n_blocks} trajectories)...")
    X_blocks = np.array_split(X_master, n_blocks, axis=0)
    Y_blocks = np.array_split(Y_master, n_blocks, axis=0)
    
    Stot_EPR_1D_blocks = {deg: [] for deg in angles_deg}
    Sdots_EPR = {deg: [] for deg in angles_deg}
    
    for b in range(n_blocks):
        X_b = X_blocks[b]
        Y_b = Y_blocks[b]
        ntraj_b = X_b.shape[0]
        
        # --- Projected 1D Angle Sweep (Grid-based EPR) ---
        for deg, rad in zip(angles_deg, angles_rad):
            Z_1d_pos = X_b * np.cos(rad) + Y_b * np.sin(rad)
            
            Z_1d_formatted = np.zeros((ntraj_b, int(nsteps), 2))
            Z_1d_formatted[:, :, 0] = np.arange(int(nsteps)) * dt
            Z_1d_formatted[:, :, 1] = Z_1d_pos
            
            if method == 'biased':
                Sdots_1d, taus_1d = EPR(Z_1d_formatted, dt, bins=int(np.sqrt(ntraj//n_blocks)))
                baseline_1d = np.mean(Sdots_1d[-100:])
                Stot_EPR_1D_blocks[deg].append(np.trapezoid(Sdots_1d - baseline_1d, taus_1d))
                Sdots_EPR[deg].append(Sdots_1d)
            else:
                Sdots_1d, taus_1d = EPR(Z_1d_formatted, dt, bins=int(np.sqrt(ntraj//n_blocks)), method=method)
                baseline_1d = 0
                Stot_EPR_1D_blocks[deg].append(np.trapezoid(Sdots_1d - baseline_1d, taus_1d))
                Sdots_EPR[deg].append(Sdots_1d)
            
    # Calculate Statistical Means and Errors for Projections
    Integrated_Stot_1D_Means = [np.mean(Stot_EPR_1D_blocks[deg]) for deg in angles_deg]
    Integrated_Stot_1D_Errs = [np.std(Stot_EPR_1D_blocks[deg]) / np.sqrt(n_blocks) for deg in angles_deg]
    Sdots = [Sdots_EPR[deg] for deg in angles_deg]

    print(f"\nTime: {time.time()-t0:.1f}s")
    print("\n--- Thermodynamic Trifecta Ledger ---")
    print(f"  Macrostate Delta F (X):      {Delta_F_X_macro:.3f} kT")
    print(f"  Macrostate Delta F (Y):      {Delta_F_Y_macro:.3f} kT")
    print("-----------------------------------")
    print(f"  Macrostate Delta F (2D):  {Delta_F_2D_macro:.3f} kT")

    # ==========================================
    # 3. Save Data
    # ==========================================
    save_filename = '2D_quartic_results_{}k_ntraj_{}_blocks_{}.npz'.format(int(ntraj//1000), n_blocks, method)
    print(f"\nSaving all results to '{save_filename}'...")

    np.savez_compressed(
        save_filename,
        SEED=SEED, A=A, K=K, D=D, dt=dt, nsteps=nsteps, ntraj=ntraj, n_blocks=n_blocks,
        angles_deg=angles_deg,
        F_x_start=F_x_start, F_x_end=F_x_end, Delta_F_X_macro=Delta_F_X_macro,
        F_y_start=F_y_start, F_y_end=F_y_end, Delta_F_Y_macro=Delta_F_Y_macro,
        F_2d_start=F_2d_start, F_2d_end=F_2d_end, Delta_F_2D_macro=Delta_F_2D_macro,
        Integrated_Stot_1D_Means=Integrated_Stot_1D_Means, Integrated_Stot_1D_Errs=Integrated_Stot_1D_Errs, Sdots=Sdots
    )
    
    print("Data successfully saved!")