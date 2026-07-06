import numpy as np
import time
import warnings
import os
import sys
#from ..utils import EPR, simulate_1d_double_well_dirac

# 1. Dynamically add the parent directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

# 2. Now use an absolute import instead of a relative one
from utils import EPR, simulate_1d_double_well_dirac, compute_macrostate_free_energy

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. Execution
# ==========================================

if __name__ == "__main__":

    SEED = 42
    np.random.seed(SEED)

    method = 'biased' # Method for the EPR estimator
    A = 8.0 # Height of the barrier
    D = 1.0 # Diffusion constant
    dt = 0.001 # Timestep
    nsteps = 500 # Length of each trajectory
    ntraj = 20000 # Total number of trajectories
    n_blocks = 20 # Number of independent blocks 

    print(f"--- Simulating Free Energy Relaxation from Dirac Start (A = {A}) ---")
    t0 = time.time()

    Z = simulate_1d_double_well_dirac(A, D, dt, nsteps, ntraj)

    Z_start = Z[:, 1, 1] 
    Z_end = Z[:, -1, 1]

    # ----------------------------------------------------
    # Block Averaging for EPR
    # ----------------------------------------------------
    print(f"\nComputing Block-Averaged EPR ({n_blocks} blocks of {ntraj//n_blocks} trajectories)...")
    Z_blocks = np.array_split(Z, n_blocks, axis=0)
    
    Sdots_blocks = []
    Stot_EPR_blocks = []

    for b, Z_b in enumerate(Z_blocks):
        # EPR Integral
        if method == 'biased':
            Sdots_b, taus = EPR(Z_b, dt, bins=int(np.sqrt(ntraj/n_blocks)))
            Sdots_blocks.append(Sdots_b)
            
            baseline_b = np.mean(Sdots_b[-100:])
            Stot_EPR_blocks.append(np.trapezoid(Sdots_b - baseline_b, taus))
        else:
            taus, Sdots_b = EPR(Z_b, dt, bins=int(np.sqrt(ntraj/n_blocks)), method=method)
            Sdots_blocks.append(Sdots_b)
            
            baseline_b = 0
            Stot_EPR_blocks.append(np.trapezoid(Sdots_b - baseline_b, taus))

    Sdots_blocks = np.array(Sdots_blocks)
    Sdots_mean = np.mean(Sdots_blocks, axis=0)
    Sdots_std = np.std(Sdots_blocks, axis=0)
    
    Integrated_Stot_Mean = np.mean(Stot_EPR_blocks)
    Integrated_Stot_Err = np.std(Stot_EPR_blocks) / np.sqrt(n_blocks)

    # ----------------------------------------------------
    # Calculate Macrostate Free Energy (Delta F)
    # ----------------------------------------------------
    print("Computing Macrostate Free Energy Drop (Delta F)...")
    
    # Establish a spatial grid encompassing both start and end distributions
    z_min = min(np.min(Z_start), np.min(Z_end)) - 1.0
    z_max = max(np.max(Z_start), np.max(Z_end)) + 1.0
    bins_thermo = int(np.sqrt(ntraj))
    
    # Compute probability densities for the start and end macrostates
    rho_start, bin_edges = np.histogram(Z_start, bins=bins_thermo, range=(z_min, z_max), density=True)
    rho_end, _ = np.histogram(Z_end, bins=bins_thermo, range=(z_min, z_max), density=True)
    
    # Define spatial points and the effective potential V(x) corresponding to the simulation
    dx = bin_edges[1] - bin_edges[0]
    x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    V_x = (A * (x_centers**2 - 1.0)**2) / D
    
    F_start, U_start, S_start = compute_macrostate_free_energy(rho_start, V_x, dx)
    F_end, U_end, S_end = compute_macrostate_free_energy(rho_end, V_x, dx)
    Delta_F = F_start - F_end
    Delta_U = U_start - U_end
    Delta_S = S_start - S_end

    # ----------------------------------------------------
    # Results and Archiving
    # ----------------------------------------------------
    print(f"Time: {time.time()-t0:.1f}s")
    print("\n--- Results ---")
    print(f"  Estimated F_start:           {F_start:.3f} kT")
    print(f"  Estimated F_end:             {F_end:.3f} kT")
    print(f"  Free Energy Drop (Delta F):   {Delta_F:.3f} kT")
    print(f"  Mean Energy Change (Delta U):   {Delta_U:.3f} kT")
    print(f"  Mean Entropy Change (T.Delta S):   {Delta_S:.3f} kT")
    print("-----------------------------------")
    print(f"  Integrated EPR Stot:         {Integrated_Stot_Mean:.3f} +/- {Integrated_Stot_Err:.3f} kT")

    save_filename = f'block_averaged_1D_{ntraj}_ntraj_{n_blocks}_blocks_{method}_results.npz'
    print(f"\nSaving all results to '{save_filename}'...")

    np.savez_compressed(
        save_filename,
        SEED=SEED, A=A, D=D, dt=dt, nsteps=nsteps, ntraj=ntraj, n_blocks=n_blocks, method=method,
        taus=taus, Sdots_blocks=Sdots_blocks, Sdots_mean=Sdots_mean, Sdots_std=Sdots_std,
        Stot_EPR_blocks=Stot_EPR_blocks, Integrated_Stot_Mean=Integrated_Stot_Mean, Integrated_Stot_Err=Integrated_Stot_Err,
        Z_start=Z_start, Z_end=Z_end,
        F_start=F_start, F_end=F_end, Delta_F=Delta_F, 
        U_start=U_start, U_end=U_end, Delta_U=Delta_U, 
        S_start=S_start, S_end=S_end, Delta_S=Delta_S
    )
    print("Data successfully saved!")
    
