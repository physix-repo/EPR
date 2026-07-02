import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time
import warnings
import os
from ..utils import EPR_optimized_knn_histogram

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. Functions
# ==========================================

def simulate_1d_double_well_dirac(A, D, dt, nsteps, ntraj):
    """Generates 1D Langevin trajectories from an exact Dirac start (x=0)."""
    Z = np.zeros((ntraj, nsteps, 2))
    Z[:, :, 0] = np.arange(nsteps) * dt  

    sqrt_2Ddt = np.sqrt(2.0 * D * dt)

    X = np.zeros(ntraj) 
    Z[:, 0, 1] = X

    for t in range(1, nsteps):
        force = -4.0 * A * X * (X**2 - 1.0)
        noise = np.random.normal(0, sqrt_2Ddt, ntraj)
        
        # CORRECT PHYSICS: Drift is force * dt (D is only in the noise)
        X = X + force * dt + noise
        Z[:, t, 1] = X

    return Z

# ==========================================
# 2. Execution
# ==========================================

if __name__ == "__main__":
    method = 'biased'
    A = 8.0
    D = 1.0
    dt = 0.001
    nsteps = 500
    ntraj = 200000 
    n_blocks = 20
    bins_thermo = int(np.sqrt(ntraj//n_blocks)) # Number of bins for the thermodynamic histogram

    print(f"--- Simulating Free Energy Relaxation from Dirac Start (A = {A}) ---")
    t0 = time.time()

    Z = simulate_1d_double_well_dirac(A, D, dt, nsteps, ntraj)

    Z_start = Z[:, 1, 1] 
    Z_end = Z[:, -1, 1]
    sigma_start = np.sqrt(2.0 * D * dt)

    # ----------------------------------------------------
    # Block Averaging for BOTH EPR and Empirical Thermo Estimation
    # ----------------------------------------------------
    print(f"\nComputing Block-Averaged Thermodynamics ({n_blocks} blocks of {ntraj//n_blocks} trajectories)...")
    Z_blocks = np.array_split(Z, n_blocks, axis=0)
    
    Sdots_blocks = []
    Stot_EPR_blocks = []
    Stot_Thermo_blocks = []

    for b, Z_b in enumerate(Z_blocks):
        # A. EPR Integral
        if method == 'biased':
            Sdots_b, taus = EPR_optimized_knn_histogram(Z_b, dt, bins=int(np.sqrt(ntraj/n_blocks)), stridet=1)
            #Sdots_b, taus = EPR_claude(Z_b, dt)
            #taus, Sdots_b = epr_plain(Z_b[:, :, 1], dt)
            Sdots_blocks.append(Sdots_b)
            
            baseline_b = np.mean(Sdots_b[-100:])
            Stot_EPR_blocks.append(np.trapezoid(Sdots_b - baseline_b, taus))
        else:
            #Sdots_b, taus = EPR_1D_unbiased(Z_b, dt, bins=50, stridet=1)
            #print(Z_b.shape)
            taus, Sdots_b = epr_plain(Z_b[:, :, 1], dt)
            Sdots_blocks.append(Sdots_b)
            
            baseline_b = 0
            Stot_EPR_blocks.append(np.trapezoid(Sdots_b - baseline_b, taus))

        # B. Direct Empirical Thermodynamics Estimation (Histogram)
        Z_start_b = Z_b[:, 1, 1]
        Z_end_b = Z_b[:, -1, 1]
        
        ln_rho_start_b = -0.5 * np.log(2.0 * np.pi * sigma_start**2) - 0.5 * (Z_start_b**2 / sigma_start**2)
        ln_rho_end_b, _, _ = get_ln_rho_histogram(Z_end_b, bins=bins_thermo)
        
        Q_b = (A * (Z_start_b**2 - 1.0)**2 - A * (Z_end_b**2 - 1.0)**2) / D
        dS_b = ln_rho_start_b - ln_rho_end_b
        Stot_Thermo_blocks.append(np.mean(Q_b + dS_b))

    Sdots_blocks = np.array(Sdots_blocks)
    Sdots_mean = np.mean(Sdots_blocks, axis=0)
    Sdots_std = np.std(Sdots_blocks, axis=0)
    
    Integrated_Stot_Mean = np.mean(Stot_EPR_blocks)
    Integrated_Stot_Err = np.std(Stot_EPR_blocks) / np.sqrt(n_blocks)

    Thermo_Stot_Mean = np.mean(Stot_Thermo_blocks)
    Thermo_Stot_Err = np.std(Stot_Thermo_blocks) / np.sqrt(n_blocks)

    # ----------------------------------------------------
    # Calculate Exact Analytical Results
    # ----------------------------------------------------
    def boltzmann_factor(x):
        return np.exp(-A * (x**2 - 1.0)**2 / D)
        
    Z_norm, _ = integrate.quad(boltzmann_factor, -np.inf, np.inf)
    V_mean_start = A * (3.0*sigma_start**4 - 2.0*sigma_start**2 + 1.0)
    S_start_mean = -0.5 * np.log(2.0 * np.pi * np.e * sigma_start**2)
    Exact_Delta_F = (V_mean_start/D + S_start_mean) - (-np.log(Z_norm))

    # ----------------------------------------------------
    # Trajectory Distributions (using Full Data Histogram)
    # ----------------------------------------------------
    ln_rho_start_full = -0.5 * np.log(2.0 * np.pi * sigma_start**2) - 0.5 * (Z_start**2 / sigma_start**2)
    ln_rho_end_full, hist_counts, hist_edges = get_ln_rho_histogram(Z_end, bins=bins_thermo)
    
    Delta_S_traj = ln_rho_start_full - ln_rho_end_full
    Q_traj = (A * (Z_start**2 - 1.0)**2 - A * (Z_end**2 - 1.0)**2) / D
    S_tot_traj = Q_traj + Delta_S_traj

    mean_Q = np.mean(Q_traj)
    mean_DeltaS = np.mean(Delta_S_traj)

    print(f"Time: {time.time()-t0:.1f}s")
    print("\n--- Results ---")
    print(f"  Mean Apparent Heat <Q>:      {mean_Q:.4f} kT")
    print(f"  Mean Sys Entropy <dS>:       {mean_DeltaS:.4f} kT")
    print("-----------------------------------")
    print(f"  1. Exact Analytical DF:      {Exact_Delta_F:.4f} kT")
    print(f"  2. Empirical Thermodynamic Stot:  {Thermo_Stot_Mean:.4f} +/- {Thermo_Stot_Err:.4f} kT")
    print(f"  3. Integrated EPR Stot:      {Integrated_Stot_Mean:.4f} +/- {Integrated_Stot_Err:.4f} kT")

    if method == 'biased':
        save_filename = 'block_averaged_1D_binning_200k_20_blocks_biased_results.npz'
    else:
        save_filename = 'block_averaged_1D_binning_500_unbiased_results.npz'
    print(f"\nSaving all results to '{save_filename}'...")
    
    np.savez_compressed(
        save_filename,
        A=A, D=D, dt=dt, nsteps=nsteps, ntraj=ntraj, n_blocks=n_blocks,
        taus=taus, Sdots_blocks=Sdots_blocks, Sdots_mean=Sdots_mean, Sdots_std=Sdots_std,
        Stot_EPR_blocks=Stot_EPR_blocks, Integrated_Stot_Mean=Integrated_Stot_Mean, Integrated_Stot_Err=Integrated_Stot_Err,
        Stot_Thermo_blocks=Stot_Thermo_blocks, Thermo_Stot_Mean=Thermo_Stot_Mean, Thermo_Stot_Err=Thermo_Stot_Err,
        Q_traj=Q_traj, Delta_S_traj=Delta_S_traj, S_tot_traj=S_tot_traj, Z_start=Z_start, Z_end=Z_end,
        mean_Q=mean_Q, mean_DeltaS=mean_DeltaS, Exact_Delta_F=Exact_Delta_F, Z_norm=Z_norm
    )
    print("Data successfully saved!")

    # ==========================================
    # 3. PLOTTING THE NORMALIZATION CHECK
    # ==========================================
    print("\nGenerating Distribution Comparison Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_eval = np.linspace(-2.5, 2.5, 1000)
    
    # 1. Empirical Histogram Bars
    ax.hist(Z_end, bins=bins_thermo, density=True, alpha=0.4, color='tab:blue', label='Empirical Histogram (Langevin endpoints)')
    
    # 2. Empirical Histogram Line (connecting bin centers to mimic a continuous PDF)
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    ax.plot(bin_centers, hist_counts, color='blue', linewidth=2, linestyle='-', label='Histogram Density Curve (Used for Stot)')
    
    # 3. Theoretical Boltzmann
    theo_pdf = boltzmann_factor(x_eval) / Z_norm
    ax.plot(x_eval, theo_pdf, color='red', linewidth=2, linestyle='--', label='Theoretical Boltzmann PDF')
    
    ax.set_title('Equilibrium Normalization Check: Theoretical vs Histogram Empirical', fontsize=14, pad=15)
    ax.set_xlabel('Position (x)', fontsize=12)
    ax.set_ylabel('Probability Density P(x)', fontsize=12)
    ax.legend(fontsize=11, loc='upper center')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    
    #plt.savefig('Boltzmann_Normalization_Histogram_Check.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'Boltzmann_Normalization_Histogram_Check.png'")