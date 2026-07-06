import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
import os

# ==========================================
# 1. Load the Saved Data
# ==========================================
method = 'biased'
ntraj = 20000
n_blocks = 20
filename = f'block_averaged_1D_{ntraj}_ntraj_{n_blocks}_blocks_{method}_results.npz'

if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found in the current directory.")
    print("Please run the simulation script first to generate the data.")
    exit()

print(f"Loading data from '{filename}'...")
data = np.load(filename)

# Parameters
A = data['A']
D = data['D']
dt = data['dt']
n_blocks = data['n_blocks']
ntraj = data['ntraj']/n_blocks

# Time Series
taus = data['taus']
Sdots_blocks = data['Sdots_blocks']
Sdots_mean = data['Sdots_mean']
Sdots_std = data['Sdots_std']

# Helper for theoretical distribution
sigma_start = np.sqrt(2.0 * D * dt)
Z_end = data['Z_end']

def boltzmann_factor(x):
    return np.exp(-A * (x**2 - 1.0)**2 / D)

Z_norm, _ = integrate.quad(boltzmann_factor, -np.inf, np.inf)

# ==========================================
# 2. PRL PUBLICATION PLOTTING STANDARDS (2x1 Stack)
# ==========================================
# PRL Single Column Width is ~3.375 inches (8.6 cm)
fig_width = 3.6 
# Total height is 2 times the standard single-plot height
fig_height = fig_width * (2 / 5) * 2 

# Standardized font sizes for APS journals
FONT_AXIS = 8
FONT_TICK = 8
FONT_LEGEND = 7
FONT_PANEL = 10
LW_MAIN = 1.2

# Create a figure with 2 rows and 1 column
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height))
ax1, ax_sdot = axes

# ----------------------------------------------------
# PANEL (a): Free Energy Relaxation 
# ----------------------------------------------------
x_eval = np.linspace(-1.5, 1.5, 400)
V_eval = A * (x_eval**2 - 1.0)**2

color_V = 'black'
ax1.plot(x_eval, V_eval, color=color_V, lw=LW_MAIN, label=r'$F(x)$')
ax1.set_xlabel('x', fontsize=FONT_AXIS, labelpad=0)
ax1.set_ylabel(r'Free energy ($k_B T$)', color=color_V, fontsize=FONT_AXIS)
ax1.tick_params(axis='y', labelcolor=color_V, labelsize=FONT_TICK)
ax1.tick_params(axis='x', labelsize=FONT_TICK)
ax1.set_ylim(0, A + 2)

ax1_twin = ax1.twinx()
color_P = 'tab:blue'

# Initial Dist
rho_start = np.exp(-0.5 * (x_eval**2 / sigma_start**2)) / np.sqrt(2 * np.pi * sigma_start**2)
ax1_twin.plot(x_eval, rho_start, color='red', linestyle='--', lw=LW_MAIN, label=r'$\rho_{\mathrm{begin}}$')

# Exact Gibbs Dist
rho_gibbs = boltzmann_factor(x_eval) / Z_norm
ax1_twin.plot(x_eval, rho_gibbs, color='#8B8589', linestyle='--', lw=LW_MAIN, label=r'$e^{-\beta F}/Z$')

# Empirical Final Dist
kde_full = gaussian_kde(Z_end)
rho_end = kde_full(x_eval)
ax1_twin.hist(Z_end, color=color_P, density=True, bins=100, alpha=0.3, label=r'$\rho_{\mathrm{end}}$')

ax1_twin.set_ylabel('Probability density', color=color_P, fontsize=FONT_AXIS)
ax1_twin.tick_params(axis='y', labelcolor=color_P, labelsize=FONT_TICK)
ax1_twin.set_ylim(0, max(np.max(rho_start), np.max(rho_gibbs)) * 1.1)

# SPLIT LEGEND LOGIC
h1, l1 = ax1.get_legend_handles_labels()       # Contains F(x) at index 0
h2, l2 = ax1_twin.get_legend_handles_labels()  # Contains rho_init(0), rho_gibbs(1), rho_final(2)

# Top Left: F(x) and Gibbs Dist
ax1.legend([h1[0], h2[1]], [l1[0], l2[1]], loc='upper left', fontsize=FONT_LEGEND, frameon=False)

# Top Right: Empirical Distributions (Initial and Final)
ax1_twin.legend([h2[0], h2[2]], [l2[0], l2[2]], loc='upper right', fontsize=FONT_LEGEND, frameon=False)
# Add Panel Label (a)
ax1.text(-0.35, 1.3, '(a)', transform=ax1.transAxes, fontsize=FONT_PANEL, fontweight='bold', va='top')

# ----------------------------------------------------
# PANEL (b): Block-Averaged Semi-Log EPR 
# ----------------------------------------------------
ax_sdot.plot(taus, Sdots_mean, 'purple', lw=LW_MAIN)
ax_sdot.fill_between(taus, Sdots_mean + Sdots_std, Sdots_mean - Sdots_std, color='purple', alpha=0.3)

#ax_sdot.set_yscale('log') 

ax_sdot.axhline(np.sqrt(ntraj)/dt/ntraj, linestyle='--', lw=LW_MAIN, color='black', label=r'Baseline = $\frac{1}{N_b \Delta t}$')
ax_sdot.set_ylabel(r'$\dot{S}_{tot}(t)$', fontsize=FONT_AXIS)
ax_sdot.set_xlabel('t', fontsize=FONT_AXIS, labelpad=0)
ax_sdot.legend(loc='upper right', fontsize=FONT_LEGEND, frameon=False)
ax_sdot.tick_params(axis='both', labelsize=FONT_TICK)
ax_sdot.set_xlim(-0.02, 0.5)

# Add Panel Label (b)
ax_sdot.text(-0.35, 1.2, '(b)', transform=ax_sdot.transAxes, fontsize=FONT_PANEL, fontweight='bold', va='top')

# ----------------------------------------------------
# FINAL FORMATTING AND EXPORT
# ----------------------------------------------------
# Force the left y-axis labels to align vertically perfectly
fig.align_ylabels([ax1, ax_sdot])

# Adjust layout so subplots don't overlap
fig.tight_layout(pad=0, h_pad=-1)
plt.show()
