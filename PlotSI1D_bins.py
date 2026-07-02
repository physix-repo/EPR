import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# Configuration & LaTeX Formatting Standards
# ==========================================
BINS_LIST = ['10', '50', '100', 'squared']
LABELS = ['(a)', '(b)', '(c)', '(d)']

# Standardized LaTeX publication sizes (in inches)
FIG_WIDTH = 7.0   # Standard double-column text width
FIG_HEIGHT = 8.5  # Comfortable height for a full page without spilling over

# Standardized font sizes to match LaTeX documents
FONT_TICK = 9
FONT_AXIS = 9
FONT_LEGEND = 9
FONT_LABEL = 10
LW_MAIN = 1.0
CAP_SIZE = 3

def plot_combined_results():
    # Initialize the LaTeX-sized figure grid
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
    
    for i, b_val in enumerate(BINS_LIST):
        filename = f'/home/sorbonne/ProductionEntropy/1DDoubleWell/data/block_averaged_1D_biased_knn_{b_val}.npz'
        ax1 = axes[i, 0]  # Left column (Time evolution curves)
        ax2 = axes[i, 1]  # Right column (Bar plots)
        
        if not os.path.exists(filename):
            print(f"Warning: The file '{filename}' was not found. Skipping row {i}.")
            ax1.axis('off')
            ax2.axis('off')
            continue

        print(f"Loading data from '{filename}'...")
        data = np.load(filename)
        
        # Extract structural variables
        ntrajs = data['ntrajs']
        taus = data['taus']
        Exact_Delta_F = float(data['Exact_Delta_F'])
        
        # Extract plotting variables for the bar chart
        epr_means = data['epr_means']
        epr_sems = data['epr_sems']
        th_means = data['th_means']
        th_sems = data['th_sems']

        # Determine Row Label text
        if b_val == 'squared':
            row_label = r"$N_b = \sqrt{N_{traj}}$"
        else:
            row_label = fr"$N_b = {b_val}$"

        # ==========================================
        # Plot 1 (Left): Time Evolution of EPR (Sdot)
        # ==========================================
        for ntraj in ntrajs:
            Sdots_blocks = data[f'Sdots_blocks_{ntraj}']
            
            # Compute mean and standard deviation across the blocks
            Sd_mean = np.mean(Sdots_blocks, axis=0)[:500]
            Sd_std = np.std(Sdots_blocks, axis=0)[:500]
            
            ax1.plot(taus[:500], Sd_mean, lw=LW_MAIN, label=f"$N_{{traj}} = {ntraj}$")
            ax1.fill_between(taus[:500], Sd_mean - Sd_std, Sd_mean + Sd_std, alpha=0.2)

        # Log-scale for biased estimators
        if 'unbiased' not in filename.lower():
            ax1.set_yscale('log')
            
        ax1.tick_params(axis='both', labelsize=FONT_TICK)
        ax1.set_ylabel(r'$\dot{S}_{tot}(t)$', fontsize=FONT_AXIS)
        
        # Add Panel Label (a, b, c, d)
        ax1.text(-0.22, 1.08, LABELS[i], transform=ax1.transAxes, 
                 fontsize=FONT_LABEL, fontweight='bold', va='top')
                 
        # Add Row Label (Number of bins) vertically to the far left
        ax1.annotate(row_label, xy=(-0.30, 0.5), xycoords='axes fraction', 
                     fontsize=FONT_LABEL, fontweight='bold', va='center', ha='center', 
                     rotation=90, annotation_clip=False)
        
        # Only add X-axis label to the bottom row to keep it clean
        if i == 3:
            ax1.set_xlabel(r't', fontsize=FONT_AXIS)
            
        # Put legend on top of the left figure
        if i == 0:
            ax1.legend(
                fontsize=FONT_LEGEND, 
                loc='lower center', 
                bbox_to_anchor=(0.5, 1.05), # Anchors the legend just above the plot
                ncol=4,                     # Spreads the items horizontally
                frameon=False
            )

        # ==========================================
        # Plot 2 (Right): Total Integrated Entropy vs. Thermodynamics
        # ==========================================
        x_pos = np.arange(len(ntrajs))
        width = 0.35

        ax2.bar(x_pos - width/2, epr_means, width, yerr=epr_sems, capsize=CAP_SIZE,
               color='tab:blue', alpha=0.85, edgecolor='black', lw=LW_MAIN, label='EPRI')
        
        #ax2.bar(x_pos + width/2, th_means, width, yerr=th_sems, capsize=CAP_SIZE,
        #       color='tab:green', alpha=0.85, edgecolor='black', lw=LW_MAIN, label='Reference estimation')
        
        ax2.axhline(Exact_Delta_F, color='k', ls='--', lw=LW_MAIN*1.5,
                   label=rf'-$\Delta F = {Exact_Delta_F:.1f}$')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([str(ntraj) for ntraj in ntrajs])
        ax2.set_ylabel(r'$T\Delta S_{tot}$ [$k_B T$]', fontsize=FONT_AXIS)

        if i == 3:
            ax2.set_xlabel(r'$N_{traj}$', fontsize=FONT_AXIS)
            
        # Put legend on top of the right figure
        if i == 0:
            ax2.legend(
                fontsize=FONT_LEGEND, 
                loc='lower center', 
                bbox_to_anchor=(0.5, 1.05), # Anchors the legend just above the plot
                ncol=2,                     # Spreads the items horizontally
                frameon=False
            )
            
        ax2.tick_params(axis='both', labelsize=FONT_TICK)

    # ==========================================
    # Finalize & Save
    # ==========================================
    os.makedirs('/home/sorbonne/ProductionEntropy/1DDoubleWell/figs', exist_ok=True)
    save_path = '/home/sorbonne/ProductionEntropy/1DDoubleWell/figs/1DCompareMethods_knn_Combined.png'
    
    # Save using strictly Object-Oriented API (No plt.show())
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    
    print(f"Combined LaTeX figure successfully saved to '{save_path}'")

if __name__ == "__main__":
    plot_combined_results()