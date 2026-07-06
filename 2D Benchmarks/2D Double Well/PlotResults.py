import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import folie as fl
from mlcolvar.utils.plot import muller_brown_potential, plot_isolines_2D, plot_metrics
import torch

ngrid = 100
res = np.loadtxt('/home/sorbonne/ProductionEntropy/QuarticDoubleWell/sol_ongrid.dat')[:, 2].reshape((ngrid, ngrid))

def plot_isolines_2D(
    function,
    component=None,
    limits=((-1.8, 1.2), (-0.4, 2.1)),
    num_points=(100, 100),
    mode="contourf",
    levels=12,
    cmap=None,
    colorbar=None,
    max_value=None,
    ax=None,
    allow_grad=False,
    ticksize=10,
    ftsize=10,
    num_labels=5, # Added parameter to control how many isolines get labeled
    **kwargs,
):
    """Plot isolines of a function/model in a 2D space."""

    # Define grid where to evaluate function
    if type(num_points) == int:
        num_points = (num_points, num_points)
    xx = np.linspace(limits[0][0], limits[0][1], num_points[0])
    yy = np.linspace(limits[1][0], limits[1][1], num_points[1])
    xv, yv = np.meshgrid(xx, yy)

    # if torch module
    if isinstance(function, torch.nn.Module):
        z = np.zeros_like(xv)
        for i in range(num_points[0]):
            for j in range(num_points[1]):
                xy = torch.Tensor([xv[i, j], yv[i, j]])
                if allow_grad:
                    s = function(xy.unsqueeze(0)).squeeze(0).detach().numpy()
                else:
                    with torch.no_grad():
                        train_mode = function.training
                        function.eval()
                        s = function(xy.unsqueeze(0)).squeeze(0).numpy()
                        function.training = train_mode
                if component is not None:
                    s = s[component]
                z[i, j] = np.squeeze(s)
    # else apply function directly to grid points
    else:
        z = function(xv, yv)

    if max_value is not None:
        z[z > max_value] = max_value

    # Setup plot
    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(6, 4.0), dpi=100)

    # Color scheme
    if cmap is None:
        if mode == "contourf":
            cmap = "fessa"
        elif mode == "contour":
            cmap = None  # Force cmap to None to prevent colormap mapping
            if 'colors' not in kwargs:
                kwargs['colors'] = 'black'  # Force all isolines to be solid white
            if colorbar is None:
                colorbar = False

    # Colorbar
    if colorbar is None:
        if mode == "contourf":
            colorbar = True
        elif mode == "contour":
            colorbar = False

    # Plot
    if mode == "contourf":
        pp = ax.contourf(xv, yv, z, levels=levels, cmap=cmap, **kwargs)
        if colorbar:
            cbar = plt.colorbar(pp, ax=ax)
            cbar.ax.tick_params(labelsize=ticksize)
    else:
        pp = ax.contour(xv, yv, z, levels=levels, cmap=cmap, **kwargs)

        # Add labels to a specific number of isolines
        if num_labels > 0 and len(pp.levels) > 0:
            # Slice the levels array to grab only the first 'num_labels' values
            levels_to_label = pp.levels[:num_labels]
            
            # Draw the labels inline, matching the color of the contour lines
            ax.clabel(
                pp, 
                levels=levels_to_label, 
                inline=True, 
                fontsize=ftsize+2, # Slightly smaller than axis fonts for readability
                colors=kwargs['colors']
            )
    
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.set_xlabel('x', fontsize=ftsize)
    ax.set_ylabel('y', fontsize=ftsize)   

    if return_axs:
        return ax
    else:
        return None



def get_nearest_values_array(
    x_query, 
    y_query, 
    grid_array=res, 
    x_bounds=(-1.5, 1.5), 
    y_bounds=(-1, 1), 
    logit=True,
    eps=res[res>0].min()/(1e2)  # Epsilon factor to prevent log(0) and division by zero
):
    """
    Finds the values of the closest grid points for arrays of coordinates.
    Safely computes logits by clamping strict 0 and 1 values.
    
    Parameters:
    - x_query, y_query: 1D numpy arrays or lists of the coordinates you want to look up.
    - grid_array: 2D numpy array of values. shape = (nx, ny)
    - x_bounds: Tuple of (x_min, x_max) of the grid.
    - y_bounds: Tuple of (y_min, y_max) of the grid.
    - logit: Boolean flag to apply logit transformation.
    - eps: Small float to clamp probabilities away from absolute 0 or 1.

    Returns:
    - A numpy array of values corresponding to the queried (x, y) pairs.
    """
    nx, ny = grid_array.shape
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    # Calculate grid spacing
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    # Ensure inputs are numpy arrays for element-wise math
    x_q = np.asarray(x_query)
    y_q = np.asarray(y_query)
    
    # Calculate indices for all points simultaneously
    i = np.round((x_q - x_min) / dx).astype(int)
    j = np.round((y_q - y_min) / dy).astype(int)
    
    # Clamp all indices to the array boundaries at once to prevent IndexErrors
    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)

    # Calculate base probability mapping
    val = (1 + grid_array[i, j]) / 2

    if logit:
        # Smooth interpolation at the ill zones
        #val_clipped = np.clip(val, eps, 1.0 - eps)
        #return np.log(val_clipped / (1.0 - val_clipped))
        return np.clip(np.log(val/(1-val)), -10, 10)
    else:
        return val



# Potential Model

a,b = 8, 40.0
quartic2d= fl.functions.Quartic2D(a=a,b=b)

n_components=1
fig,axs = plt.subplots( 1, n_components, figsize=(8*n_components,6) )
if n_components == 1:
    axs = [axs]
for i in range(n_components):
    ax = axs[i]
    u = plot_isolines_2D(quartic2d.potential_plot,limits=((-1.5, 1.5), (-1, 1)), levels=np.arange(0, 13, 2),mode='contour',ax=ax, return_axs=True, ticksize=20, fontisze=30)
    plot_isolines_2D(get_nearest_values_array, ax=ax, colorbar=True, 
                    levels=[0.0], 
                     mode='contour', 
                     linewidths=2, 
                     limits=((-1.5, 1.5), (-1, 1)),
                    num_points=(ngrid, ngrid), ticksize=20, fontisze=30, num_labels=0)
    plot_isolines_2D(get_nearest_values_array, 
                     component=i, levels=50, ax=ax, limits=((-1.5, 1.5), (-1, 1)), 
                     num_points=(ngrid, ngrid), ticksize=20, fontisze=30)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)  
    #plot_isolines_2D(model, component=i, mode='contour', levels=25, ax=ax)
plt.tight_layout()
plt.savefig('./QuarticDoubleWell/LogitCommittor.png')
plt.show()

# ==========================================
# 1. Load the Saved Data
# ==========================================
ntraj = 200000
n_blocks = 20
#filename = 'trifecta_2D_meshless_results_full.npz'
#filename = 'trifecta_2D_meshless_results_20k_10_blocks_GMM_biased_full.npz'
filename = 'quartic_results_{}k_{}_blocks_GMM_knn_squared_biased.npz'.format(int(ntraj//1000), n_blocks)
if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found in the current directory.")
    print("Please run the 2D meshless simulation script first to generate the data.")
    exit()

print(f"Loading data from '{filename}'...")
data = np.load(filename)


# Extract Data
angles_deg = data['angles_deg']

# Exact Theoretical Limits
Exact_Delta_F_X = data['Exact_Delta_F_X']
Exact_Delta_F_Y = data['Exact_Delta_F_Y']
Exact_Delta_F_2D = data['Exact_Delta_F_2D']

Analytical_Stot_2D = data['Analytical_Stot_2D']
Analytical_Stot_X = data['Analytical_Stot_X']
Analytical_Stot_Y = data['Analytical_Stot_Y']

# 2D Meshless EPR Results
Integrated_Stot_2D_Mean = data['Integrated_Stot_2D_Mean']
Integrated_Stot_2D_Err = data['Integrated_Stot_2D_Err']

# 1D Empirical Thermo Results
Thermo_Stot_X_Mean = data['Thermo_Stot_X_Mean']
Thermo_Stot_X_Err = data['Thermo_Stot_X_Err']
Thermo_Stot_Y_Mean = data['Thermo_Stot_Y_Mean']
Thermo_Stot_Y_Err = data['Thermo_Stot_Y_Err']

# 2D Empirical Thermo Results (KDE)
Thermo_Stot_2D_Mean = data['Thermo_Stot_2D_Mean']
Thermo_Stot_2D_Err = data['Thermo_Stot_2D_Err']

# 1D Grid-Projected Results
Integrated_Stot_1D_Means = data['Integrated_Stot_1D_Means']
Integrated_Stot_1D_Errs = data['Integrated_Stot_1D_Errs']

filename = 'rss_committor_results.npz'

if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found in the current directory.")
    print("Please run the 2D meshless simulation script first to generate the data.")
    exit()

data = np.load(filename)

# Extract Data
rss_list = np.array(data['rss_list'])/300 # normalize RSS by the number of points used in the fitting


print(f"Loading data from '{filename}'...")

print("Data loaded successfully. Generating plot...")

# ==========================================
# 2. PLOTTING
# ==========================================


fig, ax = plt.subplots(figsize=(10, 6))
capsize=18

# Plot the full 2D Trifecta Ceilings
#ax.axhline(Exact_Delta_F_2D, color='black', linestyle='-', linewidth=2)
#ax.axhline(Analytical_Stot_2D, color='black', linestyle='-', linewidth=2)
#ax.text((rss_list.min() + rss_list.max())/2, (Exact_Delta_F_2D)*1.01, "Exact $\Delta F$", color='black', va='center', fontsize=20)
#ax.text(0.90*(rss_list.min() + rss_list.max())/2, (Analytical_Stot_2D)*1.01, "Exact $\Delta S_{{tot}}$", color='black', va='center', fontsize=20)

# 1D Thermo Direct Estimation
#ax.plot(0, Analytical_Stot_X, marker='s', linestyle='none', color='black', markersize=capsize, markerfacecolor='none', markeredgewidth=2, label='$T\Delta S_{{tot, exact}}$')
#ax.plot(rss_list.max(), Analytical_Stot_Y, marker='s', color='black', markersize=capsize, linestyle='none', markerfacecolor='none', markeredgewidth=2)#, label='$T\Delta S_{{tot, Exact}}(Y)$')

# 2D Thermo KDE
ax.fill_between(np.linspace(-5,  np.array(rss_list).max()*1.05, 10), 
                Thermo_Stot_2D_Mean - Thermo_Stot_2D_Err, 
                Thermo_Stot_2D_Mean + Thermo_Stot_2D_Err, 
                color='tab:green', alpha=0.55)
ax.text(1.20*(rss_list.min() + rss_list.max())/2, (Thermo_Stot_2D_Mean)*1.02, "Direct estimation 2D", color='tab:green', va='center', fontsize=20)

# 2D Meshless EPR
#ax.axhline(Integrated_Stot_2D_Mean, color='red', linestyle='--', linewidth=2)
ax.text(1.20*(rss_list.min() + rss_list.max())/2, (Integrated_Stot_2D_Mean - Integrated_Stot_2D_Err)*0.965, "EPRI 2D", color='blue', va='center', fontsize=20)
ax.fill_between(np.linspace(-5, np.array(rss_list).max()*1.05, 10), 
                Integrated_Stot_2D_Mean - Integrated_Stot_2D_Err, 
                Integrated_Stot_2D_Mean + Integrated_Stot_2D_Err, 
                color='blue', alpha=0.55)

# Plot the 1D theoretical limits
#ax.axhline(Exact_Delta_F_X, color='blue', linestyle=':', alpha=0.7, label=r'$\Delta F_X$')
#ax.axhline(Exact_Delta_F_Y, color='tab:orange', linestyle=':', alpha=0.7, label=r'$\Delta F_Y$')


# Plot the measured 1D projection data

style = 'colorless'
errs=False

if style == 'colorless':

        # 1. Convert to arrays and clean data
    yerr = np.array(Integrated_Stot_1D_Errs)
    y_data = np.array(Integrated_Stot_1D_Means)
    x_data = np.array(rss_list)

    # 2. CRITICAL FIX: Force the axis limits to update BEFORE transforming
    # We temporarily update the data limits so transData knows the correct scale
    ax.update_datalim(np.column_stack([x_data, y_data]))
    ax.autoscale_view()
    markersize = 16
    marker_radius_pts = markersize / 2

    # 4. Conversion points -> pixels
    dpi = fig.dpi
    marker_radius_px = marker_radius_pts * dpi / 72.0

    # 5. Transformation pixels -> data coordinates
    # FIX: Use the median of your data as the anchor, not (0,0), for scale stability
    x_anchor = np.nanmedian(x_data)
    y_anchor = np.nanmedian(y_data)

    p0 = ax.transData.transform((x_anchor, y_anchor))
    p1 = (p0[0], p0[1] + marker_radius_px)

    inv = ax.transData.inverted()
    y0 = inv.transform(p0)[1]
    y1 = inv.transform(p1)[1]

    threshold = abs(y1 - y0)
    print(f"Calculated Data Threshold for Marker Radius: {threshold:.4f}")
    # masque : ne garder que les erreurs au-dessus du seuil
    yerr_filtered = np.where(yerr > threshold, yerr, np.nan)

    
    if errs:
        ax.errorbar(
            rss_list,
            Integrated_Stot_1D_Means,
            yerr=yerr_filtered,
            fmt='o',
            color='blue',
            markerfacecolor='blue',
            markeredgewidth=1.5,
            markersize=16,
            ecolor='crimson',
            elinewidth=1.5,
            capsize=4,
            label='EPRI',
            zorder=2,
            alpha=0.55
        )
    else:
        ax.scatter(
            rss_list,
            Integrated_Stot_1D_Means,
            color='blue',
            s=16**2,              # taille = aire du marqueur
            label='EPRI',
            zorder=2,
            alpha=0.55
        )
    # 1D Thermo Direct Estimation
    ax.plot(0, Analytical_Stot_X, marker='x', linestyle='none', color='black', markersize=capsize, markerfacecolor='none', markeredgewidth=2, label='$T\Delta S_{{tot, exact}}$')
    ax.plot(rss_list.max(), Analytical_Stot_Y, marker='x', color='black', markersize=capsize, linestyle='none', markerfacecolor='none', markeredgewidth=2)#, label='$T\Delta S_{{tot, Exact}}(Y)$')


if style == 'color':

    x = np.array(rss_list)
    y = np.array(Integrated_Stot_1D_Means)
    c = np.array(Integrated_Stot_1D_Errs)  # array utilisé pour la couleur

    from matplotlib.colors import LogNorm

    sc = ax.scatter(
        x,
        y,
        c=c,                # couleur dépend de cet array
        cmap='viridis',
        norm=LogNorm(vmin=np.min(c), vmax=np.max(c)),     # colormap (tu peux changer)
        s=200,              # taille des points
        edgecolors='black',
        label='EPRI'
    )
    # barre de couleur
    cbar = plt.colorbar(sc)
    cbar.set_label('error ($k_BT$)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    #ax.errorbar(angles_deg, Integrated_Stot_1D_Means, yerr=Integrated_Stot_1D_Errs, fmt='o-', capsize=5, 
    #            color='purple', markersize=8, linewidth=2, label=r'Apparent $S_{tot}$ (1D Grid Projection)')

    # Annotate each points 

    # 1D Thermo Direct Estimation
    ax.plot(0, Analytical_Stot_X, marker='s', linestyle='none', color='black', markersize=capsize, markerfacecolor='none', markeredgewidth=2, label='$T\Delta S_{{tot, exact}}$')
    ax.plot(rss_list.max(), Analytical_Stot_Y, marker='s', color='black', markersize=capsize, linestyle='none', markerfacecolor='none', markeredgewidth=2)#, label='$T\Delta S_{{tot, Exact}}(Y)$')


for i in range(len(rss_list)):
    if i == 0:
        plt.annotate(
                f"$\\theta = {angles_deg[i]}°, X$",
                (rss_list[i]+0.01, Integrated_Stot_1D_Means[i]+0.2),
                textcoords="offset points",
                xytext=(6, -6), 
                ha='left',
                va='top',
                fontsize=20
            )
    elif i==len(rss_list)-1:
        plt.annotate(
                f"$\\theta = {angles_deg[i]}°, Y$",
                (rss_list[i], Integrated_Stot_1D_Means[i]+0.5),
                textcoords="offset points",
                xytext=(6, -6), 
                ha='right',
                va='bottom',
                fontsize=20
            )
    else:
        plt.annotate(
                f"$\\theta = {angles_deg[i]}°$",
                (rss_list[i], Integrated_Stot_1D_Means[i]),
                textcoords="offset points",
                xytext=(6, -6), 
                ha='left',
                va='top',
                fontsize=20
            )

# Formatting and Aesthetics
#ax.set_title('Thermodynamic Trifecta: 1D Projection vs Full 2D Entropy limits', fontsize=14, pad=15)
ax.set_xlabel(r'$\epsilon$', fontsize=20)
ax.set_ylabel(r'$T\Delta S_{tot}$ ($k_BT$)', fontsize=20)

#ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='center right', fontsize=20)
ax.set_xlim(-0.01, np.array(rss_list).max()*1.05)
#ax.set_xscale('log')\
ax.tick_params(axis='both', labelsize=20)

# Add a slight padding to the Y axis to accommodate the new lines cleanly
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 1.05)

plt.tight_layout()
plt.savefig('./QuarticDoubleWell/EPRFinal{}_biased.png'.format(style))
plt.show()
