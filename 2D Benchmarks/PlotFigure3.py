import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import folie as fl
from mlcolvar.utils.plot import muller_brown_potential, plot_metrics
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

# ==========================================
# 1. PRL PUBLICATION PLOTTING STANDARDS (2x2 Grid)
# ==========================================
# APS Double Column Width is 6.75 inches
fig_width = 6.75 
fig_height = 4.0 

# Standardized font sizes for APS journals
FONT_AXIS = 8
FONT_TICK = 8
FONT_LEGEND = 7
FONT_ANNOTATE = 8
FONT_PANEL = 10
LW_MAIN = 1.2
MARKER_SIZE = 30
CAP_SIZE = 3

# ==========================================
# 2. SHARED FUNCTIONS
# ==========================================
def plot_isolines_2D(
    function,
    component=None,
    limits=((-1.8, 1.2), (-0.4, 2.1)),
    num_points=(100, 100),
    mode="contourf",
    levels=12,
    cmap=None,
    colorbar=None,
    cbar_label=None,
    max_value=None,
    ax=None,
    allow_grad=False,
    ticksize=FONT_TICK,
    ftsize=FONT_AXIS,
    num_labels=5, 
    **kwargs,
):
    """Plot isolines of a function/model in a 2D space."""
    if type(num_points) == int:
        num_points = (num_points, num_points)
    xx = np.linspace(limits[0][0], limits[0][1], num_points[0])
    yy = np.linspace(limits[1][0], limits[1][1], num_points[1])
    xv, yv = np.meshgrid(xx, yy)

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
    else:
        z = function(xv, yv)

    if max_value is not None:
        z[z > max_value] = max_value

    if cmap is None:
        if mode == "contourf":
            cmap = "fessa"
        elif mode == "contour":
            cmap = None  
            if 'colors' not in kwargs:
                kwargs['colors'] = 'black'  
            if colorbar is None:
                colorbar = False

    if colorbar is None:
        if mode == "contourf":
            colorbar = True
        elif mode == "contour":
            colorbar = False

    if mode == "contourf":
        pp = ax.contourf(xv, yv, z, levels=levels, cmap=cmap, **kwargs)
        if colorbar:
            # Strictly Object-Oriented Colorbar implementation
            cbar = ax.figure.colorbar(pp, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=ticksize)
            if cbar_label is not None:
                cbar.set_label(cbar_label, fontsize=ftsize)
    else:
        pp = ax.contour(xv, yv, z, levels=levels, cmap=cmap, **kwargs)
        if num_labels > 0 and len(pp.levels) > 0:
            levels_to_label = pp.levels[:num_labels]
            ax.clabel(
                pp, 
                levels=levels_to_label, 
                inline=True, 
                fontsize=FONT_ANNOTATE, 
                colors=kwargs['colors']
            )
    
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.set_xlabel('x', fontsize=ftsize, labelpad=0)
    ax.set_ylabel('y', fontsize=ftsize, labelpad=0)   

def get_nearest_values_array(
    x_query, y_query, grid_array, 
    x_bounds=(-1.5, 1.5), y_bounds=(-1.5, 1.5), 
    logit=True, eps=1e-6
):
    """Safely computes logits by clamping strict 0 and 1 values."""
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

    val = (1 + grid_array[i, j]) / 2

    if logit:
        return np.clip(np.log(val/(1-val)), -10, 10)
    else:
        return val


# ==========================================
# 3. FIGURE SETUP (2x2 Grid)
# ==========================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height))
ax_q_pot, ax_q_epr = axes[0] # Top row: Quartic Double Well
ax_s_pot, ax_s_epr = axes[1] # Bottom row: S-shape Potential

ngrid = 100
a, b = 8, 40.0

# ==========================================
# 4. ROW 1: QUARTIC 2D 
# ==========================================
# Load Data
res_quartic = np.loadtxt('/home/sorbonne/ProductionEntropy/QuarticDoubleWell/data/sol_ongrid.dat')[:, 2].reshape((ngrid, ngrid))
eps_quartic = res_quartic[res_quartic > 0].min() / 1e2

def quartic_eval(x, y):
    return get_nearest_values_array(x, y, grid_array=res_quartic, y_bounds=(-1, 1), eps=eps_quartic)

quartic2d = fl.functions.Quartic2D(a=a, b=b)

# Panel (a): Quartic Potential Isolines
plot_isolines_2D(quartic2d.potential_plot, limits=((-1.5, 1.5), (-1, 1)), levels=np.arange(0, 13, 2), mode='contour', ax=ax_q_pot)
plot_isolines_2D(quartic_eval, ax=ax_q_pot, colorbar=True, levels=[0.0], mode='contour', 
                 linewidths=LW_MAIN, linestyles=':', limits=((-1.5, 1.5), (-1, 1)), 
                 num_labels=1, fmt={0.0: 'Transition state'})
plot_isolines_2D(quartic_eval, component=0, levels=50, ax=ax_q_pot, limits=((-1.5, 1.5), (-1, 1)), cbar_label=r'logit $\varphi$')

ts_line = Line2D([1.0], [0], color='black', linewidth=LW_MAIN, linestyle=':', label='Transition state')
ax_q_pot.legend(handles=[ts_line], loc='lower right', fontsize=FONT_LEGEND, frameon=False)

# -> Added White Circles for Minima here <-
ax_q_pot.plot([-1, 1], [0, 0], marker='o', color='white', markeredgecolor='black', linestyle='none', markersize=8, zorder=10)

ax_q_pot.text(-0.15, 1.1, '(a)', transform=ax_q_pot.transAxes, fontsize=FONT_PANEL, fontweight='bold', va='top')

# Panel (b): Quartic EPR
ntraj = 20000
n_blocks = 20
filename_q_results = f'./2D_Double_Well/2D_quartic_results_{int(ntraj//1000)}k_ntraj_{n_blocks}_blocks_biased.npz'
data_q = np.load(filename_q_results)
data_q_rss = np.load('/home/sorbonne/ProductionEntropy/QuarticDoubleWell/data/rss_committor_results.npz')

angles_deg = data_q['angles_deg']
Delta_F_X_macro = data_q['Delta_F_X_macro']
Delta_F_Y_macro = data_q['Delta_F_Y_macro']
Delta_F_2D_macro = data_q['Delta_F_2D_macro']
Integrated_Stot_1D_Means = data_q['Integrated_Stot_1D_Means']
rss_list_q = np.array(data_q_rss['rss_list']) / 300

rss_min_q, rss_max_q = rss_list_q.min(), rss_list_q.max()
x_fill_q = np.linspace(-5, rss_max_q * 1.05, 10)

ax_q_epr.axhline(y=Delta_F_2D_macro, color='tab:green', alpha=0.55)
ax_q_epr.text(1.10 * (rss_min_q + rss_max_q)/2, Delta_F_2D_macro * 1.03, "Direct estimation 2D", color='tab:green', va='center', fontsize=FONT_LEGEND)

ax_q_epr.scatter(rss_list_q, Integrated_Stot_1D_Means, color='blue', s=MARKER_SIZE, label='EPRI', zorder=2, alpha=0.55)
ax_q_epr.plot(0, Delta_F_X_macro, marker='x', linestyle='none', color='black', markersize=CAP_SIZE*2, markeredgewidth=LW_MAIN, label=r'Direct estimation 1D')
ax_q_epr.plot(rss_max_q, Delta_F_Y_macro, marker='x', color='black', markersize=CAP_SIZE*2, linestyle='none', markeredgewidth=LW_MAIN)

for i in range(len(rss_list_q)):
    offset, ha, va = (3, -3), 'left', 'top'
    lbl = f"$\\theta={angles_deg[i]}°$"
    if i == 0: 
        lbl += ", x"
        ax_q_epr.annotate(lbl, (rss_list_q[i]*3, Integrated_Stot_1D_Means[i]*1.05), textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=FONT_ANNOTATE)
    elif i == len(rss_list_q) - 1: 
        lbl += ", y"; ha, va = 'right', 'bottom'
        ax_q_epr.annotate(lbl, (rss_list_q[i], Integrated_Stot_1D_Means[i]*1.6), textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=FONT_ANNOTATE)
    else:
        ax_q_epr.annotate(lbl, (rss_list_q[i], Integrated_Stot_1D_Means[i]), textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=FONT_ANNOTATE)

ax_q_epr.set_xlabel(r'$\epsilon$', fontsize=FONT_AXIS, labelpad=0)
ax_q_epr.set_ylabel(r'$T\Delta S_{tot}$ ($k_BT$)', fontsize=FONT_AXIS, labelpad=0)
ax_q_epr.set_xlim(-0.02, rss_max_q * 1.05)
ax_q_epr.set_ylim(0, 10.9)
ax_q_epr.legend(loc='center right', fontsize=FONT_LEGEND, frameon=False)
ax_q_epr.tick_params(axis='both', labelsize=FONT_TICK)
ax_q_epr.text(-0.15, 1.1, '(b)', transform=ax_q_epr.transAxes, fontsize=FONT_PANEL, fontweight='bold', va='top')

# ==========================================
# 5. ROW 2: S-SHAPE (Tunnel) 
# ==========================================
# Load Data
res_sshape = np.loadtxt('/home/sorbonne/ProductionEntropy/Spotential/data/sol_ongrid.dat')[:, 2].reshape((ngrid, ngrid))
eps_sshape = res_sshape[res_sshape > 0].min()

def sshape_eval(x, y):
    return get_nearest_values_array(x, y, grid_array=res_sshape, y_bounds=(-1.5, 1.5), eps=eps_sshape)

quartic2dtunnel = fl.functions.Quartic2DTunnel(a=a, b=b)

# Panel (c): S-shape Potential Isolines
plot_isolines_2D(quartic2dtunnel.potential_plot, limits=((-1.5, 1.5), (-1.5, 1.5)), levels=np.arange(0, 20, 4), mode='contour', ax=ax_s_pot)
plot_isolines_2D(sshape_eval, ax=ax_s_pot, colorbar=True, levels=[0.0], mode='contour', linewidths=LW_MAIN, linestyles=':', limits=((-1.5, 1.5), (-1.5, 1.5)), num_labels=0)
plot_isolines_2D(sshape_eval, component=0, levels=50, ax=ax_s_pot, limits=((-1.5, 1.5), (-1.5, 1.5)), cbar_label=r'logit $\varphi$')

# -> Added White Circles for Minima here <-
ax_s_pot.plot([-1, 1], [0, 0], marker='o', color='white', markeredgecolor='black', linestyle='none', markersize=8, zorder=10)

nodes = [2, 6, 16]
colors = ['yellow', 'blue', 'red']
for k, i in enumerate(nodes):
    node = np.column_stack([np.linspace(-1, 1, i), np.sin((np.linspace(-1, 1, i) + 1)*np.pi)])
    ax_s_pot.plot(node[:, 0], node[:, 1], marker='o', markersize=CAP_SIZE*1.5, linewidth=LW_MAIN, color=colors[k], label=f'N={i}')

ax_s_pot.legend(loc='upper right', bbox_to_anchor=(0.8, 1.00), fontsize=FONT_LEGEND, frameon=False)
ax_s_pot.text(-0.15, 1.1, '(c)', transform=ax_s_pot.transAxes, fontsize=FONT_PANEL, fontweight='bold', va='top')

# Panel (d): S-shape EPR
data_s = np.load('./N_shape/sinusoidal_20k_ntraj_20_blocks_biased_results.npz')
data_s_rss = np.load('./N_shape/rss_committor_sinus_results.npz')

path_nodes_list = data_s['path_nodes_list']
Delta_F_2D_macro = data_s['Delta_F_2D_macro']
Path_M = data_s['Path_M']
Comm_M = data_s['Comm_M']
rss_list_s = np.array(data_s_rss['rss_list']) / 300

rss_min_s, rss_max_s = rss_list_s.min(), rss_list_s.max()
x_fill_s = np.linspace(-5, rss_max_s * 1.05, 10)

ax_s_epr.axhline(y=Delta_F_2D_macro, color='green', alpha=0.5)
ax_s_epr.text(1.12 * (rss_min_s + rss_max_s)/2, Delta_F_2D_macro * 1.025, "Direct estimation 2D", color='green', va='center', fontsize=FONT_LEGEND)

ax_s_epr.scatter(rss_list_s, Path_M, color='blue', s=MARKER_SIZE, label='EPRI path-CV', zorder=2, alpha=0.55)
ax_s_epr.scatter(0, Comm_M, color='blue', marker='D', s=MARKER_SIZE, label='EPRI committor', zorder=2, alpha=0.55)

for i in range(len(rss_list_s)):
    offset, ha, va = (3, -3), 'left', 'top'
    if i == 0: 
        offset, ha, va = (0, 1), 'right', 'bottom'
        ax_s_epr.annotate(f"$N = {path_nodes_list[i]}$", (rss_list_s[i], Path_M[i]*1.1), textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=FONT_ANNOTATE)
    else:
        ax_s_epr.annotate(f"$N = {path_nodes_list[i]}$", (rss_list_s[i], Path_M[i]), textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=FONT_ANNOTATE)

ax_s_epr.set_xlabel(r'$\epsilon$', fontsize=FONT_AXIS, labelpad=0)
ax_s_epr.set_ylabel(r'$T\Delta S_{tot}$($k_BT$)', fontsize=FONT_AXIS, labelpad=0)
ax_s_epr.set_xlim(-0.01, rss_max_s * 1.05)
ax_s_epr.set_ylim(0, 11.5)
ax_s_epr.legend(loc='center right', fontsize=FONT_LEGEND, frameon=False)
ax_s_epr.tick_params(axis='both', labelsize=FONT_TICK)
ax_s_epr.text(-0.15, 1.1, '(d)', transform=ax_s_epr.transAxes, fontsize=FONT_PANEL, fontweight='bold', va='top')

# ==========================================
# 6. FINAL FORMATTING & EXPORT
# ==========================================
fig.tight_layout(pad=0.2, w_pad=0, h_pad=-1)

# Ensure the subdirectories exist or save it to the current root directory
os.makedirs('./CombinedPlots', exist_ok=True)
fig.savefig('./CombinedPlots/Combined_Quartic_Sshape_PRL.png', dpi=600, bbox_inches='tight')

plt.show()

print("Successfully compiled and exported double-column 2x2 PRL figure.")