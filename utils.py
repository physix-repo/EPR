import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time
import warnings
import os


# ==========================================
# 1. Simulators
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
        
        X = X + D * force * dt + noise
        Z[:, t, 1] = X

    return Z

def simulate_2d_double_well_dirac(A, K, D, dt, nsteps, ntraj):
    """
    Generates overdamped Langevin trajectories in a 2D landscape from a Dirac start (0,0).
    X: Quartic Double Well | Y: Harmonic Well
    """
    X = np.zeros((ntraj, nsteps))
    Y = np.zeros((ntraj, nsteps))
    
    sqrt_2Ddt = np.sqrt(2.0 * D * dt)
    
    for t in range(1, nsteps):
        fx = -4.0 * A * X[:, t-1] * (X[:, t-1]**2 - 1.0)
        fy = -K * Y[:, t-1]
        
        noise_x = np.random.normal(0, sqrt_2Ddt, ntraj)
        noise_y = np.random.normal(0, sqrt_2Ddt, ntraj)
        
        X[:, t] = X[:, t-1] + D * fx * dt + noise_x
        Y[:, t] = Y[:, t-1] + D * fy * dt + noise_y
        
    return X, Y

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
# 2. Estimators
# ==========================================

def EPR(Z, dt, bins=100, method='biased'):
    """
    Computes the 1D EPR using a k-NN Histogram (Equi-populated Binning).
    Ensures uniform statistical power by grouping particles into chunks 
    of equal population rather than equal spatial width.
    'biased' method compute the biased estimator. The biased needs to be removed
    for every computation of total entropy production by estimating it with a
    time average at equilibrium.
    'unbiased' method remove the analytical bias.
    """
    ntraj, nsteps, _ = Z.shape
    
    # =======================================================
    # 1. Global Diffusion Profile D(x) (Static Spatial Grid)
    # =======================================================
    # Diffusion is a physical landscape property, so we still use the 
    # historical dataset on a fixed spatial grid to construct D(x).
    xmin, xmax = np.min(Z[:, :, 1]), np.max(Z[:, :, 1])
    dx = (xmax - xmin) / bins
    if dx == 0: dx = 1e-12
    
    displacements = Z[:, 1:, 1] - Z[:, :-1, 1]
    starts = Z[:, :-1, 1]
    
    ix_all = np.clip(np.floor((starts - xmin) / dx).astype(int), 0, bins - 1).ravel()
    disp_flat = displacements.ravel()
    
    counts = np.bincount(ix_all, minlength=bins)
    mask_D = counts > 10 
    
    sum_disp = np.bincount(ix_all, weights=disp_flat, minlength=bins)
    sum_disp_sq = np.bincount(ix_all, weights=disp_flat**2, minlength=bins)
    
    mean_disp = np.zeros(bins)
    mean_disp_sq = np.zeros(bins)
    
    mean_disp[mask_D] = sum_disp[mask_D] / counts[mask_D]
    mean_disp_sq[mask_D] = sum_disp_sq[mask_D] / counts[mask_D]
    
    var_disp = np.zeros(bins)
    var_disp[mask_D] = mean_disp_sq[mask_D] - mean_disp[mask_D]**2
    
    global_D_array = np.zeros(bins)
    global_D_array[mask_D] = var_disp[mask_D] / (2.0 * dt)
    
    if np.any(~mask_D):
        global_D_array[~mask_D] = np.var(disp_flat) / (2.0 * dt)
            
    # =======================================================
    # 2. k-NN Histogram Entropy Integration
    # =======================================================
    it_indices = np.arange(1, nsteps - 1, 1)
    Sdots = np.zeros(len(it_indices))
    taus = np.zeros(len(it_indices))

    for i, it in enumerate(it_indices):
        z_curr = Z[:, it, 1]
        v_curr = (Z[:, it + 1, 1] - Z[:, it - 1, 1]) / (2.0 * dt)
        
        # --- A. Sort Particles to create the k-NN distribution ---
        sort_idx = np.argsort(z_curr)
        z_sorted = z_curr[sort_idx]
        v_sorted = v_curr[sort_idx]
        
        # --- B. Split into Equi-populated Bins ---
        # array_split divides the N particles into exactly `bins` sub-arrays.
        # Each array will have N // bins particles (e.g., 5000 / 100 = 50 particles).
        z_knn_bins = np.array_split(z_sorted, bins)
        v_knn_bins = np.array_split(v_sorted, bins)
        
        Sdot_t = 0.0
        
        for b in range(bins):
            N_b = len(z_knn_bins[b])
            if N_b == 0: 
                continue
             
            # 1. Mean velocity of this k-NN chunk
            vel_mean = np.mean(v_knn_bins[b])
            
            # 2. Map this chunk's center of mass back to the global D(x) grid
            z_center = np.mean(z_knn_bins[b])
            ix_D = np.clip(np.floor((z_center - xmin) / dx).astype(int), 0, bins - 1)
            D_local = global_D_array[ix_D]
            
            # 3. Analytical Bias Subtraction 
            if method == 'biased':
                v_sq_unbiased = (vel_mean**2)
            if method == 'unbiased':
                expected_noise_variance = D_local / (N_b * dt)
                v_sq_unbiased = (vel_mean**2) - expected_noise_variance
            
            # 4. Integrate the entropy
            rho_prob = N_b / ntraj
            Sdot_t += rho_prob * (v_sq_unbiased) / D_local

        Sdots[i] = Sdot_t
        taus[i] = it * dt

    return Sdots, taus

def compute_macrostate_free_energy(rho, V, dx):
    """Numerically integrates the Free Energy Functional U[rho] - S[rho]"""
    U = np.sum(rho * V) * dx
    mask = rho > 0
    S = -np.sum(rho[mask] * np.log(rho[mask])) * dx
    return U - S, U, S






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

