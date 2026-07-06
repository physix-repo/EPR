import numpy as np
import time 
import os
import warnings
import sys
from utils import EPR_optimized_knn_histogram


# 1. Dynamically add the parent directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# 2. Now use an absolute import instead of a relative one
from utils import EPR

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. Functions & Estimators
# ==========================================

def load_colvar_files(file_paths):
    """Loads, splits, and merges trajectories from multiple COLVAR files."""
    trajs = {k: [] for k in ['cc', 'd', 'sc', 'sw', 'c2w', 'c1w', 'ucc', 'ucw']}
    
    for file in file_paths:
        print(f"Loading data from {file}...")
        CVdata = np.loadtxt(file)
        t = CVdata[:, 0]
        
        # Identify trajectory boundaries where time resets
        split_indices = np.where(np.diff(t) < 0)[0] + 1
        splits = np.split(CVdata, split_indices)
        
        for trj in splits:
            trajs['cc'].append(trj[:, 1])
            trajs['d'].append(trj[:, 2])
            trajs['sc'].append(trj[:, 3])
            trajs['sw'].append(trj[:, 4])
            trajs['c2w'].append(trj[:, 5])
            trajs['c1w'].append(trj[:, 6])
            trajs['ucc'].append(trj[:, 7])
            trajs['ucw'].append(trj[:, 8])
            
    return trajs

def filter_trajectories(trajs):
    """Removes trajectories that are incomplete or fail to reach the target state."""
    length = len(trajs['d'][0])
    keys = list(trajs.keys())
    
    valid_indices = []
    for i in range(len(trajs['d'])):
        # Must be full length and end in the associated state (d <= 1.1)
        if len(trajs['d'][i]) == length and trajs['d'][i][-1] <= 1.1:
            valid_indices.append(i)
            
    filtered_trajs = {k: [trajs[k][i] for i in valid_indices] for k in keys}
    print(f"Filtered trajectories: {len(valid_indices)} out of {len(trajs['d'])} retained.")
    return filtered_trajs

def format_trajectories(traj_list, dt):
    """Converts a list of 1D trajectories into a (ntraj, nsteps, 2) array."""
    positions = np.array(traj_list)
    ntraj, nsteps = positions.shape
    Z = np.zeros((ntraj, nsteps, 2))
    Z[:, :, 0] = np.arange(nsteps) * dt
    Z[:, :, 1] = positions
    return Z

# ==========================================
# 2. Main Execution Pipeline
# ==========================================

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    t_global = time.time()
    
    # 1. Data Loading & Filtering
    files = [
        './Fullerenes_shooting_cv.dat'
    ]
    
    raw_trajs = load_colvar_files(files)

    clean_trajs = filter_trajectories(raw_trajs)
    
    # 2. Configuration
    method = 'biased'
    max_trajectories = 20000  
    stride = 10
    dt = 0.1 * stride
    n_blocks = 5
    bins = np.clip(int(np.sqrt(max_trajectories//n_blocks)), 10, 100)
    
    # Apply trajectory limit if specified
    if max_trajectories is not None:
        print(f"\nLimiting analysis to a maximum of {max_trajectories} trajectories...")
        for k in clean_trajs.keys():
            clean_trajs[k] = clean_trajs[k][:max_trajectories]

    save_filename = "./fullerenes_{}_block_dt_{}_ntraj_{}.npz".format(n_blocks, dt, max_trajectories)

    # Format into 3D arrays
    print("\nFormatting arrays...")
    formatted_cvs = {}
    for cv_name, data in clean_trajs.items():
        formatted_cvs[cv_name] = format_trajectories(np.array(data)[:, ::stride], dt)
    
    # 3. Block Averaging Engine
    print(f"\nStarting Block-Averaged EPR computation ({n_blocks} blocks)...")
    npz_export_dict = {}
    
    for cv_name, Z_master in formatted_cvs.items():
        print(f"  Processing CV: '{cv_name}' (Shape: {Z_master.shape})...")
        t0 = time.time()
        
        Z_blocks = np.array_split(Z_master, n_blocks, axis=0)
        
        Sdots_blocks = []
        Stot_blocks = []
        
        for b in range(n_blocks):
            Z_b = Z_blocks[b]
            
            Sdots_b, taus = EPR(Z_b, dt, bins=bins)
            Stot_b = np.trapezoid(Sdots_b - Sdots_b[-5:].mean(), taus)
            
            Sdots_blocks.append(Sdots_b)
            Stot_blocks.append(Stot_b)
            
        # Compute Statistics
        Sdots_mean = np.mean(Sdots_blocks, axis=0)
        Stot_mean = np.mean(Stot_blocks)
        Stot_err = np.std(Stot_blocks) / np.sqrt(n_blocks)
        
        # Flatten structure for clean NPZ saving
        npz_export_dict[f"{cv_name}_Sdots_blocks"] = Sdots_blocks
        npz_export_dict[f"{cv_name}_Sdots_mean"] = Sdots_mean
        npz_export_dict[f"{cv_name}_Stot_blocks"] = Stot_blocks
        npz_export_dict[f"{cv_name}_Stot_mean"] = Stot_mean
        npz_export_dict[f"{cv_name}_Stot_err"] = Stot_err
        npz_export_dict[f"{cv_name}_taus"] = taus
        
        print(f"    -> Stot = {Stot_mean:.4f} +/- {Stot_err:.4f} kT (Took {time.time()-t0:.1f}s)")
        
    # 4. Save to Disk
    print(f"\nSaving robust block-averaged results to '{save_filename}'...")
    np.savez_compressed(save_filename, n_blocks=n_blocks, dt=dt, bins=bins, stride=stride, **npz_export_dict)
    
    print(f"✅ Pipeline complete! Total Time: {time.time()-t_global:.1f}s")