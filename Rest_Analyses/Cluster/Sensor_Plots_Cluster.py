import re
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
import multiprocessing as mp
from functools import partial
import time

def process_single_subject(sub, base_config):
    """
    Process a single subject - designed to be called in parallel.
    
    Parameters:
    -----------
    sub : str
        Subject folder name (e.g., 'sub-01')
    base_config : dict
        Configuration dictionary containing all necessary parameters
    """
    print(f"Starting processing for {sub}")
    start_time = time.time()
    
    # Extract configuration
    data_path = base_config['data_path']
    sessions = base_config['sessions']
    freq_bands = base_config['freq_bands']
    sensor_types = base_config['sensor_types']
    save_plots = base_config['save_plots']
    plots_dir = base_config['plots_dir']
    
    folder_path = data_path / sub
    
    for ses in sessions:
        ses_path = folder_path / ses / 'meg'
        
        if not ses_path.exists():
            print(f"Session path does not exist for {sub}/{ses}, skipping.")
            continue
        
        rest_dir = ses_path / 'rest'
        
        if not rest_dir.exists():
            print(f"Rest directory does not exist for {sub}/{ses}, skipping.")
            continue
        
        # Create output directory for subject plots
        sub_plots_dir = plots_dir / sub / ses
        sub_plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all epoch files for this subject and session
        epo_files = list(rest_dir.glob(f'{sub}_{ses}_rest-*_clean_epo.fif'))
        epo_files.sort(key=lambda x: int(re.search(r'rest-(\d+)', x.name).group(1)))
        
        if not epo_files:
            print(f"No epoch files found for {sub}/{ses}, skipping.")
            continue
        
        print(f"Processing {sub}/{ses}, found {len(epo_files)} epoch files")
        
        # Process each sensor type separately
        for sensor_type in sensor_types:
            print(f"Processing {sub}/{ses} - {sensor_type} sensors...")
            
            # Pre-load all epoch data for this sensor type to avoid repeated I/O
            all_epochs_data = {}
            sample_epochs = None
            
            for epo_file in epo_files:
                block_num = int(re.search(r'rest-(\d+)', epo_file.name).group(1))
                print(f"Loading {epo_file.name} for {sensor_type} sensors")
                epochs = mne.read_epochs(epo_file, preload=True)  # Preload for speed
                all_epochs_data[block_num] = epochs
                if sample_epochs is None:
                    sample_epochs = epochs
            
            # Create sensor-specific info once
            picks_sensor = mne.pick_types(sample_epochs.info, meg=sensor_type)
            info_sensor = mne.pick_info(sample_epochs.info, picks_sensor)
            
            # Pre-compute all PSD data for all frequency bands
            all_psd_data = {}
            for block_num, epochs in all_epochs_data.items():
                block_psd_data = {}
                for band_name, (fmin, fmax) in freq_bands.items():
                    spectrum = epochs.compute_psd(
                        method='welch',
                        fmin=fmin, 
                        fmax=fmax, 
                        n_fft=1024, 
                        n_overlap=512, 
                        picks=sensor_type
                    )
                    psd_data = spectrum.get_data()
                    psd_mean = psd_data.mean(axis=0)
                    psd_band_mean = psd_mean.mean(axis=1)
                    block_psd_data[band_name] = psd_band_mean
                all_psd_data[block_num] = block_psd_data
            
            # Create plots for each frequency band
            for band_name, (fmin, fmax) in freq_bands.items():
                # Create a figure for this frequency band and sensor type
                fig = plt.figure(figsize=(15, 10))
                sensor_name = "Magnetometers" if sensor_type == 'mag' else "Gradiometers"
                fig.suptitle(f'{sub} {ses} - {sensor_name} - {band_name.capitalize()} Band ({fmin}-{fmax} Hz)', fontsize=16)
                
                # Create layout for the figure
                n_files = len(epo_files)
                n_cols = min(4, n_files)
                n_rows = (n_files + n_cols - 1) // n_cols
                
                # Plot each block using pre-computed data
                for i, epo_file in enumerate(epo_files):
                    block_num = int(re.search(r'rest-(\d+)', epo_file.name).group(1))
                    psd_band_mean = all_psd_data[block_num][band_name]
                    
                    # Plot on the corresponding subplot
                    ax = plt.subplot(n_rows, n_cols, i + 1)
                    
                    # Create topomap with the sensor-specific info
                    mne.viz.plot_topomap(
                        psd_band_mean,
                        info_sensor,
                        sensors=True,
                        contours=6,
                        axes=ax,
                        cmap='viridis',
                        show=False
                    )
                    
                    if block_num == 0:
                        title = "SRT Baseline"
                    elif block_num == 1:
                        title = "WL Baseline"
                    else:
                        title = f"Block {block_num-1}"
                    
                    ax.set_title(title)
                
                plt.tight_layout()
                
                # Add colorbar
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                
                # Calculate min/max values for this sensor type and frequency band
                all_values = [all_psd_data[block_num][band_name] for block_num in all_psd_data.keys()]
                all_values_flat = np.concatenate(all_values)
                min_val = np.min(all_values_flat)
                max_val = np.max(all_values_flat)
                
                norm = plt.cm.colors.Normalize(vmin=min_val, vmax=max_val)
                plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), 
                           cax=cbar_ax, label='Power (a.u.)')
                
                # Save figure with sensor type in filename
                if save_plots:
                    filename = f'{sub}_{ses}_{sensor_type}_{band_name}_topomap.png'
                    fig.savefig(sub_plots_dir / filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
            
            # Additional analysis: Create comparison across blocks for alpha frequency
            if len(epo_files) > 1:
                # Use pre-computed alpha data
                alpha_blocks_data = []
                block_labels = []
                
                for epo_file in epo_files:
                    block_num = int(re.search(r'rest-(\d+)', epo_file.name).group(1))
                    
                    if block_num == 0:
                        label = "SRT Baseline"
                    elif block_num == 1:
                        label = "WL Baseline"
                    else:
                        label = f"Block {block_num-1}"
                    
                    block_labels.append(label)
                    alpha_blocks_data.append(all_psd_data[block_num]['alpha'])
                
                # Create comparison figure for this sensor type
                vmin = min([np.min(block_data) for block_data in alpha_blocks_data])
                vmax = max([np.max(block_data) for block_data in alpha_blocks_data])
                
                n_cols = min(4, len(alpha_blocks_data))
                n_rows = (len(alpha_blocks_data) + n_cols - 1) // n_cols
                
                fig = plt.figure(figsize=(15, 10))
                sensor_name = "Magnetometers" if sensor_type == 'mag' else "Gradiometers"
                fig.suptitle(f'{sub} {ses} - {sensor_name} - Alpha Band (8-13 Hz) - Comparison Across Blocks', fontsize=16)
                
                for i, (block_data, label) in enumerate(zip(alpha_blocks_data, block_labels)):
                    ax = plt.subplot(n_rows, n_cols, i + 1)
                    mne.viz.plot_topomap(
                        block_data,
                        info_sensor,
                        sensors=True,
                        contours=6,
                        axes=ax,
                        cmap='viridis',
                        vlim=(vmin, vmax),
                        show=False
                    )
                    ax.set_title(label)
                
                plt.tight_layout()
                
                # Add colorbar
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
                plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), 
                           cax=cbar_ax, label='Alpha Power (a.u.)')
                
                if save_plots:
                    filename = f'{sub}_{ses}_{sensor_type}_alpha_comparison.png'
                    fig.savefig(sub_plots_dir / filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
                    
                # Create power changes plot for this sensor type
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Get channel names for this sensor type
                sensor_picks = mne.pick_types(sample_epochs.info, meg=sensor_type)
                ch_names = [sample_epochs.ch_names[i] for i in sensor_picks]
                
                # Define regions based on sensor type
                if sensor_type == 'mag':
                    regions = {
                        'Frontal': [ch for ch in ch_names if any(x in ch for x in ['MEG01', 'MEG02'])],
                        'Central': [ch for ch in ch_names if any(x in ch for x in ['MEG12', 'MEG13'])],
                        'Parietal': [ch for ch in ch_names if any(x in ch for x in ['MEG23', 'MEG24'])],
                        'Occipital': [ch for ch in ch_names if any(x in ch for x in ['MEG31', 'MEG32'])]
                    }
                else:  # gradiometers
                    regions = {
                        'Frontal': [ch for ch in ch_names if any(x in ch for x in ['MEG01', 'MEG02'])],
                        'Central': [ch for ch in ch_names if any(x in ch for x in ['MEG12', 'MEG13'])],
                        'Parietal': [ch for ch in ch_names if any(x in ch for x in ['MEG23', 'MEG24'])],
                        'Occipital': [ch for ch in ch_names if any(x in ch for x in ['MEG31', 'MEG32'])]
                    }
                
                for region_name, region_chs in regions.items():
                    region_chs = region_chs[:5] if len(region_chs) > 5 else region_chs
                    ch_indices = [ch_names.index(ch) for ch in region_chs if ch in ch_names]
                    
                    if not ch_indices:
                        print(f"No {sensor_type} channels found for {region_name} region, skipping.")
                        continue
                    
                    # Extract power for this region across blocks
                    region_power = [np.mean(block_data[ch_indices]) for block_data in alpha_blocks_data]
                    
                    # Plot
                    ax.plot(range(len(block_labels)), region_power, 'o-', label=region_name)
                
                sensor_name = "Magnetometers" if sensor_type == 'mag' else "Gradiometers"
                ax.set_title(f'{sub} {ses} - {sensor_name} - Alpha Power Changes Across Blocks')
                ax.set_xlabel('Block')
                ax.set_ylabel('Alpha Power (a.u.)')
                ax.set_xticks(range(len(block_labels)))
                ax.set_xticklabels(block_labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                if save_plots:
                    filename = f'{sub}_{ses}_{sensor_type}_alpha_power_changes.png'
                    fig.savefig(sub_plots_dir / filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
    
    end_time = time.time()
    print(f"Finished processing {sub} in {end_time - start_time:.2f} seconds")
    return f"{sub} completed successfully"

def plot_meg_oscillations_parallel(script_dir, 
                                  sub_folders=None,
                                  sessions=['ses-1'],
                                  freq_bands={'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
                                             'beta': (13, 30), 'gamma': (30, 48)},
                                  sensor_types=['mag', 'grad'],
                                  save_plots=True,
                                  n_jobs=None):
    """
    Generate topographic plots of oscillatory activity for MEG resting state data using parallel processing.
    
    Parameters:
    -----------
    script_dir : Path
        Path to the script directory.
    sub_folders : list
        List of subject folders to process. If None, will detect automatically.
    sessions : list
        List of session folders to process.
    freq_bands : dict
        Dictionary of frequency bands to analyze.
    sensor_types : list
        List of sensor types to plot ('mag' for magnetometers, 'grad' for gradiometers).
    save_plots : bool
        Whether to save the plots. Default is True.
    n_jobs : int or None
        Number of parallel jobs to run. If None, uses all available CPU cores.
    """
    print("Starting parallel MEG oscillations analysis...")
    start_time = time.time()
    
    base_path = script_dir.parent.parent.parent.parent
    data_path = base_path / 'Data'
    
    # If sub_folders is not provided, detect automatically
    if sub_folders is None:
        reg_pattern = re.compile(r'^sub-\d{2}$')
        sub_folders = [d.name for d in data_path.iterdir() if d.is_dir() and reg_pattern.match(d.name)]

    # Create output directory for plots
    plots_dir = data_path / 'Rest' / 'Figs' / 'Sensor_Plots'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Set number of jobs
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    print(f"Processing {len(sub_folders)} subjects using {n_jobs} parallel processes")
    print(f"Subjects to process: {sub_folders}")
    
    # Create configuration dictionary
    base_config = {
        'data_path': data_path,
        'sessions': sessions,
        'freq_bands': freq_bands,
        'sensor_types': sensor_types,
        'save_plots': save_plots,
        'plots_dir': plots_dir
    }
    
    # Create partial function with fixed configuration
    process_func = partial(process_single_subject, base_config=base_config)
    
    # Run parallel processing
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(process_func, sub_folders)
    
    end_time = time.time()
    print(f"\nAll subjects completed in {end_time - start_time:.2f} seconds")
    print("Results:")
    for result in results:
        print(f"  - {result}")
    
    print("Finished processing all subjects.")

# Keep the original function for backward compatibility
def plot_meg_oscillations(script_dir, 
                         sub_folders=None,
                         sessions=['ses-1'],
                         freq_bands={'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
                                    'beta': (13, 30), 'gamma': (30, 48)},
                         sensor_types=['mag', 'grad'],
                         save_plots=True):
    """
    Original serial version - kept for backward compatibility.
    For better performance, use plot_meg_oscillations_parallel instead.
    """
    return plot_meg_oscillations_parallel(script_dir, sub_folders, sessions, 
                                        freq_bands, sensor_types, save_plots, n_jobs=1)

# Example usage:
if __name__ == "__main__":
    script_path = Path(__file__).parent
    
    # Parallel processing (recommended for HPC)
    plot_meg_oscillations_parallel(
        script_path, 
        sub_folders=None,  # Auto-detect all subjects
        sessions=['ses-1'],
        sensor_types=['mag', 'grad'],
        save_plots=True,
        n_jobs=None  # Use all available CPU cores
    )
    