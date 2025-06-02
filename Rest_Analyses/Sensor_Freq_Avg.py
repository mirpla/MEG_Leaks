#%%
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from meg_analysis.Scripts.Behavior.WL_implicitSEplot_Extend import process_WL_data

#%%
def process_subject(files, power):
    """
    Process a single subject file and compute alpha-beta averages.
    
    Parameters
    ----------
    file_path : Path
        Path to subject's H5 file
        
    Returns
    -------
    dict
        Processing results for the subject
    """
    power_label = list(power.keys())[0]
    power_band = power[power_label] 
    
    info = None
    
    # Extract subject ID from filename
    subject_id = files[0].stem[0:6]
    print(f"\n--- Processing Subject Files for: {subject_id} ---")

    block_average = {}
    baseline_power = None
    baseline_block_name = None
    for i, file_path in enumerate(files): 
        # Load spectral data
        data = mne.time_frequency.read_spectrum(file_path)

        if info is None:
            info = data.info   
    
        # Get frequency mask for 7-15 Hz band
        freq_mask = (data.freqs >= power_band[0]) & (data.freqs <= power_band[1])
        
        # Extract data and average within the frequency band
        spectrum_data = data.get_data()  # Shape: (n_epochs, n_channels, n_freqs)
        alpha_power = spectrum_data[:, :, freq_mask].mean(axis=(0, 2))
        
        block_name = file_path.stem[18:]
        block_average[block_name] = alpha_power
        
        # Store the second block (index 1) as baseline
        if i == 1:  # Second file in sorted order
            baseline_power = alpha_power
            baseline_block_name = block_name
            print(f"Using block '{baseline_block_name}' as baseline for normalization")
    
    # Normalize all blocks by the second block
    if baseline_power is not None:
        for block_name in block_average:
            if block_name != baseline_block_name:  # Don't normalize baseline against itself
                # Percent change normalization
                block_average[block_name] = (block_average[block_name] - baseline_power) / baseline_power
        
        # Set baseline block to zeros (since (baseline - baseline) / baseline = 0)
        block_average[baseline_block_name] = np.zeros_like(baseline_power)
    
    results = np.array(list(block_average.values()))
    return results, info

#%%
def process_all_subjects(base_path, power):
    """
    Process all subject files and compute power averages.
    
    Parameters
    ----------
    base_path : str or Path
        Base path to directory containing H5 files
    power : dict
        Dictionary with frequency band information, e.g. {'name of band': (fmin, fmax)}        
        
    Returns
    -------
    dict
        Results for all subjects
    """
    
    # Establish Paths
    data_path = base_path.parent.parent
    output_path = base_path / 'averages'
    output_path.mkdir(parents=True, exist_ok=True)

    WL_data, WL_subs = process_WL_data(m=0, min_seq_length=2, plot_flag=0)

    # Get all relevant subjects
    sub_lst     = {} # store subject labels for each condition
    sub_data    = {} # store data for each condition

    ses = 'ses-1'
    sensors = ['grad','mag'] #grad
    
    all_sensor_results = {}
    for sensor in sensors:
        all_results = {}
        file_pattern = f"sub-*{sensor}_spectrum.h5"  # Adjust pattern as needed
        output_file = output_path / f"sensor_{list(power.keys())[0]}_{sensor}_averages.h5"
        
        # check for existing output file and skip if not overwriting
        if output_file.exists() and input(f"\n{output_file.name} exists. Overwrite? (y/n): ").strip().lower() != 'y':
            print(f"Skipping {sensor} processing.") 
            continue
        Condition = ['Congruent', 'Incongruent']
        cond_data = {}  # Store data for each condition
        
        for c,ConIn in enumerate(Condition):
            print(f"\n{'='*60}")
            print(f"Processing {ConIn} Subjects for Sensor: {sensor}")
            print(f"{'='*60}")
            cond_data[ConIn] = {}
            # find the subjects for the relevant conditions and throw an error if the condition is invalid
            if ConIn == 'Congruent':
                sub_lst[c]  = ['sub-' + sub_id[2:4] for sub_id in WL_subs['con_imp']] 
                sub_data[c] = WL_data['con_imp']
            elif ConIn == 'Incongruent': 
                sub_lst[c]  = ['sub-' + sub_id[2:4] for sub_id in WL_subs['incon_imp']] # find the subjects in the incongruent implicit condition
                sub_data[c] = WL_data['incon_imp']
            else: 
                raise Exception(f'Condition {ConIn} not found; Condition has to be either ''Congruent'' or ''Incongruent''')  
            
            for s, sub in enumerate(sub_lst[c]):
                try:
                    central_block = sub_data[c][s].index(12)+1
                except:
                    print(f"Central block not found for {sub} in condition {ConIn}. Skipping subject.")
                    continue
                
                ses_folder = base_path / sub / ses / 'meg'
                
                files = list(ses_folder.glob(file_pattern))
                
                sort_files = sorted(files, key=lambda p: int(p.stem.split('_rest-')[1].split('_')[0]))
                
                # Process each subject
                results,info = process_subject(sort_files, power)

                cond_data[ConIn][s] = [results[1,:],results[2,:], results[central_block,:],results[11,:]]
                           
                # Summary
                successful = sum(1 for r in cond_data[ConIn].values() if r is not None)
                print(f"Successfully processed: {successful}/{len(sub_lst[c])} subjects")
            
            print(f"\n{'='*60}")
            print(f"PROCESSING COMPLETE")
            print(f"{'='*60}")

            subjects_data = cond_data[ConIn]
            # subjects_data is a dict with subject numbers as keys and (4,102) arrays as values
            data_matrices = list(subjects_data.values())
    
            # Stack all subjects' data along a new axis (subjects axis)
            # This creates an array of shape (n_subjects, 4, 102)
            stacked_data = np.stack(data_matrices, axis=0)
    
            # Average across subjects (axis=0)
            cond_average = np.mean(stacked_data, axis=0)    

            # Center colormap around zero for proper baseline visualization
            vmin = -0.3
            vmax = 0.3
            # abs_max = np.max(np.abs(cond_average))
            # vmin = -abs_max
            # vmax = abs_max

            fig, axes = plt.subplots(1, 4, figsize=(22, 6))
            fig.suptitle(f'Topographic Maps - {ConIn.capitalize()} Condition', fontsize=16)

            block_names = ['post-srt', 'Block 1', 'Central Block', 'Block 10']
            for block in range(4):
                im, cn = mne.viz.plot_topomap(
                    cond_average[block, :], 
                    info, 
                    axes=axes[block],
                    show=False,
                    cmap='RdBu_r',
                    sensors=True,
                    names=None,
                    vlim=(vmin, vmax)
                )
                axes[block].set_title(f'{block_names[block]}', fontsize=14)

            # Add colorbar
            cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
            cbar.set_label('Power Change (% from baseline)', fontsize=12)

            # Print values for console reference
            print(f"Color scale: {vmin:.3f} to {vmax:.3f} (centered on zero)")

            plt.tight_layout()
            plt.show()    
            
        Con_data    =  cond_data['Congruent'] 
        InCon_data  =  cond_data['Incongruent']
        
        Con_matrices = list(Con_data.values())
        InCon_matrices = list(InCon_data.values())
        Con_Stack = np.stack(Con_matrices, axis=0)
        InCon_Stack = np.stack(InCon_matrices, axis=0)
        
        Con_average = np.mean(Con_Stack, axis=0) 
        Incon_average = np.mean(InCon_Stack, axis=0)
        
        Diff_average = Con_average - Incon_average
        
        # abs_max = np.max(np.abs(Diff_average))
        # vmin = -abs_max
        # vmax = abs_max
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Topographic Maps - Congruent-Incongruent Condition', fontsize=16)
        
        block_names = ['post-srt', 'Block 1', 'Central Block', 'Block 10']
        for block in range(4):
            # Plot topomap for each block
            im, cn = mne.viz.plot_topomap(
                Diff_average[block, :], 
                info, 
                axes=axes[block],
                show=False,
                cmap='RdBu_r',  # You can change this colormap as needed
                sensors=True,
                names=None,
                vlim=(vmin, vmax)  # Set consistent scale
            )
            axes[block].set_title(f'{block_names[block]}', fontsize=14)
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Power Change (% from baseline)', fontsize=12)
        
        all_sensor_results[sensor] = cond_data
   
    return all_sensor_results

#%% 
if __name__ == "__main__":
    ALPHA_BETA_BAND = {'alpha-beta': (7, 15)}
    
    # Set your paths and parameters
    base_path = Path("Z:\Data\derivatives\spectral_analysis")  # Update this path
    
    # Process all subjects
    results = process_all_subjects(
        base_path=base_path,
        power = ALPHA_BETA_BAND,
    )
    

# %%
