#%% 
from mne import read_source_estimate
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import psd_array_welch, psd_array_multitaper
from pathlib import Path
from meg_analysis.Scripts.Rest_Analyses.Source_Class import RestSourceData
#%%
# define the parameters for loading
test_data_path = 'Z:/Data/sub-67/ses-1/meg/rest/source/sub-67_ses-1_src_rest-all_sLORETA-d8-l4-snr3.h5'

save_path = Path(test_data_path).parent / 'avg_source_plots_sLORETA-d8-l4-snr3'  # Optional: set to None to not save
    
block_id=1

filename = Path(test_data_path).stem
parts = filename.split('_')
subject_id = parts[0]  # e.g., 'sub-01'
session_id = parts[1] 

# initialize the RestSourceData handler
meg_handler = RestSourceData(subject_id, session_id)

available_blocks = meg_handler.get_available_blocks(test_data_path)
print(f"Available blocks: {available_blocks}")

if block_id not in available_blocks:
    raise ValueError(f"Block {block_id} not found. Available blocks: {available_blocks}")

# Load the block data
print(f"Loading block {block_id}...")
block_data, sfreq, vertices_lh, vertices_rh = meg_handler.load_block_data(test_data_path, block_id)

psds, freqs = psd_array_multitaper(block_data,sfreq = sfreq, fmin = 1, fmax = 30, n_jobs = 10)

power_avg = np.mean(psds, axis=0)  # (sources, frequencies)





#%% Worry about plotting later
subjects_dir = "C:/fs_data"

mne.utils.set_config('SUBJECTS_DIR', subjects_dir, set_env=True)
stc_alpha = mne.SourceEstimate(
    data=alpha_power_avg[:, np.newaxis],  # Add time dimension: (8196, 1)
    vertices=[vertices_lh, vertices_rh],
    tmin=0, 
    tstep=1,  # Not 0!
    subject=subject_id
)

peak_vertex, _ = stc_alpha.get_peak(hemi = None)

if peak_vertex< len(vertices_lh):
    peak_hemi = 'lh'
else:
    peak_hemi = 'rh'

brain = stc_alpha.plot(
    subject=subject_id,
    views=['lateral', 'medial'],
    hemi='both',
    colormap='hot',
    size=(800, 600)
)

brain.add_foci(
    peak_vertex, 
    coords_as_verts=True,  # Use vertex coordinates
    hemi = peak_hemi,
    color='red', 
    scale_factor=2
)
#%% 
if save_path is not None:
    save_path.mkdir(parents=True, exist_ok=True)
    brain.save_image(save_path / f'sub-{subject_id}_ses-{session_id}_block-{block_id}_alpha_power.png')
    print(f"Saved plot to {save_path / f'sub-{subject_id}_ses-{session_id}_block-{block_id}_alpha_power.png'}")