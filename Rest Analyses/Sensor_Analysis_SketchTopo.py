import mne
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define function to compute the average PSD using FFT
def compute_avg_psd_fft(epochs):
    psd_list = []
    freqs = None
    for epoch in epochs:  # Iterate over each epoch
        # Compute the FFT for the epoch
        fft_data = np.fft.rfft(epoch, axis=1)  # FFT along the time dimension
        psd = np.abs(fft_data) ** 2  # Compute power spectral density

        # Compute the frequency bins
        if freqs is None:
            n_times = epoch.shape[1]
            sfreq = epochs.info['sfreq']
            freqs = np.fft.rfftfreq(n_times, d=1/sfreq)  # Frequency bins for the FFT
        
        psd_list.append(psd)

    # Convert list to a numpy array for easy manipulation
    psd_array = np.array(psd_list)

    # Compute the mean PSD across all epochs
    avg_psd = np.mean(psd_array, axis=0)  # Shape: (n_channels, n_freqs)
    return avg_psd, freqs

#def wl_performance():
script_dir      = Path(__file__).resolve() # Location of current scripts
data_path       = script_dir.parent.parent.parent / 'Data' # Root folder

sub = 'sub-20'
ses = 'ses-1'
file_pattern = f'{sub}_{ses}' + '_rest-{X}_clean_epo.fif'

file_indices = range(0, 12)

all_psds = []
file_labels = []

for idx in file_indices:
    # Construct the full file path
    file_name = file_pattern.format(X=idx)
    file_path = data_path / sub / ses/ 'meg' / 'rest' / file_name 

    data_epoch = mne.read_epochs( file_path,verbose=False)
    epochs = data_epoch.pick('grad')  # Use 'meg='grad'' for gradiometers

    avg_psd, freqs = compute_avg_psd_fft(data_epoch)

     # Append the average PSD to the list (avg_psd has shape: n_channels x n_freqs)
    all_psds.append(avg_psd)

    if idx == 0 :    
        file_labels.append('pre-SRT BL')  # Use file index as label
    elif idx == 1: 
        file_labels.append('post-SRT BL')
    else:
        file_labels.append(f'WL {idx-1}')

# %% Topo
all_psds = np.array(all_psds)  # Shape should now be (n_files, n_channels, n_freqs)

# Ensure the frequency range is limited to the alpha band (8-12 Hz)
alpha_band = (8, 10)
alpha_indices = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]  # Get indices of alpha frequencies

# Compute the average power in the alpha band for each channel
# Mean across the alpha band frequency range for each channel
alpha_power_across_channels = np.mean(all_psds[:, :, alpha_indices], axis=-1)  # Shape: (n_files, n_channels)

# Average the alpha power across all files (or use a specific file if preferred)
mean_alpha_power = np.mean(alpha_power_across_channels, axis=0)  # Shape: (n_channels,)

# Create the topomap plot
plt.figure(figsize=(8, 6))
mne.viz.plot_topomap(mean_alpha_power, data_epoch.info, cmap='viridis', show=True)
plt.title('Mean Alpha Band Power Distribution Across Channels')
plt.show()