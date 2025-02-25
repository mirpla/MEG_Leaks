import mne
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
    avg_psd = np.mean(psd_array, axis=0)  # Mean over epochs
    return avg_psd, freqs

#def wl_performance():
script_dir      = Path(__file__).resolve() # Location of current scripts
data_path       = script_dir.parent.parent.parent / 'Data' # Root folder

sub = 'sub-06'
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

    avg_psd, freqs = compute_avg_psd_fft(data_epoch)

    avg_psd_across_channels = np.mean(avg_psd, axis=0)  # Shape: (n_freqs,)

    # Append the average PSD to the list
    all_psds.append(avg_psd_across_channels)
    if idx == 0 :    
        file_labels.append('pre-SRT BL')  # Use file index as label
    elif idx == 1: 
        file_labels.append('post-SRT BL')
    else:
        file_labels.append(f'WL {idx-1}')
# %%
# Convert the list of PSDs to a 2D numpy array (n_files, n_freqs)
all_psds = np.array(all_psds)

freq_limit = 45
freq_indices = np.where(freqs <= freq_limit)[0]  # Get indices of frequencies up to 45 Hz

# Limit the frequency and PSD arrays to the desired range
freqs_limited = freqs[freq_indices]  # Frequency range limited to 0-45 Hz
all_psds_limited = all_psds[:, freq_indices]  # PSD values corresponding to the limited frequency range

# Create a meshgrid for the x and y axes
x = np.arange(all_psds_limited.shape[0])  # Files on x-axis (0, 1, 2, ... for each file)
y = freqs_limited  # Frequencies on y-axis (limited to 0-45 Hz)
X, Y = np.meshgrid(x, y)

# Transpose the PSD matrix so it matches the meshgrid dimensions (freqs x files)
Z = all_psds_limited.T  # Shape: (n_freqs, n_files)

# Create the 3D figure and plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Power')

# Set axis labels and title
ax.set_xlabel('Files')
ax.set_ylabel('Frequency (Hz)')
ax.set_zlabel('Power')
ax.set_title('3D Surface Plot of Power Spectral Density Across Files')

# Set x-axis labels as file names
ax.set_xticks(np.arange(len(file_labels)))  # Set the x-axis ticks to correspond to file indices
ax.set_xticklabels(file_labels, rotation=45, ha='right')  # Set file names as labels

# Set y-axis limit (0 to 45 Hz)
ax.set_ylim(0, 45)

plt.show()

# %% 

plt.figure(figsize=(12, 6))

# Use the Reds colormap to get progressively redder colors
cmap = plt.cm.Reds  # Colormap
colors = [cmap(i) for i in np.linspace(0.3, 1, len(file_indices))]  # Generate colors

# Plot each file's PSD using the progressively redder colors
for idx, (psd, color) in enumerate(zip(all_psds_limited, colors)):
    plt.plot(freqs_limited, psd, label=file_labels[idx], color=color)

# Add labels and title
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectral Densities Across Files')
plt.xlim(0, 45)  # Limit x-axis to 0-45 Hz
plt.grid(True)

# Create a legend outside the plot for better visibility
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()