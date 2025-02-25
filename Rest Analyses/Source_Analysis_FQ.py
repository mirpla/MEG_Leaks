import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import mne
from pathlib import Path
import importlib.util
from Source_Class import RestSourceData

def compute_power_spectra(data, sfreq, fmin=0, fmax=100):
    """Compute power spectra for all epochs and average."""
    n_epochs, n_sources, n_times = data.shape
    
    # Calculate FFT parameters
    n_fft = n_times
    freq_resolution = sfreq / n_fft
    frequencies = np.arange(0, sfreq/2, freq_resolution)
    freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
    frequencies = frequencies[freq_mask]
    
    # Initialize array for power spectra
    power_spectra = np.zeros((n_sources, len(frequencies)))
    
    # Compute FFT for each epoch and average
    for epoch in range(n_epochs):
        # Apply Hanning window
        window = np.hanning(n_times)
        windowed_data = data[epoch] * window[np.newaxis, :]
        
        # Compute FFT
        fft_data = fft.fft(windowed_data, axis=1)
        # Convert to power and keep only positive frequencies
        power = np.abs(fft_data[:, :n_times//2])**2
        # Add to average
        power_spectra += power[:, freq_mask]
    
    # Average across epochs
    power_spectra /= n_epochs
    
    return frequencies, power_spectra

def get_roi_indices_combined(vertices_lh, vertices_rh, labels_dir, roi_groups):
    """Get source indices for grouped ROIs but separate hemispheres."""
    roi_indices = {}
    
    for group_name, label_names in roi_groups.items():
        for hemi in ['lh', 'rh']:
            indices = []
            for label_name in label_names:
                label = mne.read_label(f"{labels_dir}/{hemi}.{label_name}.label")
                vertices = vertices_lh if hemi == 'lh' else vertices_rh
                offset = len(vertices_lh) if hemi == 'rh' else 0
                
                # Find indices for this label
                these_indices = [np.where(vertices == vert)[0][0] + offset 
                               for vert in label.vertices if vert in vertices]
                indices.extend(these_indices)
            
            if indices:
                roi_indices[f"{group_name}_{hemi}"] = indices
    
    return roi_indices

def plot_roi_spectra(frequencies, power_spectra, roi_indices, title="Power Spectra by ROI"):
    """Plot power spectra for different ROIs."""
    plt.figure(figsize=(12, 8))
    
    # Plot each ROI
    for roi_name, indices in roi_indices.items():
        # Average across sources within ROI
        roi_power = np.mean(power_spectra[indices], axis=0)
        
        # Plot in log scale
        plt.semilogy(frequencies, roi_power, label=roi_name)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (log scale)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Add frequency band annotations
    band_ranges = {
        'δ': (1, 4),
        'θ': (4, 8),
        'α': (8, 13),
        'β': (13, 30),
        'γ': (30, 100)
    }
    
    y_min, y_max = plt.ylim()
    for band, (fmin, fmax) in band_ranges.items():
        plt.fill_between([fmin, fmax], y_min, y_max, alpha=0.1)
        plt.text((fmin + fmax)/2, y_max, band, 
                horizontalalignment='center', verticalalignment='bottom')
    
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    sub = 'sub-24'
    ses = 'ses-1'
    block = 0
    h5_file = f'//analyse7/Project0407/Data/{sub}/{ses}/meg/rest/source/{sub}_{ses}_src_rest-all_dSPM-d8-l2-snr3.h5'
    
    data = RestSourceData(subject_id = sub, session_id = ses)
    block_data, sfreq, vertices_lh, vertices_rh = data.load_block_data(h5_file, block_id=block)
    
    
    # Compute power spectra
    rois={'primary_motor'   : ['BA4a_exvivo', 'BA4p_exvivo'],  # Combine anterior and posterior M1
          'premotor'        : ['BA6_exvivo']   
          }
    fmin=1,
    fmax=100,
    frequencies, power_spectra = compute_power_spectra(block_data, sfreq, fmin, fmax)

    # Get ROI indices
    labels_dir = Path('C:/fs_data') / sub / 'label' # Update this path
    roi_indices = get_roi_indices_combined(vertices_lh, vertices_rh, labels_dir, rois)
    
    # Plot results
    fig = plot_roi_spectra(frequencies, power_spectra, roi_indices,
                          title=f"Power Spectra by ROI - Block {block}")
