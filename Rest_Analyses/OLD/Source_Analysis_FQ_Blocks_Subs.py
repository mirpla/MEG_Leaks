import mne
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy import fft
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm  # for progress bars

from meg_analysis.Scripts.Behavior.WL_Analysis import wl_performance

from fooof import FOOOF
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg

def process_single_subject(sub: str, ses: str, side: str, labels_dir: Path, h5_file: Path, 
                         fmin: float = 1, fmax: float = 50) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """Process MEG data for a single subject."""
    with h5py.File(h5_file, 'r') as f:
        # Get available blocks
        blocks = sorted([int(b.split('_')[1]) for b in f['blocks'].keys()])
        
        # Get necessary metadata
        sfreq = f['info'].attrs['sfreq']
        vertices_lh = f['source_space/vertices_lh'][:]
        vertices_rh = f['source_space/vertices_rh'][:]
        
        motor_indices = get_motor_indices(vertices_lh, vertices_rh, labels_dir, sub)
        all_indices = np.unique(motor_indices[side])
        
        # Initialize storage for spectra
        block_spectra = []
        
        # Calculate FFT parameters
        n_times = f[f'blocks/block_{blocks[0]:02d}/data'].shape[2]
        n_fft = n_times
        freq_resolution = sfreq / n_fft
        frequencies = np.arange(0, sfreq/2, freq_resolution)
        freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
        frequencies = frequencies[freq_mask]
        
        # Process each block
        for block in tqdm(blocks, desc=f"Processing {sub}"):
            data = f[f'blocks/block_{block:02d}/data']
            n_epochs = data.shape[0]
            block_power = np.zeros((n_epochs, len(frequencies)))
            
            for epoch in range(n_epochs):
                motor_data = data[epoch, all_indices, :]
                window = np.hanning(n_times)
                windowed_data = motor_data * window[np.newaxis, :]
                fft_data = fft.fft(windowed_data, axis=1)
                power = np.abs(fft_data[:, :n_times//2])**2
                block_power[epoch] = np.mean(power[:, freq_mask], axis=0)
            
            block_spectra.append(np.mean(block_power, axis=0))
            
    return frequencies, block_spectra, blocks


def extract_periodic_components(frequencies, spectrum, freq_range=[1, 50]):
    """Extract alpha and beta periodic components using FOOOF."""
    fm = FOOOF(
        peak_width_limits=[1, 8],
        max_n_peaks=6,
        min_peak_height=0.1,
        peak_threshold=2.0,
        aperiodic_mode='fixed'
    )
    
    # Fit FOOOF model
    fm.fit(frequencies, spectrum, freq_range)
    
    # Define frequency bands
    alpha_range = (8, 13)
    beta_range = (13, 30)
    
    # Get peaks in alpha and beta bands
    alpha_peak = None
    beta_peak = None
    
    for peak in fm.peak_params_:
        freq, power, width = peak
        if alpha_range[0] <= freq <= alpha_range[1]:
            alpha_peak = power
        elif beta_range[0] <= freq <= beta_range[1]:
            beta_peak = power
    
    return {
        'alpha': alpha_peak if alpha_peak is not None else 0,
        'beta': beta_peak if beta_peak is not None else 0,
        'aperiodic_offset': fm.aperiodic_params_[0],
        'aperiodic_slope': fm.aperiodic_params_[1]
    }

def plot_periodic_components(block_spectra, blocks, behavioral_data, frequencies,
                           central_block=4, window_size=2):
    """Plot periodic alpha and beta components with behavioral data."""
    # Select relevant blocks
    block_indices = [i for i, b in enumerate(blocks) 
                    if central_block - window_size <= b <= central_block + window_size]
    selected_blocks = [blocks[i] for i in block_indices]
    selected_spectra = [block_spectra[i] for i in block_indices]
    selected_behavioral = [behavioral_data[i] for i in block_indices]
    
    # Extract periodic components for each block
    components = []
    for spectrum in selected_spectra:
        components.append(extract_periodic_components(frequencies, spectrum))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    # Plot periodic components
    x = [b+1 for b in selected_blocks]  # Add 1 to block numbers
    alpha_values = [comp['alpha'] for comp in components]
    beta_values = [comp['beta'] for comp in components]
    
    ax1.plot(x, alpha_values, 'o-', label='Alpha', linewidth=2)
    ax1.plot(x, beta_values, 'o-', label='Beta', linewidth=2)
    ax1.axvline(x=central_block+1, color='r', linestyle='--', label='Central Block')  # Add 1 to central block
    ax1.set_xlabel('Block Number')
    ax1.set_ylabel('Peak Power')
    ax1.set_title('Periodic Components')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(x)
    
    # Plot behavioral data
    ax2.plot(x, selected_behavioral, 'k-o', linewidth=2)
    ax2.axvline(x=central_block+1, color='r', linestyle='--', label='Central Block')  # Add 1 to central block
    ax2.set_xlabel('Block Number')
    ax2.set_ylabel('Behavioral Measure')
    ax2.set_title('Behavioral Performance')
    ax2.grid(True)
    ax2.set_xticks(x)
    
    plt.tight_layout()
    return fig

def plot_aperiodic_components(block_spectra, blocks, behavioral_data, frequencies,
                            central_block=4, window_size=2):
    """Plot aperiodic offset and slope with behavioral data."""
    # Select relevant blocks
    block_indices = [i for i, b in enumerate(blocks) 
                    if central_block - window_size <= b <= central_block + window_size]
    selected_blocks = [blocks[i] for i in block_indices]
    selected_spectra = [block_spectra[i] for i in block_indices]
    selected_behavioral = [behavioral_data[i] for i in block_indices]
    
    # Extract aperiodic components for each block
    components = []
    for spectrum in selected_spectra:
        components.append(extract_periodic_components(frequencies, spectrum))
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    # Plot offset
    ax1 = fig.add_subplot(gs[0])
    x = [b+1 for b in selected_blocks]  # Add 1 to block numbers
    offset_values = [comp['aperiodic_offset'] for comp in components]
    
    ax1.plot(x, offset_values, 'o-', color='blue', linewidth=2)
    ax1.axvline(x=central_block+1, color='r', linestyle='--', label='Central Block')
    ax1.set_xlabel('Block Number')
    ax1.set_ylabel('Offset')
    ax1.set_title('Aperiodic Offset')
    ax1.grid(True)
    ax1.set_xticks(x)
    
    # Plot slope
    ax2 = fig.add_subplot(gs[1])
    slope_values = [comp['aperiodic_slope'] for comp in components]
    
    ax2.plot(x, slope_values, 'o-', color='green', linewidth=2)
    ax2.axvline(x=central_block+1, color='r', linestyle='--', label='Central Block')
    ax2.set_xlabel('Block Number')
    ax2.set_ylabel('Slope')
    ax2.set_title('Aperiodic Slope')
    ax2.grid(True)
    ax2.set_xticks(x)
    
    # Plot behavioral data
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(x, selected_behavioral, 'k-o', linewidth=2)
    ax3.axvline(x=central_block+1, color='r', linestyle='--', label='Central Block')
    ax3.set_xlabel('Block Number')
    ax3.set_ylabel('Behavioral Measure')
    ax3.set_title('Behavioral Performance')
    ax3.grid(True)
    ax3.set_xticks(x)
    
    plt.tight_layout()
    return fig


def analyze_spectra_with_fooof(frequencies, block_spectra, freq_range=[1, 50]):
    """Analyze power spectra using FOOOF for each block."""
    # Initialize FOOOF object
    fm = FOOOF(
        peak_width_limits=[1, 8],
        max_n_peaks=6,
        min_peak_height=0.1,
        peak_threshold=2.0,
        aperiodic_mode='fixed'
    )
    
    # Initialize storage for components
    aperiodic_fits = []
    oscillatory_fits = []
    peak_params = []
    
    # Process each block
    for spectrum in block_spectra:
        # Fit FOOOF model
        fm.fit(frequencies, spectrum, freq_range)
        
        # Get model components
        aperiodic = fm.get_model('aperiodic')  # This is the correct method
        fooofed = fm.fooofed_spectrum_         # Full model fit
        oscillatory = fooofed - aperiodic      # Oscillatory component
        
        # Store results
        aperiodic_fits.append(aperiodic)
        oscillatory_fits.append(oscillatory)
        peak_params.append(fm.peak_params_)
        
    return aperiodic_fits, oscillatory_fits, peak_params

def plot_spectral_components(frequencies, block_spectra, blocks, freq_range=[1, 50]):
    """Create three-panel plot showing full, aperiodic, and oscillatory components."""
    # Run FOOOF analysis
    aperiodic_fits, oscillatory_fits, peak_params = analyze_spectra_with_fooof(
        frequencies, block_spectra, freq_range)
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    
    # Create colormap for blocks
    colors = plt.cm.Reds(np.linspace(0, 1, len(blocks)))
    
    # 1. Full Power Spectra
    ax1 = fig.add_subplot(gs[0])
    for i, (spectrum, color) in enumerate(zip(block_spectra, colors)):
        ax1.semilogy(frequencies, spectrum, color=color, label=f'Block {blocks[i]}')
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (log scale)')
    ax1.set_title('Full Power Spectra')
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.set_xlim(freq_range)
    
    # 2. Aperiodic Components
    ax2 = fig.add_subplot(gs[1])
    for i, (aperiodic, color) in enumerate(zip(aperiodic_fits, colors)):
        ax2.semilogy(frequencies, aperiodic, color=color, label=f'Block {blocks[i]}')
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (log scale)')
    ax2.set_title('Aperiodic Components')
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    ax2.set_xlim(freq_range)
    
    # 3. Oscillatory Components
    ax3 = fig.add_subplot(gs[2])
    for i, (oscillatory, color) in enumerate(zip(oscillatory_fits, colors)):
        # Subtract 1 from oscillatory fit to center around 0
        ax3.plot(frequencies, oscillatory - 1, color=color, label=f'Block {blocks[i]}')
    
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power (relative to aperiodic)')
    ax3.set_title('Oscillatory Components')
    ax3.grid(True, which='both', linestyle='--', alpha=0.6)
    ax3.set_xlim(freq_range)
    
    # Add frequency band annotations to all plots
    band_ranges = {
        'δ': (1, 4),
        'θ': (4, 8),
        'α': (8, 13),
        'β': (13, 30),
        'γ': (30, 50)
    }
    
    for ax in [ax1, ax2, ax3]:
        y_min, y_max = ax.get_ylim()
        for band, (fmin, fmax) in band_ranges.items():
            ax.fill_between([fmin, fmax], y_min, y_max, alpha=0.1)
            ax.text((fmin + fmax)/2, y_max, band, 
                   horizontalalignment='center', verticalalignment='bottom')
        
        # Ensure tick at 10 Hz
        current_ticks = list(ax.get_xticks())
        if 10 not in current_ticks:
            current_ticks.append(10)
            ax.set_xticks(sorted(current_ticks))
    
    # Add legend to the right of the last subplot
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig, (aperiodic_fits, oscillatory_fits, peak_params)

def get_motor_indices(vertices_lh, vertices_rh, labels_dir, subject):
    """Get combined motor cortex indices for both hemispheres."""
    motor_labels = ['BA4a_exvivo', 'BA4p_exvivo']
    motor_indices = {'left': [], 'right': []}
    
    for hemi, indices in zip(['lh', 'rh'], [motor_indices['left'], motor_indices['right']]):
        for label_name in motor_labels:
            label = mne.read_label(f"{labels_dir}/{hemi}.{label_name}.label")
            vertices = vertices_lh if hemi == 'lh' else vertices_rh
            offset = len(vertices_lh) if hemi == 'rh' else 0
            
            # Find indices for this label
            these_indices = [np.where(vertices == vert)[0][0] + offset 
                           for vert in label.vertices if vert in vertices]
            indices.extend(these_indices)
    
    return motor_indices

def compute_band_powers(frequencies, spectra, bands=None):
    """Compute average power in different frequency bands."""
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        # Find frequency indices for this band
        freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
        
        # Compute mean power in band for each block
        powers = np.array([np.mean(spectrum[freq_mask]) for spectrum in spectra])
        
        # Store mean and SEM
        band_powers[band_name] = {
            'mean': np.mean(powers),
            'sem': np.std(powers) / np.sqrt(len(powers))
        }
    
    return band_powers

def save_processed_data(output_file, subjects_data):
    """
    Save processed FFT data to HDF5 file, skipping existing subjects.
    """
    # Check if file exists
    output_file = Path(output_file)
    if output_file.exists():
        # Open existing file in append mode
        with h5py.File(output_file, 'a') as f:
            existing_subjects = set(f.keys())
            
            for sub_id, data in subjects_data.items():
                sub_key = f'sub-{sub_id}'
                if sub_key in existing_subjects:
                    print(f"Skipping {sub_key} - already exists in file")
                    continue
                
                print(f"Saving new data for {sub_key}")
                sub_group = f.create_group(sub_key)
                
                # Store spectra as dataset with compression
                sub_group.create_dataset('spectra', data=data['spectra'],
                                       compression='gzip', compression_opts=9)
                
                # Store metadata
                sub_group.create_dataset('block_nums', data=data['block_nums'])
                sub_group.create_dataset('frequencies', data=data['frequencies'])
                sub_group.attrs['sfreq'] = data['sfreq']
                sub_group.attrs['central_block'] = data['central_block']
                
    else:
        # Create new file if it doesn't exist
        with h5py.File(output_file, 'w') as f:         
            # Save all subjects
            for sub_id, data in subjects_data.items():
                sub_group = f.create_group(f'sub-{sub_id}')
                
                # Store spectra as dataset with compression
                sub_group.create_dataset('spectra', data=data['spectra'],
                                       compression='gzip', compression_opts=9)
                
                # Store metadata
                sub_group.create_dataset('block_nums', data=data['block_nums'])
                sub_group.create_dataset('frequencies', data=data['frequencies'])
                sub_group.attrs['sfreq'] = data['sfreq']
                sub_group.attrs['central_block'] = data['central_block']

def save_subject(out_file, sub, subject_data):
    mode = 'a' if out_file.exists() else 'w'
    
    with h5py.File(out_file, mode) as f:
       # Add file attributes if creating new file
       if mode == 'w':
           f.attrs['description'] = 'Processed MEG FFT data for specific ROI'
       
       # Check if subject already exists
       if sub in f:
           print(f"Skipping {sub} - already exists in file")
           return False
       
       # Create new subject group and save data
       print(f"Saving data for {sub}")
       sub_group = f.create_group(sub)
       
       # Store spectra as dataset with compression
       sub_group.create_dataset('spectra', data=subject_data['spectra'],
                              compression='gzip', compression_opts=9)
       
       # Store metadata
       sub_group.create_dataset('blocks', data=subject_data['blocks'])
       sub_group.create_dataset('freqs', data=subject_data['freqs'])
       sub_group.attrs['central_block'] = subject_data['central_block']
       
#%% Start Script
if __name__ == "__main__":
    # load behavioral data of all subjects
    WL_data, WL_subs = wl_performance()
    
    # paramters
    sub_lst     = {}
    sub_lst[0]  = []#['sub-02','sub-04','sub-06','sub-13','sub-21','sub-24','sub-31','sub-35','sub-36']
    sub_lst[1]  = []
    
    subject_data = {}
    behavioral_data = {}
    ses = 'ses-1'   
    LR  = ['left','right'] 
    Condition = ['Congruent', 'Incongruent']
    for c,ConIn in enumerate(Condition):
        for side_idx, side in enumerate(LR): # left only for now
            window_size = 2 # windowsize next to central block
            
            # general paths
            base_path   = Path('//analyse7/Project0407/')
            out_file    = base_path / 'Data' / 'Rest' / f'{ConIn}_{side}_Motor.h5'
            # Frequency Range
            fmin = 1
            fmax = 50
            
            # Process each subject
            for sub in sub_lst[c]:
                subjects_data = {}  
                # subject specific paths
                source_path = base_path / f'/Data/{sub}/{ses}/meg/rest/source/'
                h5_file     = source_path / f'{sub}_{ses}_src_rest-all_dSPM-d8-l2-snr3.h5'
                labels_dir  = Path('C:/fs_data') / sub / 'label'
                
                frequencies, block_spectra, blocks = process_single_subject(
                    sub, ses, side, labels_dir, h5_file, fmin, fmax)
    
                # Load behavioral data and find the first occurance of full list learning
                sub_int     = int(sub.split('-')[1]) 
                subidx = [x for x, subnum in WL_subs.items() if subnum == sub_int]   # find the data for this subject 
                                
                subject_data['spectra']         = block_spectra
                subject_data['freqs']           = frequencies 
                subject_data['blocks']          = blocks
                subject_data['central_block']   = WL_data[subidx[0]].index(12)
                
                save_subject(out_file, sub, subject_data)
        #%% Plot individual spectra per subjects
                # fig_sub     = source_path / f'{sub}_{ses}_motor_freq.png'                        
                # fig1, components = plot_spectral_components(frequencies, block_spectra, blocks)
                # fig1.savefig(fig_sub, dpi=300)
                # plt.close(fig1)
                
        # #%% behav figs        
        #      # Create periodic components plot
        #     fig2 = plot_periodic_components(
        #         subjects_data,  WL_data[subidx[0]], central_blocks, window_size
        #      )
             
        #      # Create aperiodic components plot
        #     fig3 = plot_aperiodic_components(
        #         block_spectra, blocks,  WL_data[subidx[0]],
        #         frequencies, central_block=central_blocks, window_size=window_size
        #      )
         
        #     # # Save figures
            # fig1.savefig(f'{sub}_spectral_components_behavioral.png', dpi=300, bbox_inches='tight')
            # fig2.savefig(f'{sub}_band_powers_behavioral.png', dpi=300, bbox_inches='tight')
        
        
