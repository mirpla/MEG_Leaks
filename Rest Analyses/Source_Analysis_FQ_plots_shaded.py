import h5py
import math
import numpy as np
import matplotlib.pyplot as plt

from fooof import FOOOF
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from typing import List, Dict, Tuple

from meg_analysis.Scripts.Behavior.WL_Analysis import wl_performance

from pathlib import Path

def load_processed_data(file_path):
    """Load processed FFT data from HDF5 file."""
    data = {}
    with h5py.File(file_path, 'r') as f:
        for sub_id in f.keys():
            sub_data = {
                'spectra': f[f'{sub_id}/spectra'][:],
                'blocks': f[f'{sub_id}/blocks'][:],
                'freqs': f[f'{sub_id}/freqs'][:],
                'central_block': f[f'{sub_id}'].attrs['central_block']
            }
            data[sub_id] = sub_data
    return data


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

def plot_multi_subject_components(subjects_data: Dict, window_size: int,
                                WL_data: Dict, WL_idx: Dict, component_type: str, fill_color = 'k') -> plt.Figure:
    """Plot periodic or aperiodic components with confidence intervals/standard deviation."""
    mean_width = 3
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[1, 1, 1], constrained_layout=True)

    plt.rcParams['font.family'] = 'Calibri'

    mean_color = 'k'
        
    fill_alpha = 0.2
    
    all_components = {
        'alpha': [], 'beta': [], 
        'offset': [], 'slope': [],
        'behavioral': []
    }
    
    # First pass: collect all data
    for sub_idx, sub in enumerate(subjects_data.keys()):
        s_data = subjects_data[sub]
        central_block = s_data['central_block']
        spectra = s_data['spectra']
        freq = s_data['freqs']
        block = s_data['blocks']
        
        # Select the blocks around the central block
        block_indices = [i for i, b in enumerate(block) 
                        if central_block - window_size <= b <= central_block + window_size]
        filtered_blocks = [x for x in block_indices if 0 <= x <= 9]
        selected_spectra = [spectra[i+2] for i in filtered_blocks]
        
        # Extract components
        components = []
        for spectrum in selected_spectra:
            components.append(extract_periodic_components(freq, spectrum))
        
        n_items = 1 + (window_size * 2)
        x = np.arange(0, n_items)
        
        # Extract values
        alpha_values = [comp['alpha'] for comp in components]
        beta_values = [comp['beta'] for comp in components]
        offset_values = [comp['aperiodic_offset'] for comp in components]
        slope_values = [comp['aperiodic_slope'] for comp in components]
        
        # Get behavioral data
        sub_int = int(sub.split('-')[1])
        WL_idx_sub = [x for x, subnum in WL_subs.items() if subnum == sub_int]
        behav = WL_data[WL_idx_sub[0]]
        selected_behav = [behav[i] for i in filtered_blocks]
        
        # Pad values if necessary
        if len(selected_spectra) < n_items:
            pad_length = n_items - len(selected_spectra)
            if any(value >= 10 for value in block_indices):
                # Pad at end
                alpha_values.extend([np.nan] * pad_length)
                beta_values.extend([np.nan] * pad_length)
                offset_values.extend([np.nan] * pad_length)
                slope_values.extend([np.nan] * pad_length)
                selected_behav.extend([np.nan] * pad_length)
            else:
                # Pad at start
                alpha_values = ([np.nan] * pad_length) + alpha_values
                beta_values = ([np.nan] * pad_length) + beta_values
                offset_values = ([np.nan] * pad_length) + offset_values
                slope_values = ([np.nan] * pad_length) + slope_values
                selected_behav = ([np.nan] * pad_length) + selected_behav
        
        # Normalize values
        alpha_values = normalize_measures(alpha_values, window_size)
        beta_values = normalize_measures(beta_values, window_size)
        slope_values = normalize_measures(slope_values, window_size)
        offset_values = normalize_measures(offset_values, window_size)
        
        all_components['alpha'].append(alpha_values)
        all_components['beta'].append(beta_values)
        all_components['offset'].append(offset_values)
        all_components['slope'].append(slope_values)
        all_components['behavioral'].append(selected_behav)
    
    # Calculate statistics and plot
    x = np.arange(0, n_items)
    for comp_name in all_components.keys():
        comp_array = np.array(all_components[comp_name])
        mean_comp = np.nanmean(comp_array, axis=0)
        std_comp = np.nanstd(comp_array, axis=0)
        
        # Calculate 95% confidence interval
        n_subjects = np.sum(~np.isnan(comp_array), axis=0)
        sem_comp = std_comp / np.sqrt(n_subjects)
        ci_95 = 1.96 * sem_comp
        
        if component_type == 'periodic':
            if comp_name == 'alpha':
                ax1.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax1.fill_between(x, mean_comp - ci_95, mean_comp + ci_95, 
                               color=fill_color, alpha=fill_alpha, label='95% CI')
            elif comp_name == 'beta':
                ax2.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax2.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
            elif comp_name == 'behavioral':
                ax3.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax3.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
        else:
            if comp_name == 'offset':
                ax1.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax1.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
            elif comp_name == 'slope':
                ax2.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax2.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
            elif comp_name == 'behavioral':
                ax3.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax3.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
    
    # Set labels and titles
    if component_type == 'periodic':
        ax1.set_title('Alpha Power')
        ax2.set_title('Beta Power')
    else:
        ax1.set_title('Aperiodic Offset')
        ax2.set_title('Aperiodic Slope')
    
    ax3.set_title('Behavioral Performance')
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    ax1.set_ylim(-60, 60)
    ax2.set_ylim(-60, 60)
    ax3.set_ylim(1, 12)
    ax1.set_ylabel('Change from reference (%)')
    ax2.set_ylabel('Change from reference (%)')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Block Number')
        ax.grid(True)
        ax.legend()
    
    return fig

def plot_multi_subject_components_unlocked(subjects_data: Dict,
                                WL_data: Dict, WL_idx: Dict, component_type: str, fill_color = 'k') -> plt.Figure:
    """Plot periodic or aperiodic components with confidence intervals/standard deviation."""
    mean_width = 3
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[1, 1, 1], constrained_layout=True)

    plt.rcParams['font.family'] = 'Calibri'

    mean_color = 'k'
    fill_color = 'b'
    fill_alpha = 0.2
    
    all_components = {
        'alpha': [], 'beta': [], 
        'offset': [], 'slope': [],
        'behavioral': []
    }
    
    # First pass: collect all data
    for sub_idx, sub in enumerate(subjects_data.keys()):
        s_data = subjects_data[sub]
        spectra = s_data['spectra']
        freq = s_data['freqs']
        block = s_data['blocks']
        
        selected_spectra = [spectra[i] for i in block]
        
        # Extract components
        components = []
        for spectrum in selected_spectra:
            components.append(extract_periodic_components(freq, spectrum))
        
        
        # Extract values
        alpha_values = [comp['alpha'] for comp in components]
        beta_values = [comp['beta'] for comp in components]
        offset_values = [comp['aperiodic_offset'] for comp in components]
        slope_values = [comp['aperiodic_slope'] for comp in components]
        
      
        # Get behavioral data
        sub_int = int(sub.split('-')[1])
        WL_idx_sub = [x for x, subnum in WL_subs.items() if subnum == sub_int]
        behav = [np.nan, np.nan] + WL_data[WL_idx_sub[0]]
        
        # Normalize values
        alpha_values = normalize_measures_bl(alpha_values)
        beta_values = normalize_measures_bl(beta_values)
        slope_values = normalize_measures_bl(slope_values)
        offset_values = normalize_measures_bl(offset_values)
        
        if np.any(np.isinf(alpha_values)):
            continue
        
        all_components['alpha'].append(alpha_values)
        all_components['beta'].append(beta_values)
        all_components['offset'].append(offset_values)
        all_components['slope'].append(slope_values)
        all_components['behavioral'].append(behav)
    
    # Calculate statistics and plot
    x = np.arange(0, len(behav))
    for comp_name in all_components.keys():
        comp_array = np.array(all_components[comp_name])
        mean_comp = np.nanmean(comp_array, axis=0)
        std_comp = np.nanstd(comp_array, axis=0)
        
        # Calculate 95% confidence interval
        n_subjects = np.sum(~np.isnan(comp_array), axis=0)
        sem_comp = std_comp / np.sqrt(n_subjects)
        ci_95 = 1.96 * sem_comp
        
        if component_type == 'periodic':
            if comp_name == 'alpha':
                ax1.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax1.fill_between(x, mean_comp - ci_95, mean_comp + ci_95, 
                               color=fill_color, alpha=fill_alpha, label='95% CI')
            elif comp_name == 'beta':
                ax2.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax2.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
            elif comp_name == 'behavioral':
                ax3.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax3.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
        else:
            if comp_name == 'offset':
                ax1.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax1.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
            elif comp_name == 'slope':
                ax2.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax2.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
            elif comp_name == 'behavioral':
                ax3.plot(x, mean_comp, color=mean_color, linewidth=mean_width, label='Mean')
                ax3.fill_between(x, mean_comp - ci_95, mean_comp + ci_95,
                               color=fill_color, alpha=fill_alpha, label='95% CI')
    
    # Set labels and titles
    if component_type == 'periodic':
        ax1.set_title('Alpha Power')
        ax2.set_title('Beta Power')
    else:
        ax1.set_title('Aperiodic Offset')
        ax2.set_title('Aperiodic Slope')
    
    ax3.set_title('Behavioral Performance')
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    ax1.set_ylim(-60, 60)
    ax2.set_ylim(-60, 60)
    ax3.set_ylim(1, 12)
    ax1.set_ylabel('Change from reference (%)')
    ax2.set_ylabel('Change from reference (%)')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Block Number')
        ax.grid(True)
        ax.legend()
    
    return fig


def normalize_measures(measure, window_size):
    measure = np.array(measure, dtype=float)
    measure[measure == 0] = np.nan
    reference_measure = measure[window_size] 
    measure = ((measure - reference_measure) / reference_measure) * 100
    
    return measure

def normalize_measures_bl(measure):
    measure = np.array(measure, dtype=float)
    
    if measure[0]  == 0:
        reference_measure = measure[1]
    else:
        reference_measure = measure[0] 
    
    measure = ((measure - reference_measure) / reference_measure) * 100

    
    return measure
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

# %%
window_size = 3
LR  = ['left'] 
Condition = ['Congruent', 'Incongruent']
base_path   = Path('//analyse7/Project0407/')
fig_path    = base_path / '/Data/Rest/Figs/Sub_Spectra/' 
WL_data, WL_subs = wl_performance(Path(base_path))

#%%
for c,ConIn in enumerate(Condition):
    for side_idx, side in enumerate(LR): # left only for now
        if c == 0:
            col = 'b'
        else:
            col = 'r'
            
        file_path   = base_path / f'/Data/Rest/{ConIn}_{side}_Motor.h5'
        if file_path.exists():
            data = load_processed_data(file_path)
        else:
            print(f'file {file_path.name} not found! Skipping...')
            continue
        # for sub in data.keys():
        #     fig1,components = plot_spectral_components(data[sub]['freqs'], data[sub]['spectra'], data[sub]['blocks'])
        #    # fig1.savefig(fig_path / f'{sub}_{ConIn}_{side}_Motor_spectral_components.png', dpi=300, bbox_inches='tight')
           #plt.close(fig1)
            
        # fig2 = plot_multi_subject_components_unlocked(
        #     data, WL_data, WL_subs, 'periodic', col)
        fig2 = plot_multi_subject_components(
            data, window_size, WL_data, WL_subs, 'periodic', col)
        fig2.suptitle(f'{ConIn} {side} Motor Oscillatory activity')
        
        #fig3 = plot_multi_subject_components_unlocked(
        #    data, WL_data, WL_subs, 'aperiodic', col)
        #fig3 = plot_multi_subject_components(
        #    data, window_size, WL_data, WL_subs, 'aperiodic', col)
        # fig3.suptitle(f'{ConIn} {side} Motor Aperiodic activity')
        #fig2.savefig('multi_subject_periodic_components.png', dpi=300, bbox_inches='tight')