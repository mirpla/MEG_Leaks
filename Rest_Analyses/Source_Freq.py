#%% 
import mne
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import welch
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import your RestSourceData class
from meg_analysis.Scripts.Rest_Analyses.Source_Class import RestSourceData

class MEGSourceAnalyzer:
    """
    Analyzer for MEG source-localized resting-state data with frequency analysis and visualization.
    """
    
    def __init__(self, base_path):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        base_path : str or Path
            Base path to the data directory
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'Data'
        
        # Define frequency bands of interest
        self.freq_bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 45)
        }
        
        # Cache directory for processed frequency data
        self.cache_dir = self.data_path / 'freq_analysis_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_source_file_path(self, subject, session, src_d, src_l, src_method, snr=3):
        """Get the path to a source file for given parameters."""
        source_dir = self.data_path / subject / session / 'meg' / 'rest' / 'source'
        filename = f'{subject}_{session}_src_rest-all_{src_method}-d{int(src_d*10)}-l{int(src_l*10)}-snr{int(snr)}.h5'
        return source_dir / filename
    
    def get_cache_file_path(self, subject, session, src_d, src_l, src_method, snr=3):
        """Get the path to cached frequency analysis for given parameters."""
        filename = f'{subject}_{session}_freq_analysis_{src_method}-d{int(src_d*10)}-l{int(src_l*10)}-snr{int(snr)}.pkl'
        return self.cache_dir / filename
    
    def compute_frequency_analysis(self, subject, session, src_d=0.8, src_l=0.4, 
                                 src_method='sLORETA', snr=3, force_recompute=False):
        """
        Compute frequency analysis for all blocks of a subject.
        
        Parameters
        ----------
        subject : str
            Subject ID (e.g., 'sub-01')
        session : str
            Session ID (e.g., 'ses-1')
        src_d : float
            Depth parameter used in source localization
        src_l : float
            Loose parameter used in source localization
        src_method : str
            Inverse method used ('dSPM', 'sLORETA', etc.)
        snr : int
            SNR parameter used
        force_recompute : bool
            If True, recompute even if cache exists
            
        Returns
        -------
        freq_data : dict
            Dictionary containing frequency analysis results
        """
        
        cache_file = self.get_cache_file_path(subject, session, src_d, src_l, src_method, snr)
        
        # Check if cached version exists
        if cache_file.exists() and not force_recompute:
            print(f"Loading cached frequency analysis for {subject} {session}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Computing frequency analysis for {subject} {session}")
        
        # Get source file path
        source_file = self.get_source_file_path(subject, session, src_d, src_l, src_method, snr)
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Initialize handler
        handler = RestSourceData(subject, session)
        
        # Get available blocks
        available_blocks = handler.get_available_blocks(source_file)
        print(f"Found {len(available_blocks)} blocks: {available_blocks}")
        
        # Initialize results dictionary
        freq_data = {
            'subject': subject,
            'session': session,
            'parameters': {
                'src_d': src_d, 'src_l': src_l, 'src_method': src_method, 'snr': snr
            },
            'freq_bands': self.freq_bands,
            'blocks': {},
            'vertices_lh': None,
            'vertices_rh': None,
            'sfreq': None
        }
        
        # Process each block
        for block_idx in available_blocks:
            print(f"Processing block {block_idx}...")
            
            # Load block data
            block_data, sfreq, vertices_lh, vertices_rh = handler.load_block_data(
                source_file, block_idx)
            
            # Store metadata (same for all blocks)
            if freq_data['vertices_lh'] is None:
                freq_data['vertices_lh'] = vertices_lh
                freq_data['vertices_rh'] = vertices_rh
                freq_data['sfreq'] = sfreq
            
            # Compute power spectral density for each epoch
            n_epochs, n_sources, n_times = block_data.shape
            
            # Initialize arrays for this block
            block_results = {
                'n_epochs': n_epochs,
                'band_power': {},  # Will store average power per frequency band
                'full_psd': None   # Will store full PSD if needed
            }
            
            # Compute PSD for each source and epoch
            print(f"  Computing PSD for {n_sources} sources, {n_epochs} epochs...")
            
            # Use Welch's method for robust PSD estimation
            freqs, psd_array = welch(
                block_data, 
                fs=sfreq, 
                nperseg=min(n_times, int(sfreq * 2)),  # 2-second windows
                noverlap=None,
                axis=-1
            )
            
            # Average across epochs for each source
            mean_psd = np.mean(psd_array, axis=0)  # Shape: (n_sources, n_freqs)
            
            # Compute band power for each frequency band
            for band_name, (fmin, fmax) in self.freq_bands.items():
                # Find frequency indices for this band
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                
                if np.any(freq_mask):
                    # Average power in this frequency band for each source
                    band_power = np.mean(mean_psd[:, freq_mask], axis=1)
                    block_results['band_power'][band_name] = band_power
                else:
                    print(f"  Warning: No frequencies found for {band_name} band")
                    block_results['band_power'][band_name] = np.zeros(n_sources)
            
            # Store results for this block
            freq_data['blocks'][block_idx] = block_results
            
        # Cache the results
        print(f"Saving frequency analysis to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(freq_data, f)
        
        return freq_data
    
    def plot_brain_activity(self, freq_data, block_idx, band='alpha', view='lateral', 
                          subject_mri=None, subjects_dir=None):
        """
        Plot brain activity for a specific block and frequency band.
        
        Parameters
        ----------
        freq_data : dict
            Frequency analysis results from compute_frequency_analysis
        block_idx : int
            Block index to plot
        band : str
            Frequency band to plot ('theta', 'alpha', 'beta', 'low_gamma')
        view : str
            Brain view ('lateral', 'medial', 'dorsal', 'ventral')
        subject_mri : str, optional
            Subject MRI to use for plotting (if None, uses fsaverage)
        subjects_dir : str, optional
            Path to subjects directory
        """
        
        if block_idx not in freq_data['blocks']:
            raise ValueError(f"Block {block_idx} not found in data")
        
        if band not in freq_data['blocks'][block_idx]['band_power']:
            raise ValueError(f"Band {band} not found in data")
        
        # Get the band power data
        band_power = freq_data['blocks'][block_idx]['band_power'][band]
        vertices_lh = freq_data['vertices_lh']
        vertices_rh = freq_data['vertices_rh']
        
        # Create SourceEstimate object for visualization
        stc = mne.SourceEstimate(
            data=band_power[:, np.newaxis],  # Add time dimension
            vertices=[vertices_lh, vertices_rh],
            tmin=0,
            tstep=1,
            subject=subject_mri or 'fsaverage'
        )
        
        # Plot on brain
        brain = stc.plot(
            subject=subject_mri or 'fsaverage',
            subjects_dir=subjects_dir,
            hemi='both',
            views=view,
            initial_time=0,
            time_unit='s',
            size=(800, 600),
            background='white',
            foreground='black',
            colormap='hot',
            transparent=True
        )
        
        brain.add_text(0.1, 0.9, 
                      f"{freq_data['subject']} - Block {block_idx} - {band.capitalize()} ({self.freq_bands[band][0]}-{self.freq_bands[band][1]} Hz)",
                      'title', font_size=14)
        
        return brain
    
    def plot_block_comparison(self, freq_data, band='alpha', figsize=(15, 10)):
        """
        Plot comparison of brain activity across all blocks for a specific frequency band.
        
        Parameters
        ----------
        freq_data : dict
            Frequency analysis results
        band : str
            Frequency band to plot
        figsize : tuple
            Figure size
        """
        
        blocks = sorted(freq_data['blocks'].keys())
        n_blocks = len(blocks)
        
        # Calculate subplot layout
        n_cols = min(4, n_blocks)
        n_rows = int(np.ceil(n_blocks / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_blocks == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, block_idx in enumerate(blocks):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get band power for this block
            band_power = freq_data['blocks'][block_idx]['band_power'][band]
            
            # Create a simple histogram of power values
            ax.hist(band_power, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'Block {block_idx}')
            ax.set_xlabel(f'{band.capitalize()} Power')
            ax.set_ylabel('Number of Sources')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(n_blocks, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col] 
            ax.remove()
        
        plt.suptitle(f'{freq_data["subject"]} - {band.capitalize()} Band Power Distribution Across Blocks', 
                     fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_frequency_bands_summary(self, freq_data, block_idx, figsize=(12, 8)):
        """
        Plot summary of all frequency bands for a specific block.
        
        Parameters
        ----------
        freq_data : dict
            Frequency analysis results
        block_idx : int
            Block to analyze
        figsize : tuple
            Figure size
        """
        
        if block_idx not in freq_data['blocks']:
            raise ValueError(f"Block {block_idx} not found")
        
        bands = list(self.freq_bands.keys())
        n_bands = len(bands)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, band in enumerate(bands):
            if idx < len(axes):
                band_power = freq_data['blocks'][block_idx]['band_power'][band]
                
                # Plot histogram
                axes[idx].hist(band_power, bins=50, alpha=0.7, 
                             color=f'C{idx}', edgecolor='black')
                axes[idx].set_title(f'{band.capitalize()} ({self.freq_bands[band][0]}-{self.freq_bands[band][1]} Hz)')
                axes[idx].set_xlabel('Power')
                axes[idx].set_ylabel('Number of Sources')
                axes[idx].grid(True, alpha=0.3)
                
                # Add statistics
                mean_power = np.mean(band_power)
                std_power = np.std(band_power)
                axes[idx].axvline(mean_power, color='red', linestyle='--', 
                                label=f'Mean: {mean_power:.2e}')
                axes[idx].legend()
        
        plt.suptitle(f'{freq_data["subject"]} - Block {block_idx} - All Frequency Bands', 
                     fontsize=14)
        plt.tight_layout()
        return fig
    
    def analyze_subject(self, subject, session, src_d=0.8, src_l=0.4, 
                       src_method='sLORETA', create_plots=True, 
                       subjects_dir=None):
        """
        Complete analysis pipeline for a single subject.
        
        Parameters
        ----------
        subject : str
            Subject ID
        session : str
            Session ID
        src_d, src_l : float
            Source localization parameters
        src_method : str
            Inverse method
        create_plots : bool
            Whether to create visualization plots
        subjects_dir : str, optional
            Path to subjects directory for MRI plotting
            
        Returns
        -------
        freq_data : dict
            Frequency analysis results
        """
        
        print(f"\n=== Analyzing {subject} {session} ===")
        
        # Compute frequency analysis
        freq_data = self.compute_frequency_analysis(
            subject, session, src_d, src_l, src_method)
        
        if create_plots:
            print("Creating visualization plots...")
            
            # Plot comparison across blocks for alpha band
            fig1 = self.plot_block_comparison(freq_data, band='alpha')
            plt.show()
            
            # Plot frequency bands summary for first few blocks
            blocks = sorted(freq_data['blocks'].keys())
            for block_idx in blocks[:3]:  # Show first 3 blocks
                fig2 = self.plot_frequency_bands_summary(freq_data, block_idx)
                plt.show()
        
        return freq_data
    
    def batch_analyze_subjects(self, subjects, session, src_d=0.8, src_l=0.4, 
                              src_method='sLORETA', subjects_dir=None):
        """
        Analyze multiple subjects in batch.
        
        Parameters
        ----------
        subjects : list
            List of subject IDs
        session : str
            Session ID
        src_d, src_l : float
            Source localization parameters
        src_method : str
            Inverse method
        subjects_dir : str, optional
            Path to subjects directory
            
        Returns
        -------
        results : dict
            Dictionary with results for each subject
        """
        
        results = {}
        
        for subject in subjects:
            try:
                print(f"\n{'='*50}")
                print(f"Processing {subject}")
                print(f"{'='*50}")
                
                freq_data = self.analyze_subject(
                    subject, session, src_d, src_l, src_method, 
                    create_plots=False, subjects_dir=subjects_dir)
                
                results[subject] = freq_data
                print(f"✓ Successfully processed {subject}")
                
            except Exception as e:
                print(f"✗ Error processing {subject}: {str(e)}")
                results[subject] = None
        
        return results


# Example usage function
def run_analysis_example():
    """
    Example of how to use the MEG source analyzer.
    """
    
    # Initialize analyzer (adjust base_path to your data location)
    base_path = Path("//analyse7/Project0407/")  # Update this path
    analyzer = MEGSourceAnalyzer(base_path)
    
    # Define your analysis parameters (matching your source localization)
    src_d = 0.8
    src_l = 0.4
    src_f = False
    src_method = 'sLORETA'
    
    # Example subjects and session
    subjects = ['sub-67', 'sub-04']  # Update with your subject IDs
    session = 'ses-1'  # Update with your session ID
    
    # Analyze single subject
    print("Analyzing single subject...")
    freq_data = analyzer.analyze_subject(
        subjects[0], session, src_d, src_l, src_method)
    
    # Plot brain activity for specific blocks and bands
    print("Creating brain plots...")
    
    # Plot alpha activity for first block (baseline)
    brain1 = analyzer.plot_brain_activity(
        freq_data, block_idx=0, band='alpha', view='lateral')
    
    # Plot alpha activity for a task block
    available_blocks = list(freq_data['blocks'].keys())
    if len(available_blocks) > 2:
        brain2 = analyzer.plot_brain_activity(
            freq_data, block_idx=available_blocks[2], band='alpha', view='lateral')
    
    # Batch analyze multiple subjects
    print("Batch analyzing subjects...")
    all_results = analyzer.batch_analyze_subjects(
        subjects, session, src_d, src_l, src_method)
    
    return analyzer, all_results

if __name__ == "__main__":
    # Run the example
    analyzer, results = run_analysis_example()
# %%
