#%%
#!/usr/bin/env python3
"""
Cleaned MEG Source PSD Analysis Script with Grand Average Functionality
Analyzes baseline-normalized power spectra from HDF5 files and creates brain visualizations
with grand averaging across subjects in common source space
"""

import numpy as np
import mne
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
from typing import Dict, List, Tuple, Optional
import argparse
import re
from meg_analysis.Scripts.Rest_Analyses.Source_Class import RestSourceData
from meg_analysis.Scripts.Behavior.WL_implicitSEplot_Extend import process_WL_data


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MEGGrandAverageStorage:
    """
    Class for storing and managing subject data for grand averaging in common source space.
    Uses MNE's built-in functions and handles morphing to fsaverage.
    """
    
    def __init__(self, target_subject: str = 'fsaverage'):
        """
        Initialize storage for grand averaging.
        
        Parameters
        ----------
        target_subject : str
            Target subject for morphing (typically 'fsaverage')
        """
        self.subjects_data = {}  # Dict[subject_id, SourceEstimate_list]
        self.condition_mapping = {}  # Dict[subject_id, condition_label]
        self.target_subject = target_subject
        self.freq_band = None
        self.n_blocks = None
        self.method_name = None  # Store the analysis method name
        
    def add_subject(self, subject_id: str, normalized_power: np.ndarray, 
                   vertices_lh: np.ndarray, vertices_rh: np.ndarray,
                   condition: str, freq_band: Tuple[float, float],
                   subjects_dir: str, method_name: str = None):
        """
        Add a subject's data to storage and morph to common space.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        normalized_power : np.ndarray
            Normalized power data, shape (n_blocks, n_sources)
        vertices_lh, vertices_rh : np.ndarray
            Hemisphere vertices for this subject
        condition : str
            Condition label
        freq_band : tuple
            Frequency band
        subjects_dir : str
            FreeSurfer subjects directory
        method_name : str
            Source localization method name (e.g., 'eLORETA', 'multitaper')
        """
        # Create SourceEstimates for each block
        block_stcs = []
        for block_idx in range(normalized_power.shape[0]):
            # Create individual SourceEstimate
            stc = mne.SourceEstimate(
                data=normalized_power[block_idx, :, np.newaxis],
                vertices=[vertices_lh, vertices_rh],
                tmin=0,
                tstep=1,
                subject=subject_id
            )
            
            # Morph to common space if needed
            if subject_id != self.target_subject:
                morph = mne.compute_source_morph(
                    stc, subject_from=subject_id, subject_to=self.target_subject,
                    subjects_dir=subjects_dir, smooth=5, warn=False
                )
                stc_morphed = morph.apply(stc)
            else:
                stc_morphed = stc
            
            block_stcs.append(stc_morphed)
        
        # Store morphed data
        self.subjects_data[subject_id] = block_stcs
        self.condition_mapping[subject_id] = condition
        
        # Set shared parameters
        if self.freq_band is None:
            self.freq_band = freq_band
            self.n_blocks = len(block_stcs)
            self.method_name = method_name
        
        logger.info(f"Added {subject_id} ({condition}) - morphed to {self.target_subject}")
    
    def get_condition_subjects(self, condition: str) -> List[str]:
        """Get subjects for a specific condition."""
        return [sub_id for sub_id, cond in self.condition_mapping.items() if cond == condition]
    
    def get_available_conditions(self) -> List[str]:
        """Get available conditions."""
        return list(set(self.condition_mapping.values()))
    
    def compute_grand_average_stc(self, condition: str, block_id: int,
                                 subjects_subset: Optional[List[str]] = None) -> mne.SourceEstimate:
        """
        Compute grand average by manually averaging SourceEstimate data.
        
        Parameters
        ----------
        condition : str
            Condition to average
        block_id : int
            Block index to average
        subjects_subset : list, optional
            Specific subjects to include
            
        Returns
        -------
        mne.SourceEstimate
            Grand average source estimate
        """
        # Get subjects for this condition
        if subjects_subset is None:
            subjects_to_use = self.get_condition_subjects(condition)
        else:
            condition_subjects = set(self.get_condition_subjects(condition))
            subjects_to_use = [sub for sub in subjects_subset if sub in condition_subjects]
        
        if not subjects_to_use:
            raise ValueError(f"No subjects found for condition '{condition}'")
        
        # Collect STCs for this block
        block_stcs = [self.subjects_data[subject_id][block_id] for subject_id in subjects_to_use]
        
        # Manually average the data
        # Stack all subject data arrays
        stacked_data = np.stack([stc.data for stc in block_stcs], axis=0)  # (n_subjects, n_sources, n_times)
        
        # Average across subjects
        averaged_data = np.mean(stacked_data, axis=0)  # (n_sources, n_times)
        
        # Create new SourceEstimate with averaged data
        # Use the first STC as template for vertices and timing info
        template_stc = block_stcs[0]
        grand_avg_stc = mne.SourceEstimate(
            data=averaged_data,
            vertices=template_stc.vertices,
            tmin=template_stc.tmin,
            tstep=template_stc.tstep,
            subject=self.target_subject
        )
        
        logger.info(f"Grand average for '{condition}', block {block_id}: {len(subjects_to_use)} subjects")
        return grand_avg_stc
    
    def compute_condition_difference_stc(self, condition1: str, condition2: str, block_id: int,
                                        subjects_subset1: Optional[List[str]] = None,
                                        subjects_subset2: Optional[List[str]] = None) -> mne.SourceEstimate:
        """Compute difference between conditions using MNE operations."""
        stc1 = self.compute_grand_average_stc(condition1, block_id, subjects_subset1)
        stc2 = self.compute_grand_average_stc(condition2, block_id, subjects_subset2)
        return stc1 - stc2
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        conditions = self.get_available_conditions()
        summary = {
            'total_subjects': len(self.subjects_data),
            'conditions': {},
            'freq_band': self.freq_band,
            'n_blocks': self.n_blocks,
            'target_subject': self.target_subject,
            'method_name': self.method_name
        }
        
        for condition in conditions:
            subjects = self.get_condition_subjects(condition)
            summary['conditions'][condition] = {
                'n_subjects': len(subjects),
                'subjects': subjects
            }
        
        return summary


class MEGPSDAnalyzer:
    """Main analyzer class with grand averaging capabilities."""
    
    def __init__(self, subjects_dir: str = "C:/fs_data", freq_band: Tuple[float, float] = (8, 16),
                 linux_root: str = '/analyse/Project0407', windows_root: str = 'Z:',
                 target_subject: str = 'fsaverage'):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        subjects_dir : str
            Path to FreeSurfer subjects directory
        freq_band : tuple
            Frequency band for analysis (fmin, fmax) in Hz
        linux_root, windows_root : str
            Path conversion parameters
        target_subject : str
            Target subject for morphing (typically 'fsaverage')
        """
        self.subjects_dir = Path(subjects_dir)
        self.freq_band = freq_band
        self.linux_root = linux_root
        self.windows_root = windows_root
        self.target_subject = target_subject
        
        # Set MNE subjects directory
        mne.utils.set_config('SUBJECTS_DIR', str(self.subjects_dir), set_env=True)
        
        # Initialize grand average storage
        self.grand_average_storage = MEGGrandAverageStorage(target_subject)
    
    def load_psd_data(self, psd_file_path: Path) -> Dict:
        """Load PSD data from HDF5 file."""
        data = {}
        
        with h5py.File(psd_file_path, 'r') as f:
            # Load metadata
            data['subject_id'] = f.attrs['subject_id']
            data['session_id'] = f.attrs['session_id']
            data['fmin'] = f.attrs['fmin']
            data['fmax'] = f.attrs['fmax']
            data['sfreq'] = f.attrs['sfreq']
            data['method'] = f.attrs['psd_method']
            
            # Load frequencies
            data['freqs'] = f['frequencies/freqs'][:]
            
            # Load PSD blocks
            data['blocks'] = {}
            block_names = sorted([name for name in f['psd_blocks'].keys() if name.startswith('block_')])
            
            for block_name in block_names:
                block_id = int(block_name.split('_')[1])
                data['blocks'][block_id] = f[f'psd_blocks/{block_name}/power_avg'][:]
            
            # Load or reference vertices
            if 'source_space' in f and f['source_space'].attrs.get('stored_locally', False):
                data['vertices_lh'] = f['source_space/vertices_lh'][:]
                data['vertices_rh'] = f['source_space/vertices_rh'][:]
            else:
                source_file_path = f.attrs['source_file_path']
                data['vertices_lh'], data['vertices_rh'] = self._load_vertices_from_source(source_file_path)
        
        logger.info(f"Loaded PSD data for {data['subject_id']}: {len(data['blocks'])} blocks")
        return data
    
    def _convert_path_to_local(self, original_path: str) -> Path:
        """Convert Linux server path to Windows path."""
        path_obj = Path(original_path)
        path_str = str(path_obj).replace('\\', '/')
        
        if path_str.startswith(self.linux_root):
            converted = path_str.replace(self.linux_root, self.windows_root, 1)
            windows_path = Path(converted.replace('/', '\\'))
            logger.debug(f"Converted path: {original_path} -> {windows_path}")
            return windows_path
        else:
            logger.debug(f"No path conversion needed: {original_path}")
            return Path(original_path)
    
    def _load_vertices_from_source(self, source_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load vertices from original source file."""
        converted_path = self._convert_path_to_local(source_file_path)
        
        if not converted_path.exists():
            raise FileNotFoundError(f"Source file not found: {converted_path}")
        
        with h5py.File(converted_path, 'r') as f:
            vertices_lh = f['source_space/vertices_lh'][:]
            vertices_rh = f['source_space/vertices_rh'][:]
        return vertices_lh, vertices_rh
    
    def extract_frequency_band(self, psd_data: Dict, freq_band: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Extract and average power in specified frequency band."""
        if freq_band is None:
            freq_band = self.freq_band
        
        freqs = psd_data['freqs']
        freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        
        if not np.any(freq_mask):
            raise ValueError(f"No frequencies found in band {freq_band[0]}-{freq_band[1]} Hz")
        
        logger.info(f"Extracting {freq_band[0]}-{freq_band[1]} Hz band: {np.sum(freq_mask)} frequencies")
        
        # Extract and average power across frequency band for all blocks
        n_blocks = len(psd_data['blocks'])
        n_sources = list(psd_data['blocks'].values())[0].shape[0]
        band_power = np.zeros((n_blocks, n_sources))
        
        for i, (block_id, power_data) in enumerate(sorted(psd_data['blocks'].items())):
            band_power[i, :] = np.mean(power_data[:, freq_mask], axis=1)
        
        return band_power
    
    def baseline_normalize(self, band_power: np.ndarray, baseline_block: int = 0) -> np.ndarray:
        """Apply baseline normalization: (block - baseline) / baseline * 100"""
        baseline = band_power[baseline_block, :]
        baseline_safe = np.where(baseline == 0, np.finfo(float).eps, baseline)
        normalized = ((band_power - baseline[np.newaxis, :]) / baseline_safe[np.newaxis, :]) * 100
        
        logger.info(f"Applied baseline normalization using block {baseline_block}. "
                   f"Range: {normalized.min():.1f}% to {normalized.max():.1f}%")
        return normalized
    
    def add_subject_to_grand_average(self, subject_id: str, normalized_power: np.ndarray,
                                   vertices_lh: np.ndarray, vertices_rh: np.ndarray,
                                   condition: str, method_name: str = None):
        """Add subject data to grand average storage with automatic morphing."""
        self.grand_average_storage.add_subject(
            subject_id, normalized_power, vertices_lh, vertices_rh,
            condition, self.freq_band, str(self.subjects_dir), method_name
        )
    
    def get_storage_summary(self) -> Dict:
        """Get summary of stored grand average data."""
        return self.grand_average_storage.get_summary_stats()


# Utility functions
def convert_wl_to_subject_id(wl_code) -> str:
    """Convert WL code to subject ID (handles both 'WL22' and 1022 formats)."""
    if isinstance(wl_code, str) and wl_code.startswith('WL'):
        subject_num = int(wl_code[2:])
        return f'sub-{subject_num:02d}'
    elif isinstance(wl_code, (int, float)):
        last_two = int(str(int(wl_code))[-2:])
        return f'sub-{last_two:02d}'
    else:
        raise ValueError(f"Cannot convert {wl_code} to subject ID")


def create_condition_lookup(WL_subs: Dict[str, List]) -> Dict[str, str]:
    """Create condition lookup from WL_subs data."""
    condition_lookup = {}
    
    for wl_code in WL_subs['con_imp']:
        subject_id = convert_wl_to_subject_id(wl_code)
        condition_lookup[subject_id] = 'Congruent'
    
    for wl_code in WL_subs['incon_imp']:
        subject_id = convert_wl_to_subject_id(wl_code)
        condition_lookup[subject_id] = 'Incongruent'
    
    logger.info(f"Created condition lookup: {len(condition_lookup)} subjects")
    return condition_lookup


def extract_method_from_filename(filename: str) -> str:
    """
    Extract the method name from the PSD filename.
    
    Parameters
    ----------
    filename : str
        The PSD filename (e.g., 'sub-02_psd-eLORETA-d8-l3-snr3.h5')
        
    Returns
    -------
    str
        The method string (e.g., 'eLORETA-d8-l3-snr3')
    """
    # Extract everything between 'psd-' and '.h5'
    match = re.search(r'psd-(.+?)\.h5', filename)
    if match:
        return match.group(1)
    else:
        # Fallback: try to extract from pattern
        match = re.search(r'psd-(\w+(?:-\w+)*)', filename)
        if match:
            return match.group(1)
        else:
            logger.warning(f"Could not extract method from filename: {filename}")
            return "unknown-method"


def find_psd_files_in_subfolders(data_root: Path, psd_pattern: str) -> Dict[str, Path]:
    """
    Find PSD files in subject subfolders.
    
    Parameters
    ----------
    data_root : Path
        Root directory containing subject subfolders
    psd_pattern : str
        Pattern to match PSD files
        
    Returns
    -------
    dict
        Dictionary mapping subject_id to file path
    """
    found_files = {}
    missing_subjects = []
    
    # Look for subject folders (e.g., sub-01, sub-02, etc.)
    subject_folders = [f for f in data_root.iterdir() if f.is_dir() and f.name.startswith('sub-')]
    
    logger.info(f"Found {len(subject_folders)} subject folders in {data_root}")
    
    for subject_folder in subject_folders:
        subject_id = subject_folder.name
        sub_folder = subject_folder / 'ses-1' / 'meg' 
        # Look for PSD files matching the pattern in this subject's folder
        psd_files = list(sub_folder.glob(psd_pattern))
        
        if len(psd_files) == 0:
            missing_subjects.append(subject_id)
            logger.warning(f"No PSD files found for {subject_id} in {sub_folder}")
        elif len(psd_files) == 1:
            found_files[subject_id] = psd_files[0]
            logger.info(f"Found PSD file for {subject_id}: {psd_files[0].name}")
        else:
            # Multiple files found - take the first one but warn
            found_files[subject_id] = psd_files[0]
            logger.warning(f"Multiple PSD files found for {subject_id}, using: {psd_files[0].name}")
    
    logger.info(f"Successfully found PSD files for {len(found_files)} subjects")
    if missing_subjects:
        logger.info(f"Missing PSD files for {len(missing_subjects)} subjects: {missing_subjects}")
    
    return found_files


# Plotting functions with method name inclusion
def plot_grand_average_brain(storage: MEGGrandAverageStorage, 
                           condition: str, block_id: int,
                           subjects_subset: Optional[List[str]] = None,
                           colormap: str = 'hot',
                           views: List[str] = ['lateral', 'medial'],
                           save_path: Optional[Path] = None) -> mne.viz.Brain:
    """Plot grand average brain using MNE defaults."""
    grand_avg_stc = storage.compute_grand_average_stc(condition, block_id, subjects_subset)
    
    # Get number of subjects for title
    if subjects_subset is None:
        subjects_to_use = storage.get_condition_subjects(condition)
    else:
        condition_subjects = set(storage.get_condition_subjects(condition))
        subjects_to_use = [sub for sub in subjects_subset if sub in condition_subjects]
    
    n_subjects = len(subjects_to_use)
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    brain = grand_avg_stc.plot(
        subject=storage.target_subject,
        hemi='both',
        views=views,
        colormap=colormap,
        title=f'{condition} Grand Average (n={n_subjects}) - Block {block_id} - {storage.freq_band[0]}-{storage.freq_band[1]}Hz - {method_str}'
    )
    
    if save_path:
        brain.save_image(save_path)
        logger.info(f"Saved grand average brain to {save_path}")
    
    return brain


def plot_condition_difference_brain(storage: MEGGrandAverageStorage,
                                  condition1: str, condition2: str, block_id: int,
                                  subjects_subset1: Optional[List[str]] = None,
                                  subjects_subset2: Optional[List[str]] = None,
                                  colormap: str = 'RdBu_r',
                                  views: List[str] = ['lateral', 'medial'],
                                  save_path: Optional[Path] = None) -> mne.viz.Brain:
    """Plot condition difference brain."""
    stc_diff = storage.compute_condition_difference_stc(
        condition1, condition2, block_id, subjects_subset1, subjects_subset2
    )
    
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    brain = stc_diff.plot(
        subject=storage.target_subject,
        hemi='both',
        views=views,
        colormap=colormap,
        title=f'{condition1} - {condition2} (Block {block_id}) - {storage.freq_band[0]}-{storage.freq_band[1]}Hz - {method_str}'
    )
    
    if save_path:
        brain.save_image(save_path)
        logger.info(f"Saved condition difference brain to {save_path}")
    
    diff_data = stc_diff.data[:, 0]
    logger.info(f"Difference range: {diff_data.min():.2f}% to {diff_data.max():.2f}%")
    
    return brain


def plot_time_course_comparison(storage: MEGGrandAverageStorage, 
                              conditions: Optional[List[str]] = None,
                              save_path: Optional[Path] = None):
    """Plot time course comparison between conditions."""
    if conditions is None:
        conditions = storage.get_available_conditions()
    
    plt.figure(figsize=(10, 6))
    
    for condition in conditions:
        block_means = []
        block_sems = []
        
        for block_id in range(storage.n_blocks):
            grand_avg_stc = storage.compute_grand_average_stc(condition, block_id)
            block_data = grand_avg_stc.data[:, 0]
            block_means.append(np.mean(block_data))
            block_sems.append(np.std(block_data) / np.sqrt(len(block_data)))
        
        blocks = np.arange(len(block_means))
        plt.errorbar(blocks, block_means, yerr=block_sems,
                    marker='o', label=condition, capsize=5, linewidth=2)
    
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    plt.xlabel('Block')
    plt.ylabel('Power Change (%)')
    plt.title(f'Grand Average Time Course - {storage.freq_band[0]}-{storage.freq_band[1]} Hz - {method_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved time course plot to {save_path}")
    
    plt.show()


def run_complete_grand_average_analysis(analyzer: MEGPSDAnalyzer,
                                      output_dir: Path,
                                      baseline_block: int = 0,
                                      method_name: str = None) -> Dict:
    """Run complete grand average analysis with method name in outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    storage = analyzer.grand_average_storage
    conditions = storage.get_available_conditions()
    method_str = storage.method_name if storage.method_name else (method_name or "unknown-method")
    
    if len(conditions) == 0:
        logger.error("No conditions found in storage!")
        return {}
    
    # Print analysis summary
    print("\n=== Grand Average Analysis Summary ===")
    print(f"Target space: {storage.target_subject}")
    print(f"Frequency band: {storage.freq_band[0]}-{storage.freq_band[1]} Hz")
    print(f"Method: {method_str}")
    print(f"Number of blocks: {storage.n_blocks}")
    print(f"Total subjects: {len(storage.subjects_data)}")
    
    for condition in conditions:
        subjects = storage.get_condition_subjects(condition)
        print(f"{condition}: {len(subjects)} subjects ({subjects})")
    
    # Plot grand averages
    for condition in conditions:
        for block_id in range(storage.n_blocks):
            if block_id == baseline_block:
                continue
                
            save_path = output_dir / f'grand_avg_{condition}_block{block_id}_{storage.freq_band[0]}-{storage.freq_band[1]}Hz_{method_str}.png'
            brain = plot_grand_average_brain(storage, condition, block_id, save_path=save_path)
            brain.close()
    
    # Plot condition differences
    if len(conditions) == 2:
        cond1, cond2 = conditions
        for block_id in range(storage.n_blocks):
            if block_id == baseline_block:
                continue
                
            save_path = output_dir / f'difference_{cond1}_vs_{cond2}_block{block_id}_{storage.freq_band[0]}-{storage.freq_band[1]}Hz_{method_str}.png'
            brain = plot_condition_difference_brain(storage, cond1, cond2, block_id, save_path=save_path)
            brain.close()
    
    # Plot time course
    time_course_path = output_dir / f'time_course_comparison_{storage.freq_band[0]}-{storage.freq_band[1]}Hz_{method_str}.png'
    plot_time_course_comparison(storage, conditions, save_path=time_course_path)
    
    return {
        'conditions': conditions,
        'n_subjects_per_condition': {cond: len(storage.get_condition_subjects(cond)) for cond in conditions},
        'total_subjects': len(storage.subjects_data),
        'freq_band': storage.freq_band,
        'target_subject': storage.target_subject,
        'method_name': method_str
    }


def plot_multi_block_grand_average_brain(storage: MEGGrandAverageStorage, 
                                       condition: str,
                                       blocks_to_plot: List[int],
                                       subjects_subset: Optional[List[str]] = None,
                                       views: List[str] = ['lateral', 'medial'],
                                       colormap: str = 'hot',
                                       save_path: Optional[Path] = None,
                                       baseline_block: int = 0,
                                       figure_size: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot multiple blocks of grand average brain data in a single figure.
    """
    
    # Get subjects info for title
    if subjects_subset is None:
        subjects_to_use = storage.get_condition_subjects(condition)
    else:
        condition_subjects = set(storage.get_condition_subjects(condition))
        subjects_to_use = [sub for sub in subjects_subset if sub in condition_subjects]
    
    n_subjects = len(subjects_to_use)
    n_blocks = len(blocks_to_plot)
    n_views = len(views)
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    # Create figure with subplots
    fig = plt.figure(figsize=figure_size)
    
    # Create grid: blocks x views
    gs = GridSpec(n_blocks, n_views, figure=fig, hspace=0.3, wspace=0.1)
    
    # Store min/max values across all blocks for consistent scaling
    all_data_values = []
    stcs = {}
    
    # First pass: compute all STCs and collect data values
    for block_id in blocks_to_plot:
        grand_avg_stc = storage.compute_grand_average_stc(condition, block_id, subjects_subset)
        stcs[block_id] = grand_avg_stc
        all_data_values.extend(grand_avg_stc.data[:, 0])
    
    # Set consistent color limits
    vmin, vmax = np.percentile(all_data_values, [5, 95])
    
    # Plot each block and view combination
    for block_idx, block_id in enumerate(blocks_to_plot):
        grand_avg_stc = stcs[block_id]
        
        for view_idx, view in enumerate(views):
            ax = fig.add_subplot(gs[block_idx, view_idx])
            
            # Create brain plot for this view
            brain = grand_avg_stc.plot(
                subject=storage.target_subject,
                hemi='both',
                views=[view],
                colormap=colormap,
                clim={'kind': 'value', 'lims': [vmin, vmax*0.3, vmax]},
                size=(400, 400),
                background='white',
                foreground='black'
            )
            
            # Get the brain image and add to subplot
            screenshot = brain.screenshot()
            ax.imshow(screenshot)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add titles
            if block_idx == 0:
                ax.set_title(f'{view.capitalize()} View', fontsize=12, fontweight='bold')
            if view_idx == 0:
                ax.set_ylabel(f'Block {block_id}', fontsize=12, fontweight='bold', rotation=0, 
                             labelpad=50, verticalalignment='center')
            
            brain.close()
    
    # Add overall title
    fig.suptitle(f'{condition} Grand Average (n={n_subjects}) - {storage.freq_band[0]}-{storage.freq_band[1]}Hz - {method_str}\n'
                 f'Baseline: Block {baseline_block}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Power Change (%)', rotation=270, labelpad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-block grand average brain to {save_path}")
    
    plt.show()


def plot_multi_block_condition_difference_brain(storage: MEGGrandAverageStorage,
                                              condition1: str, condition2: str,
                                              blocks_to_plot: List[int],
                                              subjects_subset1: Optional[List[str]] = None,
                                              subjects_subset2: Optional[List[str]] = None,
                                              views: List[str] = ['lateral', 'medial'],
                                              colormap: str = 'RdBu_r',
                                              save_path: Optional[Path] = None,
                                              baseline_block: int = 0,
                                              figure_size: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot multiple blocks of condition difference brain data in a single figure.
    """
    
    n_blocks = len(blocks_to_plot)
    n_views = len(views)
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    # Create figure
    fig = plt.figure(figsize=figure_size)
    gs = GridSpec(n_blocks, n_views, figure=fig, hspace=0.3, wspace=0.1)
    
    # Collect all difference data for consistent scaling
    all_diff_values = []
    stc_diffs = {}
    
    for block_id in blocks_to_plot:
        stc_diff = storage.compute_condition_difference_stc(
            condition1, condition2, block_id, subjects_subset1, subjects_subset2
        )
        stc_diffs[block_id] = stc_diff
        all_diff_values.extend(stc_diff.data[:, 0])
    
    # Set symmetric color limits for difference plot
    abs_max = np.percentile(np.abs(all_diff_values), 95)
    vmin, vmax = -abs_max, abs_max
    
    # Plot each block and view
    for block_idx, block_id in enumerate(blocks_to_plot):
        stc_diff = stc_diffs[block_id]
        
        for view_idx, view in enumerate(views):
            ax = fig.add_subplot(gs[block_idx, view_idx])
            
            brain = stc_diff.plot(
                subject=storage.target_subject,
                hemi='both',
                views=[view],
                colormap=colormap,
                clim={'kind': 'value', 'lims': [vmin, 0, vmax]},
                size=(400, 400),
                background='white',
                foreground='black'
            )
            
            screenshot = brain.screenshot()
            ax.imshow(screenshot)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if block_idx == 0:
                ax.set_title(f'{view.capitalize()} View', fontsize=12, fontweight='bold')
            if view_idx == 0:
                ax.set_ylabel(f'Block {block_id}', fontsize=12, fontweight='bold', rotation=0,
                             labelpad=50, verticalalignment='center')
            
            brain.close()
    
    # Add title and colorbar
    fig.suptitle(f'{condition1} - {condition2} Difference - {storage.freq_band[0]}-{storage.freq_band[1]}Hz - {method_str}\n'
                 f'Baseline: Block {baseline_block}', fontsize=14, fontweight='bold')
    
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Power Difference (%)', rotation=270, labelpad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-block condition difference brain to {save_path}")
    
    plt.show()


def plot_single_view_multi_block(storage: MEGGrandAverageStorage,
                                condition: str,
                                blocks_to_plot: List[int],
                                view: str = 'dorsal',
                                subjects_subset: Optional[List[str]] = None,
                                colormap: str = 'hot',
                                save_path: Optional[Path] = None,
                                baseline_block: int = 0,
                                figure_size: Tuple[int, int] = (15, 4)) -> None:
    """
    Plot multiple blocks in a single row for one specific view (e.g., dorsal-only).
    """
    
    # Get subjects info
    if subjects_subset is None:
        subjects_to_use = storage.get_condition_subjects(condition)
    else:
        condition_subjects = set(storage.get_condition_subjects(condition))
        subjects_to_use = [sub for sub in subjects_subset if sub in condition_subjects]
    
    n_subjects = len(subjects_to_use)
    n_blocks = len(blocks_to_plot)
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    # Create figure
    fig, axes = plt.subplots(1, n_blocks, figsize=figure_size)
    if n_blocks == 1:
        axes = [axes]
    
    # Collect all data for consistent scaling
    all_data_values = []
    stcs = {}
    
    for block_id in blocks_to_plot:
        grand_avg_stc = storage.compute_grand_average_stc(condition, block_id, subjects_subset)
        stcs[block_id] = grand_avg_stc
        all_data_values.extend(grand_avg_stc.data[:, 0])
    
    vmin, vmax = np.percentile(all_data_values, [5, 95])
    
    # Plot each block
    for idx, block_id in enumerate(blocks_to_plot):
        grand_avg_stc = stcs[block_id]
        
        brain = grand_avg_stc.plot(
            subject=storage.target_subject,
            hemi='both',
            views=[view],
            colormap=colormap,
            clim={'kind': 'value', 'lims': [vmin, vmax*0.3, vmax]},
            size=(400, 400),
            background='white',
            foreground='black'
        )
        
        screenshot = brain.screenshot()
        axes[idx].imshow(screenshot)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].set_title(f'Block {block_id}', fontsize=12, fontweight='bold')
        
        brain.close()
    
    # Add overall title
    fig.suptitle(f'{condition} Grand Average - {view.capitalize()} View (n={n_subjects})\n'
                 f'{storage.freq_band[0]}-{storage.freq_band[1]}Hz - {method_str}, Baseline: Block {baseline_block}', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Power Change (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved single-view multi-block brain to {save_path}")
    
    plt.show()


def run_complete_grand_average_analysis_modified(analyzer: MEGPSDAnalyzer,
                                               output_dir: Path,
                                               baseline_block: int = 0,
                                               views_to_plot: List[str] = ['lateral', 'medial', 'dorsal'],
                                               create_multi_block_plots: bool = True,
                                               create_single_view_plots: bool = True,
                                               create_individual_plots: bool = False) -> Dict:
    """Modified analysis function with multi-block plotting options and method names."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    storage = analyzer.grand_average_storage
    conditions = storage.get_available_conditions()
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    if len(conditions) == 0:
        logger.error("No conditions found in storage!")
        return {}
    
    # Print analysis summary
    print("\n=== Grand Average Analysis Summary ===")
    print(f"Target space: {storage.target_subject}")
    print(f"Frequency band: {storage.freq_band[0]}-{storage.freq_band[1]} Hz")
    print(f"Method: {method_str}")
    print(f"Number of blocks: {storage.n_blocks}")
    print(f"Total subjects: {len(storage.subjects_data)}")
    
    for condition in conditions:
        subjects = storage.get_condition_subjects(condition)
        print(f"{condition}: {len(subjects)} subjects ({subjects})")
    
    # Determine blocks to plot (excluding baseline)
    blocks_to_plot = [i for i in range(storage.n_blocks) if i != baseline_block]
    
    # 1. Create multi-block, multi-view plots (new main feature)
    if create_multi_block_plots:
        print("\nCreating multi-block plots...")
        for condition in conditions:
            # Multi-view grid plot
            save_path = output_dir / f'multi_block_grid_{condition}_{analyzer.freq_band[0]}-{analyzer.freq_band[1]}Hz_{method_str}.png'
            plot_multi_block_grand_average_brain(
                storage, condition, blocks_to_plot,
                views=views_to_plot, save_path=save_path, baseline_block=baseline_block
            )
    
    # 2. Create single-view, multi-block plots (e.g., dorsal-only across blocks)
    if create_single_view_plots:
        print("\nCreating single-view multi-block plots...")
        for condition in conditions:
            for view in views_to_plot:
                save_path = output_dir / f'multi_block_{view}_{condition}_{analyzer.freq_band[0]}-{analyzer.freq_band[1]}Hz_{method_str}.png'
                plot_single_view_multi_block(
                    storage, condition, blocks_to_plot, view=view,
                    save_path=save_path, baseline_block=baseline_block
                )
    
    # 3. Create condition difference plots
    if len(conditions) == 2:
        cond1, cond2 = conditions
        
        if create_multi_block_plots:
            # Multi-block difference plot
            save_path = output_dir / f'multi_block_difference_{cond1}_vs_{cond2}_{analyzer.freq_band[0]}-{analyzer.freq_band[1]}Hz_{method_str}.png'
            plot_multi_block_condition_difference_brain(
                storage, cond1, cond2, blocks_to_plot,
                views=views_to_plot, save_path=save_path, baseline_block=baseline_block
            )
    
    # 4. Create individual block plots (original functionality)
    if create_individual_plots:
        print("\nCreating individual block plots...")
        for condition in conditions:
            for block_id in blocks_to_plot:
                save_path = output_dir / f'individual_{condition}_block{block_id}_{analyzer.freq_band[0]}-{analyzer.freq_band[1]}Hz_{method_str}.png'
                brain = plot_grand_average_brain(storage, condition, block_id, 
                                               views=['lateral', 'medial'], save_path=save_path)
                brain.close()
    
    # 5. Plot time course (unchanged)
    time_course_path = output_dir / f'time_course_comparison_{analyzer.freq_band[0]}-{analyzer.freq_band[1]}Hz_{method_str}.png'
    plot_time_course_comparison(storage, conditions, save_path=time_course_path)
    
    return {
        'conditions': conditions,
        'n_subjects_per_condition': {cond: len(storage.get_condition_subjects(cond)) for cond in conditions},
        'total_subjects': len(storage.subjects_data),
        'freq_band': storage.freq_band,
        'target_subject': storage.target_subject,
        'blocks_plotted': blocks_to_plot,
        'views_used': views_to_plot,
        'method_name': method_str
    }

#%% Individual parts
# Individual Subject Plotting Functions
# Add these functions to your existing script

def plot_individual_subject_brain(storage: MEGGrandAverageStorage,
                                 subject_id: str, 
                                 block_id: int,
                                 colormap: str = 'hot',
                                 views: List[str] = ['lateral', 'medial'],
                                 save_path: Optional[Path] = None,
                                 baseline_block: int = 0) -> mne.viz.Brain:
    """
    Plot brain data for a single subject and single block.
    
    Parameters
    ----------
    storage : MEGGrandAverageStorage
        Storage object containing subject data
    subject_id : str
        Subject identifier
    block_id : int
        Block to plot
    colormap : str
        Colormap for visualization
    views : list
        Brain views to display
    save_path : Path, optional
        Path to save the figure
    baseline_block : int
        Baseline block used for normalization
        
    Returns
    -------
    mne.viz.Brain
        Brain visualization object
    """
    if subject_id not in storage.subjects_data:
        raise ValueError(f"Subject {subject_id} not found in storage")
    
    if block_id >= len(storage.subjects_data[subject_id]):
        raise ValueError(f"Block {block_id} not available for subject {subject_id}")
    
    # Get the SourceEstimate for this subject and block
    stc = storage.subjects_data[subject_id][block_id]
    condition = storage.condition_mapping[subject_id]
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    # Plot the brain
    brain = stc.plot(
        subject=storage.target_subject,
        hemi='both',
        views=views,
        colormap=colormap,
        title=f'{subject_id} ({condition}) - Block {block_id} - {storage.freq_band[0]}-{storage.freq_band[1]}Hz - {method_str}\nBaseline: Block {baseline_block}'
    )
    
    if save_path:
        brain.save_image(save_path)
        logger.info(f"Saved individual subject brain plot to {save_path}")
    
    # Log data range for this subject
    data_range = stc.data[:, 0]
    logger.info(f"{subject_id} Block {block_id} range: {data_range.min():.2f}% to {data_range.max():.2f}%")
    
    return brain


def plot_individual_subject_multi_block(storage: MEGGrandAverageStorage,
                                       subject_id: str,
                                       blocks_to_plot: List[int],
                                       views: List[str] = ['lateral', 'medial'],
                                       colormap: str = 'hot',
                                       save_path: Optional[Path] = None,
                                       baseline_block: int = 0,
                                       figure_size: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot multiple blocks for a single subject in one figure.
    
    Parameters
    ----------
    storage : MEGGrandAverageStorage
        Storage object containing subject data
    subject_id : str
        Subject identifier
    blocks_to_plot : list
        List of block indices to plot
    views : list
        Brain views to display
    colormap : str
        Colormap for visualization
    save_path : Path, optional
        Path to save the figure
    baseline_block : int
        Baseline block used for normalization
    figure_size : tuple
        Figure size (width, height)
    """
    if subject_id not in storage.subjects_data:
        raise ValueError(f"Subject {subject_id} not found in storage")
    
    condition = storage.condition_mapping[subject_id]
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    n_blocks = len(blocks_to_plot)
    n_views = len(views)
    
    # Create figure
    fig = plt.figure(figsize=figure_size)
    gs = GridSpec(n_blocks, n_views, figure=fig, hspace=0.3, wspace=0.1)
    
    # Collect all data values for consistent scaling
    all_data_values = []
    stcs = {}
    
    for block_id in blocks_to_plot:
        if block_id >= len(storage.subjects_data[subject_id]):
            logger.warning(f"Block {block_id} not available for subject {subject_id}, skipping")
            continue
        stc = storage.subjects_data[subject_id][block_id]
        stcs[block_id] = stc
        all_data_values.extend(stc.data[:, 0])
    
    # Set consistent color limits
    if all_data_values:
        vmin, vmax = np.percentile(all_data_values, [5, 95])
    else:
        logger.error(f"No valid data found for subject {subject_id}")
        return
    
    # Plot each block and view combination
    for block_idx, block_id in enumerate(blocks_to_plot):
        if block_id not in stcs:
            continue
            
        stc = stcs[block_id]
        
        for view_idx, view in enumerate(views):
            ax = fig.add_subplot(gs[block_idx, view_idx])
            
            # Create brain plot for this view
            brain = stc.plot(
                subject=storage.target_subject,
                hemi='both',
                views=[view],
                colormap=colormap,
                clim={'kind': 'value', 'lims': [vmin, vmax*0.3, vmax]},
                size=(400, 400),
                background='white',
                foreground='black'
            )
            
            # Get screenshot and add to subplot
            screenshot = brain.screenshot()
            ax.imshow(screenshot)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add titles
            if block_idx == 0:
                ax.set_title(f'{view.capitalize()} View', fontsize=12, fontweight='bold')
            if view_idx == 0:
                ax.set_ylabel(f'Block {block_id}', fontsize=12, fontweight='bold', rotation=0,
                             labelpad=50, verticalalignment='center')
            
            brain.close()
    
    # Add overall title
    fig.suptitle(f'{subject_id} ({condition}) - {storage.freq_band[0]}-{storage.freq_band[1]}Hz - {method_str}\n'
                 f'Baseline: Block {baseline_block}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Power Change (%)', rotation=270, labelpad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved individual subject multi-block plot to {save_path}")
    
    plt.show()


def plot_condition_subjects_comparison(storage: MEGGrandAverageStorage,
                                     condition: str,
                                     block_id: int,
                                     max_subjects: int = 6,
                                     views: List[str] = ['lateral', 'medial'],
                                     colormap: str = 'hot',
                                     save_path: Optional[Path] = None,
                                     baseline_block: int = 0,
                                     figure_size: Tuple[int, int] = (15, 12)) -> None:
    """
    Plot multiple subjects from the same condition for comparison.
    
    Parameters
    ----------
    storage : MEGGrandAverageStorage
        Storage object containing subject data
    condition : str
        Condition to plot subjects from
    block_id : int
        Block to plot
    max_subjects : int
        Maximum number of subjects to plot
    views : list
        Brain views to display
    colormap : str
        Colormap for visualization
    save_path : Path, optional
        Path to save the figure
    baseline_block : int
        Baseline block used for normalization
    figure_size : tuple
        Figure size (width, height)
    """
    # Get subjects for this condition
    subjects = storage.get_condition_subjects(condition)
    
    if len(subjects) == 0:
        logger.error(f"No subjects found for condition {condition}")
        return
    
    # Limit number of subjects if needed
    subjects_to_plot = subjects[:max_subjects]
    if len(subjects) > max_subjects:
        logger.info(f"Plotting first {max_subjects} of {len(subjects)} subjects for condition {condition}")
    
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    n_subjects = len(subjects_to_plot)
    n_views = len(views)
    
    # Create figure
    fig = plt.figure(figsize=figure_size)
    gs = GridSpec(n_subjects, n_views, figure=fig, hspace=0.3, wspace=0.1)
    
    # Collect all data values for consistent scaling across subjects
    all_data_values = []
    stcs = {}
    
    for subject_id in subjects_to_plot:
        if block_id >= len(storage.subjects_data[subject_id]):
            logger.warning(f"Block {block_id} not available for subject {subject_id}, skipping")
            continue
        stc = storage.subjects_data[subject_id][block_id]
        stcs[subject_id] = stc
        all_data_values.extend(stc.data[:, 0])
    
    # Set consistent color limits
    if all_data_values:
        vmin, vmax = np.percentile(all_data_values, [5, 95])
    else:
        logger.error(f"No valid data found for any subject in condition {condition}")
        return
    
    # Plot each subject and view combination
    for subj_idx, subject_id in enumerate(subjects_to_plot):
        if subject_id not in stcs:
            continue
            
        stc = stcs[subject_id]
        
        for view_idx, view in enumerate(views):
            ax = fig.add_subplot(gs[subj_idx, view_idx])
            
            # Create brain plot for this view
            brain = stc.plot(
                subject=storage.target_subject,
                hemi='both',
                views=[view],
                colormap=colormap,
                clim={'kind': 'value', 'lims': [vmin, vmax*0.3, vmax]},
                size=(400, 400),
                background='white',
                foreground='black'
            )
            
            # Get screenshot and add to subplot
            screenshot = brain.screenshot()
            ax.imshow(screenshot)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add titles
            if subj_idx == 0:
                ax.set_title(f'{view.capitalize()} View', fontsize=12, fontweight='bold')
            if view_idx == 0:
                ax.set_ylabel(f'{subject_id}', fontsize=12, fontweight='bold', rotation=0,
                             labelpad=50, verticalalignment='center')
            
            brain.close()
    
    # Add overall title
    fig.suptitle(f'{condition} Condition - Block {block_id} - {storage.freq_band[0]}-{storage.freq_band[1]}Hz - {method_str}\n'
                 f'Individual Subjects Comparison (Baseline: Block {baseline_block})', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Power Change (%)', rotation=270, labelpad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved condition subjects comparison plot to {save_path}")
    
    plt.show()


def plot_subject_time_course(storage: MEGGrandAverageStorage,
                           subject_id: str,
                           save_path: Optional[Path] = None,
                           baseline_block: int = 0) -> None:
    """
    Plot time course for a single subject across all blocks.
    
    Parameters
    ----------
    storage : MEGGrandAverageStorage
        Storage object containing subject data
    subject_id : str
        Subject identifier
    save_path : Path, optional
        Path to save the figure
    baseline_block : int
        Baseline block used for normalization
    """
    if subject_id not in storage.subjects_data:
        raise ValueError(f"Subject {subject_id} not found in storage")
    
    condition = storage.condition_mapping[subject_id]
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    # Calculate mean power for each block
    block_means = []
    block_stds = []
    
    for block_id in range(len(storage.subjects_data[subject_id])):
        stc = storage.subjects_data[subject_id][block_id]
        block_data = stc.data[:, 0]
        block_means.append(np.mean(block_data))
        block_stds.append(np.std(block_data))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    blocks = np.arange(len(block_means))
    
    plt.errorbar(blocks, block_means, yerr=block_stds, 
                marker='o', capsize=5, linewidth=2, markersize=8,
                label=f'{subject_id} ({condition})')
    
    # Highlight baseline block
    plt.axvline(x=baseline_block, color='red', linestyle='--', alpha=0.7, 
                label=f'Baseline (Block {baseline_block})')
    
    plt.xlabel('Block')
    plt.ylabel('Power Change (%)')
    plt.title(f'{subject_id} Time Course - {condition} Condition\n'
              f'{storage.freq_band[0]}-{storage.freq_band[1]} Hz - {method_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved subject time course plot to {save_path}")
    
    plt.show()


def run_individual_subject_analysis(analyzer: MEGPSDAnalyzer,
                                   output_dir: Path,
                                   baseline_block: int = 0,
                                   analysis_type: str = 'all',
                                   specific_subjects: Optional[List[str]] = None,
                                   max_subjects_per_condition: int = 6) -> Dict:
    """
    Run comprehensive individual subject analysis.
    
    Parameters
    ----------
    analyzer : MEGPSDAnalyzer
        Analyzer object with loaded data
    output_dir : Path
        Output directory for plots
    baseline_block : int
        Baseline block for normalization
    analysis_type : str
        Type of analysis: 'all', 'multi_block', 'condition_comparison', 'time_course'
    specific_subjects : list, optional
        Specific subjects to analyze (if None, analyze all)
    max_subjects_per_condition : int
        Maximum subjects per condition for comparison plots
        
    Returns
    -------
    dict
        Summary of analysis performed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    storage = analyzer.grand_average_storage
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    # Get subjects to analyze
    if specific_subjects is None:
        subjects_to_analyze = list(storage.subjects_data.keys())
    else:
        subjects_to_analyze = [s for s in specific_subjects if s in storage.subjects_data]
    
    if len(subjects_to_analyze) == 0:
        logger.error("No valid subjects found for analysis")
        return {}
    
    blocks_to_plot = [i for i in range(storage.n_blocks) if i != baseline_block]
    
    print(f"\n=== Individual Subject Analysis ===")
    print(f"Analyzing {len(subjects_to_analyze)} subjects")
    print(f"Method: {method_str}")
    print(f"Baseline block: {baseline_block}")
    print(f"Analysis type: {analysis_type}")
    
    analysis_summary = {
        'subjects_analyzed': subjects_to_analyze,
        'method_name': method_str,
        'baseline_block': baseline_block,
        'analysis_type': analysis_type,
        'plots_created': []
    }
    
    # 1. Multi-block plots for individual subjects
    if analysis_type in ['all', 'multi_block']:
        print("\nCreating multi-block plots for individual subjects...")
        for subject_id in subjects_to_analyze:
            condition = storage.condition_mapping[subject_id]
            save_path = output_dir / f'individual_multi_block_{subject_id}_{condition}_{storage.freq_band[0]}-{storage.freq_band[1]}Hz_{method_str}.png'
            
            plot_individual_subject_multi_block(
                storage, subject_id, blocks_to_plot,
                views=['lateral', 'medial', 'dorsal'],
                save_path=save_path, baseline_block=baseline_block
            )
            analysis_summary['plots_created'].append(f'multi_block_{subject_id}')
    
    # 2. Condition comparison plots
    if analysis_type in ['all', 'condition_comparison']:
        print("\nCreating condition comparison plots...")
        conditions = storage.get_available_conditions()
        
        for condition in conditions:
            for block_id in blocks_to_plot:
                save_path = output_dir / f'condition_comparison_{condition}_block{block_id}_{storage.freq_band[0]}-{storage.freq_band[1]}Hz_{method_str}.png'
                
                plot_condition_subjects_comparison(
                    storage, condition, block_id,
                    max_subjects=max_subjects_per_condition,
                    save_path=save_path, baseline_block=baseline_block
                )
                analysis_summary['plots_created'].append(f'condition_comparison_{condition}_block{block_id}')
    
    # 3. Individual time courses
    if analysis_type in ['all', 'time_course']:
        print("\nCreating individual time course plots...")
        for subject_id in subjects_to_analyze:
            condition = storage.condition_mapping[subject_id]
            save_path = output_dir / f'time_course_{subject_id}_{condition}_{storage.freq_band[0]}-{storage.freq_band[1]}Hz_{method_str}.png'
            
            plot_subject_time_course(
                storage, subject_id, save_path=save_path, baseline_block=baseline_block
            )
            analysis_summary['plots_created'].append(f'time_course_{subject_id}')
    
    print(f"\nIndividual subject analysis complete!")
    print(f"Created {len(analysis_summary['plots_created'])} plots in {output_dir}")
    
    return analysis_summary

# %%

def plot_subject_time_course(storage: MEGGrandAverageStorage,
                           subject_id: str,
                           save_path: Optional[Path] = None,
                           baseline_block: int = 0) -> None:
    """
    Plot time course for a single subject across all blocks.
    
    Parameters
    ----------
    storage : MEGGrandAverageStorage
        Storage object containing subject data
    subject_id : str
        Subject identifier
    save_path : Path, optional
        Path to save the figure
    baseline_block : int
        Baseline block used for normalization
    """
    if subject_id not in storage.subjects_data:
        raise ValueError(f"Subject {subject_id} not found in storage")
    
    condition = storage.condition_mapping[subject_id]
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    # Calculate mean power for each block
    block_means = []
    block_stds = []
    
    for block_id in range(len(storage.subjects_data[subject_id])):
        stc = storage.subjects_data[subject_id][block_id]
        block_data = stc.data[:, 0]
        block_means.append(np.mean(block_data))
        block_stds.append(np.std(block_data))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    blocks = np.arange(len(block_means))
    
    plt.errorbar(blocks, block_means, yerr=block_stds, 
                marker='o', capsize=5, linewidth=2, markersize=8,
                label=f'{subject_id} ({condition})')
    
    # Highlight baseline block
    plt.axvline(x=baseline_block, color='red', linestyle='--', alpha=0.7, 
                label=f'Baseline (Block {baseline_block})')
    
    plt.xlabel('Block')
    plt.ylabel('Power Change (%)')
    plt.title(f'{subject_id} Time Course - {condition} Condition\n'
              f'{storage.freq_band[0]}-{storage.freq_band[1]} Hz - {method_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved subject time course plot to {save_path}")
    
    plt.show()


def plot_individual_subjects_time_course_comparison(storage: MEGGrandAverageStorage,
                                                  subjects_to_plot: Optional[List[str]] = None,
                                                  condition: Optional[str] = None,
                                                  max_subjects: int = 8,
                                                  save_path: Optional[Path] = None,
                                                  baseline_block: int = 0,
                                                  show_grand_average: bool = True) -> None:
    """
    Plot time course comparison for multiple individual subjects, similar to grand average style.
    
    Parameters
    ----------
    storage : MEGGrandAverageStorage
        Storage object containing subject data
    subjects_to_plot : list, optional
        Specific subjects to plot. If None, uses subjects from condition
    condition : str, optional
        Condition to plot subjects from. If None, plots from all conditions
    max_subjects : int
        Maximum number of subjects to plot
    save_path : Path, optional
        Path to save the figure
    baseline_block : int
        Baseline block used for normalization
    show_grand_average : bool
        Whether to overlay the grand average for comparison
    """
    
    # Determine which subjects to plot
    if subjects_to_plot is None:
        if condition is not None:
            subjects_to_plot = storage.get_condition_subjects(condition)[:max_subjects]
        else:
            # Take subjects from all conditions
            all_subjects = list(storage.subjects_data.keys())
            subjects_to_plot = all_subjects[:max_subjects]
    else:
        subjects_to_plot = subjects_to_plot[:max_subjects]
    
    if len(subjects_to_plot) == 0:
        logger.error("No subjects found for time course plotting")
        return
    
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for subjects
    colors = plt.cm.tab10(np.linspace(0, 1, len(subjects_to_plot)))
    
    # Plot individual subjects
    for idx, subject_id in enumerate(subjects_to_plot):
        if subject_id not in storage.subjects_data:
            continue
            
        subject_condition = storage.condition_mapping[subject_id]
        
        # Calculate time course for this subject
        block_means = []
        block_stds = []
        
        for block_id in range(storage.n_blocks):
            stc = storage.subjects_data[subject_id][block_id]
            block_data = stc.data[:, 0]
            block_means.append(np.mean(block_data))
            block_stds.append(np.std(block_data))
        
        blocks = np.arange(len(block_means))
        
        # Plot with individual styling
        plt.plot(blocks, block_means, 
                marker='o', linewidth=1.5, markersize=6,
                color=colors[idx], alpha=0.7,
                label=f'{subject_id} ({subject_condition})')
    
    # Optionally overlay grand average for comparison
    if show_grand_average and condition is not None:
        # Calculate grand average for this condition
        grand_avg_means = []
        grand_avg_sems = []
        
        for block_id in range(storage.n_blocks):
            grand_avg_stc = storage.compute_grand_average_stc(condition, block_id)
            block_data = grand_avg_stc.data[:, 0]
            grand_avg_means.append(np.mean(block_data))
            grand_avg_sems.append(np.std(block_data) / np.sqrt(len(block_data)))
        
        blocks = np.arange(len(grand_avg_means))
        plt.errorbar(blocks, grand_avg_means, yerr=grand_avg_sems,
                    marker='s', capsize=5, linewidth=3, markersize=8,
                    color='black', alpha=0.9,
                    label=f'{condition} Grand Average')
    
    # Highlight baseline block
    plt.axvline(x=baseline_block, color='red', linestyle='--', alpha=0.7, linewidth=2,
                label=f'Baseline (Block {baseline_block})')
    
    # Styling to match your grand average plots
    plt.xlabel('Block', fontsize=12)
    plt.ylabel('Power Change (%)', fontsize=12)
    
    # Create title
    if condition is not None:
        title = f'Individual Subjects Time Course - {condition} Condition'
    else:
        title = f'Individual Subjects Time Course - Multiple Conditions'
    
    title += f'\n{storage.freq_band[0]}-{storage.freq_band[1]} Hz - {method_str}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Legend with better positioning
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved individual subjects time course comparison to {save_path}")
    
    plt.show()


def plot_condition_split_time_course_comparison(storage: MEGGrandAverageStorage,
                                              max_subjects_per_condition: int = 6,
                                              save_path: Optional[Path] = None,
                                              baseline_block: int = 0,
                                              show_grand_averages: bool = True) -> None:
    """
    Plot time course comparison with subjects split by condition, matching grand average style.
    
    Parameters
    ----------
    storage : MEGGrandAverageStorage
        Storage object containing subject data
    max_subjects_per_condition : int
        Maximum subjects per condition to plot
    save_path : Path, optional
        Path to save the figure
    baseline_block : int
        Baseline block used for normalization
    show_grand_averages : bool
        Whether to overlay grand averages for each condition
    """
    
    conditions = storage.get_available_conditions()
    if len(conditions) == 0:
        logger.error("No conditions found for time course plotting")
        return
    
    method_str = storage.method_name if storage.method_name else "unknown-method"
    
    plt.figure(figsize=(14, 8))
    
    # Define colors for conditions
    condition_colors = {'Congruent': 'blue', 'Incongruent': 'red'}  # Adjust as needed
    if len(conditions) > len(condition_colors):
        # Generate more colors if needed
        extra_colors = plt.cm.Set1(np.linspace(0, 1, len(conditions)))
        for i, cond in enumerate(conditions):
            if cond not in condition_colors:
                condition_colors[cond] = extra_colors[i]
    
    # Plot individual subjects by condition
    for condition in conditions:
        subjects = storage.get_condition_subjects(condition)[:max_subjects_per_condition]
        base_color = condition_colors.get(condition, 'gray')
        
        # Convert color to RGB for alpha manipulation
        if isinstance(base_color, str):
            base_rgb = plt.cm.colors.to_rgb(base_color)
        else:
            base_rgb = base_color[:3]
        
        for idx, subject_id in enumerate(subjects):
            if subject_id not in storage.subjects_data:
                continue
            
            # Calculate time course for this subject
            block_means = []
            
            for block_id in range(storage.n_blocks):
                stc = storage.subjects_data[subject_id][block_id]
                block_data = stc.data[:, 0]
                block_means.append(np.mean(block_data))
            
            blocks = np.arange(len(block_means))
            
            # Plot individual subject with lighter color and thinner line
            alpha_val = 0.4 if show_grand_averages else 0.7
            line_width = 1.0 if show_grand_averages else 1.5
            
            plt.plot(blocks, block_means,
                    color=base_rgb, alpha=alpha_val, linewidth=line_width,
                    marker='o', markersize=4)
    
    # Plot grand averages if requested
    if show_grand_averages:
        for condition in conditions:
            # Calculate grand average for this condition
            grand_avg_means = []
            grand_avg_sems = []
            
            for block_id in range(storage.n_blocks):
                grand_avg_stc = storage.compute_grand_average_stc(condition, block_id)
                block_data = grand_avg_stc.data[:, 0]
                grand_avg_means.append(np.mean(block_data))
                
                # Calculate SEM across subjects for this block
                subjects = storage.get_condition_subjects(condition)
                subject_means = []
                for subject_id in subjects:
                    if subject_id in storage.subjects_data:
                        subj_stc = storage.subjects_data[subject_id][block_id]
                        subj_mean = np.mean(subj_stc.data[:, 0])
                        subject_means.append(subj_mean)
                
                if len(subject_means) > 1:
                    grand_avg_sems.append(np.std(subject_means) / np.sqrt(len(subject_means)))
                else:
                    grand_avg_sems.append(0)
            
            blocks = np.arange(len(grand_avg_means))
            base_color = condition_colors.get(condition, 'gray')
            
            plt.errorbar(blocks, grand_avg_means, yerr=grand_avg_sems,
                        marker='o', capsize=5, linewidth=3, markersize=8,
                        color=base_color, label=f'{condition} (Grand Average)',
                        alpha=0.9)
    
    # Add custom legend entries for individual subjects
    if not show_grand_averages:
        for condition in conditions:
            base_color = condition_colors.get(condition, 'gray')
            plt.plot([], [], color=base_color, alpha=0.7, linewidth=1.5,
                    label=f'{condition} (Individual Subjects)')
    
    # Highlight baseline block
    plt.axvline(x=baseline_block, color='black', linestyle='--', alpha=0.7, linewidth=2,
                label=f'Baseline (Block {baseline_block})')
    
    # Styling to match your existing plots
    plt.xlabel('Block', fontsize=12)
    plt.ylabel('Power Change (%)', fontsize=12)
    
    title = f'Individual Subjects vs Grand Average Time Course\n'
    title += f'{storage.freq_band[0]}-{storage.freq_band[1]} Hz - {method_str}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved condition split time course comparison to {save_path}")
    
    plt.show()
#%% 

# UPDATED MAIN EXECUTION SECTION
if __name__ == "__main__":
    #%% Load data and initialize
    WL_data, WL_subs = process_WL_data(m=0, min_seq_length=2, plot_flag=0)
    
    data_root = Path('Z:/Data/derivatives/spectral_analysis')
    subjects_dir = Path('C:/fs_data')
    freq_min, freq_max = 8, 16
    baseline_block = 0
    
    # UPDATED: Specify the method you want to analyze
    # Examples:
    # method = 'multitaper'
    # method = 'eLORETA-d8-l3-snr3'
    # method = 'dSPM-d8-l3-snr3'
    method = 'multitaper-1.0-30.0Hz_eLORETA-d8-l3-snr3'  # Change this to your desired method
    
    psd_pattern = f"*psd-{method}.h5"
    print(f"Looking for files matching pattern: {psd_pattern}")
    
    # UPDATED: Use new function to find files in subfolders
    psd_files_dict = find_psd_files_in_subfolders(data_root, psd_pattern)
    
    if len(psd_files_dict) == 0:
        print(f"ERROR: No PSD files found matching pattern '{psd_pattern}' in subfolders of {data_root}")
        print("Available subject folders:")
        subject_folders = [f.name for f in data_root.iterdir() if f.is_dir() and f.name.startswith('sub-')]
        print(subject_folders[:10])  # Show first 10
        exit()
    
    print(f"Found {len(psd_files_dict)} PSD files total")
    
    # Initialize analyzer
    analyzer = MEGPSDAnalyzer(
        subjects_dir=subjects_dir,
        freq_band=(freq_min, freq_max),
        target_subject='fsaverage'
    )
    
    processed_subjects = []
    failed_subjects = []
    
    # Create condition lookup (unchanged)
    condition_lookup = {}
    for wl_code in WL_subs['con_imp']:
        temp_sub = int(wl_code[-2:])
        sub_id = f'sub-{temp_sub:02d}'
        condition_lookup[sub_id] = 'Congruent'
    
    for wl_code in WL_subs['incon_imp']:
        temp_sub = int(wl_code[-2:])
        sub_id = f'sub-{temp_sub:02d}'
        condition_lookup[sub_id] = 'Incongruent'
    
    logger.info(f"Condition lookup created: {len(condition_lookup)} subjects")

    # UPDATED: Process subjects using the new file dictionary
    for subject_id, psd_file in psd_files_dict.items():
        try:
            # Check if this subject is in our condition mapping
            if subject_id not in condition_lookup:
                logger.warning(f"Skipping {subject_id}: not in condition mapping")
                failed_subjects.append(subject_id)
                continue
            
            # Load and process the data
            psd_data = analyzer.load_psd_data(psd_file)
            band_power = analyzer.extract_frequency_band(psd_data)
            normalized_power = analyzer.baseline_normalize(band_power, baseline_block)
            
            condition = condition_lookup[subject_id]
            
            # UPDATED: Extract method name from filename and store with data
            method_name = extract_method_from_filename(psd_file.name)
            
            # Store for grand averaging
            analyzer.add_subject_to_grand_average(
                subject_id, normalized_power,
                psd_data['vertices_lh'], psd_data['vertices_rh'],
                condition, method_name
            )
            
            processed_subjects.append((subject_id, condition))
            logger.info(f"Processed {subject_id} ({condition}) with method {method_name} - Total: {len(processed_subjects)}")
            
        except Exception as e:
            logger.error(f"Error processing {subject_id} (file: {psd_file}): {str(e)}")
            failed_subjects.append(subject_id)
            continue
    
    # Print storage summary
    summary = analyzer.get_storage_summary()
    print("\n=== Grand Average Storage Summary ===")
    print(f"Total subjects: {summary['total_subjects']}")
    print(f"Frequency band: {summary['freq_band']} Hz")
    print(f"Method: {summary['method_name']}")
    print(f"Blocks: {summary['n_blocks']}")
    print("\nConditions:")
    for condition, info in summary['conditions'].items():
        print(f"  {condition}: {info['n_subjects']} subjects")
        print(f"    Subjects: {info['subjects']}")
    
    if failed_subjects:
        print(f"\nFailed to process {len(failed_subjects)} subjects: {failed_subjects}")

#%% UPDATED: Run the analysis with method name in output folder
    method_str = analyzer.grand_average_storage.method_name or method
    output_dir = Path(f'grand_average_results_{method_str}_{freq_min}-{freq_max}Hz')

    # Create comprehensive multi-block plots with method name in filenames
    GA_output = run_complete_grand_average_analysis_modified(
        analyzer, 
        output_dir,
        baseline_block=baseline_block,
        views_to_plot=['lateral', 'medial', 'dorsal'],
        create_multi_block_plots=True,
        create_single_view_plots=True,
        create_individual_plots=False
    )

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Method analyzed: {method_str}")
    print(f"Frequency band: {freq_min}-{freq_max} Hz")

# %%
    specific_subject = 'sub-01'  # Change to any subject in your data
    if specific_subject in analyzer.grand_average_storage.subjects_data:
        single_subject_dir = Path('individual_results_single_subject')
        single_subject_dir.mkdir(parents=True, exist_ok=True)
        
        blocks_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Adjust based on your data
        save_path = single_subject_dir / f'{specific_subject}_multi_block.png'
        
        plot_individual_subject_multi_block(
            analyzer.grand_average_storage, 
            specific_subject, 
            blocks_to_plot,
            save_path=save_path,
            baseline_block=0
        )
    
    # Example 2: Compare subjects within a condition
    conditions = analyzer.grand_average_storage.get_available_conditions()
    if len(conditions) > 0:
        comparison_dir = Path('individual_results_condition_comparison')
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        for condition in conditions:
            save_path = comparison_dir / f'{condition}_subjects_block1.png'
            plot_condition_subjects_comparison(
                analyzer.grand_average_storage,
                condition,
                block_id=1,
                max_subjects=26,
                save_path=save_path
            )
    
    # Example 3: Run comprehensive individual analysis
    individual_dir = Path('individual_results_comprehensive')
    run_individual_subject_analysis(
        analyzer,
        individual_dir,
        baseline_block=0,
        analysis_type='all',  # or 'multi_block', 'condition_comparison', 'time_course'
        max_subjects_per_condition=26
    )
# NEW: Run the modified grand average analysis with different plotting options
#%%
output_dir = Path('grand_average_results_multi_block')

# Option 1: Create comprehensive multi-block plots with multiple views
GA_output = run_complete_grand_average_analysis_modified(
    analyzer, 
    output_dir,
    baseline_block=0,
    views_to_plot=['lateral', 'medial', 'dorsal'],  # Add dorsal view
    create_multi_block_plots=True,      # Grid of blocks x views
    create_single_view_plots=True,      # Dorsal-only across blocks, etc.
    create_individual_plots=False       # Skip individual block plots
)

# %%
time_course_dir = Path('individual_results_time_course')
time_course_dir.mkdir(parents=True, exist_ok=True)
    
for condition in conditions:
    save_path = time_course_dir / f'time_course_with_grand_avg_{condition}.png'
    plot_individual_subjects_time_course_comparison(
        analyzer.grand_average_storage,
        condition=condition,
        max_subjects=26,
        save_path=save_path,
        show_grand_average=True
    )
# %%
# Option 2: Create dorsal-only multi-block plots
dorsal_output_dir = Path('grand_average_results_dorsal_only')
dorsal_output_dir.mkdir(parents=True, exist_ok=True)

storage = analyzer.grand_average_storage
conditions = storage.get_available_conditions()
blocks_to_plot = [i for i in range(storage.n_blocks) if i != baseline_block]

# Create dorsal-only plots for each condition
for condition in conditions:
    save_path = dorsal_output_dir / f'dorsal_multi_block_{condition}_{freq_min}-{freq_max}Hz.png'
    plot_single_view_multi_block(
        storage, condition, blocks_to_plot, 
        view='dorsal', save_path=save_path, baseline_block=baseline_block,
        figure_size=(20, 5)  # Wide figure for multiple blocks
    )

# %%
# Option 3: Create custom view combinations
custom_output_dir = Path('grand_average_results_custom')
custom_output_dir.mkdir(parents=True, exist_ok=True)

# Example: Only lateral and dorsal views in a 2x11 grid
for condition in conditions:
    save_path = custom_output_dir / f'lateral_dorsal_grid_{condition}_{freq_min}-{freq_max}Hz.png'
    plot_multi_block_grand_average_brain(
        storage, condition, blocks_to_plot,
        views=['lateral', 'dorsal'],  # Only these two views
        save_path=save_path, 
        baseline_block=baseline_block,
        figure_size=(12, 8)  # Adjust size for 2 views
    )

# %%
# Option 4: Create difference plots with multiple views
if len(conditions) == 2:
    diff_output_dir = Path('grand_average_results_differences')
    diff_output_dir.mkdir(parents=True, exist_ok=True)
    
    cond1, cond2 = conditions
    save_path = diff_output_dir / f'difference_multi_block_{cond1}_vs_{cond2}_{freq_min}-{freq_max}Hz.png'
    
    plot_multi_block_condition_difference_brain(
        storage, cond1, cond2, blocks_to_plot,
        views=['lateral', 'medial', 'dorsal'],
        save_path=save_path,
        baseline_block=baseline_block,
        figure_size=(15, 10)
    )

print("\nAnalysis complete! Check the output directories for multi-block plots.")
# %%
