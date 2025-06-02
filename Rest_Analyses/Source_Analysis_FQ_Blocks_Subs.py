#%% 
import mne
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy import fft
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm  # for progress bars

from meg_analysis.Scripts.Behavior.WL_implicitSEplot_Extend import process_WL_data

from fooof import FOOOF
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
#%% 
'Scripts related to plotting effects dependant/locked on the word-list performance'
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
        
        motor_indices = get_motor_indices(vertices_lh, vertices_rh, labels_dir)
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


def get_motor_indices(vertices_lh, vertices_rh, labels_dir):
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
def motor_FFT_analysis(source_param, Condition = ['Congruent', 'Incongruent'], LR = ['left','right'] , ses = 'ses-1'):
    '''
    This script processes the source data of the motor cortex for each subject and condition.
    It calculates the FFT for each block and saves the data in a hdf5 file.
    If the subject already exists in the hdf5 file, it skips the subject.
    
    Inputs:
        Condition: list of conditions to process (default: ['Congruent', 'Incongruent'])
        LR: list of hemispheres to process (default: ['left','right'])
        ses: session to process (default: 'ses-1'), at the moment only ses-1 is implemented anyway, but might revisit in the future
        explicitness: list of explicitness to process (default: ['Explicit','Implicit']); planned but not yet implemented
    
    Outputs:
        None, the data is saved in a hdf5 file in the Data/Rest folder of the project directory.
            The file contains a section for each subject with average spectra per block separated by frequency
    '''
    # load behavioral data of all subjects      
    WL_data, WL_subs = process_WL_data(m=0, min_seq_length=2, plot_flag=0)
    
    # subject selection
    sub_lst     = {} # store subject labels for each condition
    sub_data    = {} # store data for each condition

    method = source_param['method']
    d      = source_param['depth']
    l      = source_param['loose']
    snr    = source_param['snr']

    # prepare variables/paths and set parameters
    subject_data = {}
    for c,ConIn in enumerate(Condition):
        for side_idx, side in enumerate(LR): # left only for now       
            # find the subjects for the relevant conditions and throw an error if the condition is invalid
            if ConIn == 'Congruent':
                sub_lst[c]  = ['sub-' + sub_id[2:4] for sub_id in WL_subs['con_imp']] 
                sub_data[c] = WL_data['con_imp']
            elif ConIn == 'Incongruent': 
                sub_lst[c]  = ['sub-' + sub_id[2:4] for sub_id in WL_subs['incon_imp']] # find the subjects in the incongruent implicit condition
                sub_data[c] = WL_data['incon_imp']
            else: 
                raise Exception(f'Condition {ConIn} not found; Condition has to be either ''Congruent'' or ''Incongruent''')  
            
            # general paths
            base_path   = Path('//analyse7/Project0407/')
            out_file    = base_path / 'Data' / 'Rest' / f'{ConIn}_{side}_{method}_Motor.h5'
            # Frequency Range
            fmin = 1
            fmax = 50
            
            # Process each subject
            for s, sub in enumerate(sub_lst[c]):
                subjects_data = {}  
                # subject specific paths
                source_path = base_path / f'/Data/{sub}/{ses}/meg/rest/source/'

                h5_file     = source_path / f'{sub}_{ses}_src_rest-all_{method}-d{d}-l{l}-snr{snr}.h5'
                labels_dir  = Path('C:/fs_data') / sub / 'label'
                
                # skip subjects that have already been processed
                mode = 'a' if out_file.exists() else 'w'

                with h5py.File(out_file, mode) as f:
                    # Add file attributes if creating new file
                    if mode == 'w':
                        f.attrs['description'] = 'Processed MEG FFT data for specific ROI'
                    
                    # Check if subject already exists
                    if sub in f:
                        print(f"Skipping {sub} - already exists in file")
                        continue
                
                frequencies, block_spectra, blocks = process_single_subject(
                    sub, ses, side, labels_dir, h5_file, fmin, fmax)
                    
                subject_data['spectra']         = block_spectra
                subject_data['freqs']           = frequencies 
                subject_data['blocks']          = blocks
                try: # if the central block is not in the data, set it to 0
                    subject_data['central_block']   = sub_data[c][s].index(12)
                except:
                    subject_data['central_block']   = 0 
                save_subject(out_file, sub, subject_data)
