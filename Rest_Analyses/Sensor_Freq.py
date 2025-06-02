#%% 
import re
import mne
import numpy as np
import h5py
from pathlib import Path

from datetime import datetime
import json


def calc_psds(script_dir, 
                      sub_folders=None,
                      freqs=[1, 80],
                      session=['ses-1'],
                      sensor_types=['mag', 'grad'],
                      n_processes=None):
    """
    Calculate PSDs for the whole spectrum with parallel processing and save to HDF5.
    
    Parameters:
    -----------
    script_dir : Path
        Path to the script directory.
    sub_folders : list
        List of subject folders to process. If None, will detect automatically.
    sessions : list
        List of session folders to process.
    freqs : list
        Frequency range of interest (default is [1, 80] Hz).
    sensor_types : list
        List of sensor types to process ('mag' for magnetometers, 'grad' for gradiometers).
    n_processes : int
        Number of processes to use. If None, uses CPU count - 1.
    """
    base_path = script_dir.parent.parent.parent
    data_path = base_path / 'Data'
    
    ses = session[0]
    
    if n_processes is None:
        n_processes = -1 # Use all available cores minus one
    
    # If sub_folders is not provided, detect automatically
    if sub_folders is None:
        reg_pattern = re.compile(r'^sub-\d{2}$')
        sub_folders = [d.name for d in data_path.iterdir() if d.is_dir() and reg_pattern.match(d.name)]
    
    for sub in sub_folders:
        folder_path = data_path / sub
        ses_path = folder_path / ses / 'meg'
        
        if not ses_path.exists():
            print( f"Session path does not exist for {sub}/{ses}, skipping.")
            continue
        
        rest_dir = ses_path / 'rest'
        if not rest_dir.exists():
           print(f"Rest directory does not exist for {sub}/{ses}, skipping.")
           continue 
        
        # Create output directory for HDF5 files
        derivatives_dir = data_path / 'derivatives' / 'spectral_analysis'
        sub_derivatives_dir = derivatives_dir / sub / ses / 'meg'
        sub_derivatives_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all epoch files for this subject and session
        epo_files = list(rest_dir.glob(f'{sub}_{ses}_rest-*_clean_epo.fif'))
        epo_files.sort(key=lambda x: int(re.search(r'rest-(\d+)', x.name).group(1)))
        
        if not epo_files:
            print(f"No epoch files found for {sub}/{ses}, skipping.")
            continue
        results = []
        
        # Process each sensor type separately
        for sensor_type in sensor_types:
            # Create HDF5 file for this subject, session, and sensor type
            h5_filename = f'{sub}_{ses}_task-rest_{sensor_type}_psd.h5'
            h5_filepath = sub_derivatives_dir / h5_filename
            
            with h5py.File(h5_filepath, 'w') as h5f:
                # Store metadata
                h5f.attrs['subject'] = sub
                h5f.attrs['session'] = ses
                h5f.attrs['sensor_type'] = sensor_type
                h5f.attrs['freq_min'] = freqs[0]
                h5f.attrs['freq_max'] = freqs[1]
                h5f.attrs['creation_date'] = datetime.now().isoformat()
                h5f.attrs['mne_version'] = mne.__version__
                h5f.attrs['method'] = 'multitaper'
                
                all_psds = []
                block_info = []
                              
                # Process each epoch file
                for i, epo_file in enumerate(epo_files):
                    # Extract block number and condition
                    block_match = re.search(r'rest-(\d+)', epo_file.name)
                    block_num = int(block_match.group(1))
                                        # Load the epoch data
                    epochs = mne.read_epochs(epo_file, verbose=True)

                    # Calculate PSD using multitaper method
                    # Using n_jobs=1 here since we're already parallelizing at subject level
                    spectrum = epochs.compute_psd(
                        method='multitaper', 
                        fmin=freqs[0], 
                        fmax=freqs[1],
                        bandwidth=2,  # Bandwidth for multitaper, also corresponds pretty much to default in MNE with window size 4 and sFreq of 500
                        adaptive=True,
                        picks=sensor_type,  # Use the specified sensor type
                        n_jobs= n_processes, 
                        verbose=True
                    )
                    
                    # Get PSD data and frequencies
                    psd_data = spectrum.get_data()  # Shape: (n_epochs, n_channels, n_freqs)

                    # Store block information
                    block_info.append({
                        'block_number': block_num,
                        'n_epochs': psd_data.shape[0],
                        'filename': epo_file.name
                    })
                    
                    all_psds.append(psd_data)
                    
                all_freqs = spectrum.freqs
                ch_names = spectrum.ch_names
                
                if all_psds:
                    # Concatenate all PSDs
                    all_psds_array = np.concatenate(all_psds, axis=0)
                    
                    # Store data in HDF5
                    h5f.create_dataset('psd', data=all_psds_array, compression='gzip')
                    h5f.create_dataset('freqs', data=all_freqs, compression='gzip')
                    h5f.create_dataset('ch_names', data=[name.encode('utf-8') for name in ch_names])
                    
                    # Store block information
                    block_group = h5f.create_group('blocks')
                    for idx, block in enumerate(block_info):
                        block_subgroup = block_group.create_group(f'block_{idx:02d}')
                        for key, value in block.items():
                            if isinstance(value, str):
                                block_subgroup.attrs[key] = value.encode('utf-8')
                            else:
                                block_subgroup.attrs[key] = value
                    
                    # Calculate and store epoch indices for each block
                    epoch_indices = []
                    cumsum = 0
                    for block in block_info:
                        start_idx = cumsum
                        end_idx = cumsum + block['n_epochs']
                        epoch_indices.append([start_idx, end_idx])
                        cumsum = end_idx
                    
                    h5f.create_dataset('epoch_indices', data=epoch_indices)
                    
                    results.append(f"Successfully processed {sub}/{ses} {sensor_type}: "
                                f"{all_psds_array.shape[0]} epochs, {len(ch_names)} channels, "
                                f"{len(all_freqs)} frequencies")
                else:
                    results.append(f"No valid data found for {sub}/{ses} {sensor_type}")
        print(results)
        
def load_psd_data(h5_filepath):
    """
    Convenience function to load PSD data from HDF5 file.
    
    Parameters:
    -----------
    h5_filepath : Path or str
        Path to the HDF5 file.
        
    Returns:
    --------
    dict : Dictionary containing PSD data and metadata.
    """
    with h5py.File(h5_filepath, 'r') as h5f:
        data = {
            'psd': h5f['psd'][:],
            'freqs': h5f['freqs'][:],
            'ch_names': [name.decode('utf-8') for name in h5f['ch_names'][:]],
            'epoch_indices': h5f['epoch_indices'][:],
            'metadata': dict(h5f.attrs),
            'blocks': {}
        }
        
        # Load block information
        if 'blocks' in h5f:
            for block_key in h5f['blocks'].keys():
                block_data = {}
                for attr_key, attr_value in h5f['blocks'][block_key].attrs.items():
                    if isinstance(attr_value, bytes):
                        block_data[attr_key] = attr_value.decode('utf-8')
                    else:
                        block_data[attr_key] = attr_value
                data['blocks'][block_key] = block_data
    
    return data

# Example usage:
if __name__ == "__main__":
    script_path = Path(__file__).parent
    
    # Create BIDS-compliant directory structure and metadata
    base_path = script_path.parent.parent.parent
    data_path = base_path / 'Data'
    
    # Calculate PSDs with parallel processing
    calc_psds(
        script_path, 
        sub_folders=None,  # You can specify subjects or leave None for auto-detect
        session=['ses-1'],
        sensor_types=['mag', 'grad'],
        n_processes=3  # Adjust based on your system
    )
    
    # Example of how to load the data back
    # psd_file = data_path / 'derivatives' / 'spectral_analysis' / 'sub-67' / 'ses-1' / 'meg' / 'sub-67_ses-1_task-rest_mag_psd.h5'
    # if psd_file.exists():
    #     data = load_psd_data(psd_file)
    #     print(f"Loaded PSD data: {data['psd'].shape}")
    #     print(f"Frequency range: {data['freqs'][0]:.1f} - {data['freqs'][-1]:.1f} Hz")
    #     print(f"Channels: {len(data['ch_names'])}")
# %%
