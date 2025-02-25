import mne 
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path

class RestSourceData:
    """Class to handle MEG source localization data storage and retrieval using HDF5."""
    
    def __init__(self, subject_id, session_id):
        """
        Initialize MEG source data handler.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., 'sub-02')
        session_id : str
            Session identifier (e.g., 'ses-1')
        """
        self.subject_id = subject_id
        self.session_id = session_id

# %%    read metadata        
    def _extract_data_info(self, epochs, stc, inv_operator):
        """Extract metadata from the MEG data objects."""
        info = {}
        
        # Timing information
        info['sfreq'] = float(epochs.info['sfreq'])
        info['tmin'] = float(epochs.tmin)
        info['tmax'] = float(epochs.tmax)
        
        # Epochs information
        info['analysis_type'] = 'resting_state'
        info['epoch_duration'] = float(epochs.tmax - epochs.tmin)
        info['n_epochs'] = int(len(epochs))
        info['n_channels'] = int(len(epochs.ch_names))
        
        # Source space information
        info['src_type'] = str(inv_operator['src'][0]['type'])
        info['n_sources'] = int(len(stc.vertices[0]) + len(stc.vertices[1]))
          
        return info
    
# %% File initialisation 
    def initialize_file(self, fname, epochs, stc, inv_operator):
        """Initialize HDF5 file with source space information and metadata."""
        with h5py.File(fname, 'w') as f:  # 'w' mode already overwrites the file
              # Store global metadata
              f.attrs['subject_id'] = str(self.subject_id)
              f.attrs['session_id'] = str(self.session_id)
              f.attrs['creation_date'] = str(datetime.now().isoformat())
              f.attrs['mne_version'] = str(mne.__version__)
              
              # Store data information
              info = self._extract_data_info(epochs, stc, inv_operator)
              
              # Create groups - will throw error if files or groups already exist
              info_group = f.create_group('info')
              src_group = f.create_group('source_space')
              f.create_group('blocks')
              
              # Store info
              for key, value in info.items():
                  info_group.attrs[key] = value
              
              # Store source space information
              src_group.create_dataset('vertices_lh', data=stc.vertices[0])
              src_group.create_dataset('vertices_rh', data=stc.vertices[1])
              src_group.attrs['tmin'] = float(stc.tmin)
              src_group.attrs['tstep'] = float(stc.tstep)
            

# %% Add information to block containing source estimates 
    def add_block(self, fname, block_id, stc_list):
        """Add a block of source estimates to an existing file."""
        with h5py.File(fname, 'a') as f:
            block_name = f'block_{block_id:02d}'
            
            # Check if block already exists
            if block_name in f['blocks']:
                print(f"Block {block_name} already exists. Skipping...")
                return
            
            # Create block group
            block_group = f['blocks'].create_group(block_name)
            
            # Get dimensions
            n_epochs = len(stc_list)
            n_sources = stc_list[0].data.shape[0]
            n_times = stc_list[0].data.shape[1]
            
            # Store block metadata
            block_group.attrs['n_epochs'] = n_epochs
            block_group.attrs['n_sources'] = n_sources
            block_group.attrs['n_times'] = n_times
            
            # Create dataset for this block's data
            data = block_group.create_dataset(
                'data',
                shape=(n_epochs, n_sources, n_times),
                dtype=np.float32,
                compression='gzip',
                compression_opts=9,
                chunks=True
            )
            
            # Fill the dataset
            for i, stc in enumerate(stc_list):
                data[i, :, :] = stc.data
                
                
# %% method for loading data from a specific block
    def load_block_data(self, fname, block_id=0):
        """
        Load data and metadata from a specific block.
        
        Parameters
        ----------
        fname : str or Path
            Path to the HDF5 file
  
        block_id : int
            Block number to load (default=0)
              
        Returns
        ------
        block_data : numpy.ndarray
            The source data for the block (epochs × sources × times)
        sfreq : float
            Sampling frequency
        vertices_lh : numpy.ndarray
              Left hemisphere vertices
        vertices_rh : numpy.ndarray
              Right hemisphere vertices
              """
    
        with h5py.File(fname, 'r') as f:
            # Check if block exists
            block_name = f'block_{block_id:02d}'
            if block_name not in f['blocks']:
                raise ValueError(f"Block {block_id} not found in file")
            
            # Load data
            block_data = f[f'blocks/{block_name}/data'][:]
            
            # Load relevant metadata
            sfreq = f['info'].attrs['sfreq']
            
            # Load source space information
            vertices_lh = f['source_space/vertices_lh'][:]
            vertices_rh = f['source_space/vertices_rh'][:]
            
        return block_data, sfreq, vertices_lh, vertices_rh
#%% Method for finding avaible blocks in h5 source file
    def get_available_blocks(self, fname):
        """
        Get list of available blocks in the source file.
        
        Parameters
        ----------
        fname : str or Path
            Path to the HDF5 file
            
        Returns
        -------
        list
            List of available block numbers
        """
        with h5py.File(fname, 'r') as f:
            blocks = list(f['blocks'].keys())
            # Convert block_00, block_01, etc. to integers
            block_nums = [int(block.split('_')[1]) for block in blocks]
            return sorted(block_nums)