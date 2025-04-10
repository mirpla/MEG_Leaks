#%%
import mne 
import pandas as pd
from pathlib import Path
import os
import concurrent.futures
import numpy as np
import time

# Enable GPU acceleration for MNE
os.environ['MNE_USE_CUDA'] = 'true'

from meg_analysis.Scripts.Rest_Analyses.Source_Class import RestSourceData

#%%
def make_source_rest(sub, ses, src_d, src_l, src_f, inv_method = 'dSPM', n_jobs=None):
    """Create the source localised files for a given subject and session containing all the blocks.
        sub : list of str
            Subject ID (e.g., 'sub-01')
        ses : str
            Session ID (e.g., 'ses-1')
        src_d : float
            Depth weighting (0.0 to 1.0)
        src_l : float
            Loose orientation (0.0 to 1.0)
        src_f : bool
            Fixed orientation (True/False)
        inv_method : str
            Inverse method to use for the source localisation (e.g., 'dSPM', 'sLORETA')
        n_jobs : int or None
            Number of jobs to run in parallel. If None, it will use the value from MNE config.
    """
    
    if n_jobs is None:
        # Auto-detect cores, but leave some headroom
        n_jobs = max(1, os.cpu_count() - 2)
        
    print(f"Using {n_jobs} CPU cores for processing")
    if os.environ.get('MNE_USE_CUDA') == 'true':
        print("CUDA GPU acceleration is enabled")
    
    snr         = 3.0  # Signal-to-noise ratio
    inv_lambda  = 1.0 / snr ** 2 # Regularization parameter
    
    # general paths
    fs_path         = Path('C:/fs_data/')
    script_dir      = Path(__file__).resolve() # Location of current scripts
    base_path       = script_dir.parent.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' 
    ER_path         = data_path / 'empty-room'
    data            = pd.read_csv(data_path / 'Subject_Information.csv',encoding='latin1') 
    
    # get index for subject and find the ER date
    sub_idx         = data[data['sub'] == sub].index[0] # get the index of the subject in the dataframe
    ER_date         = data['ER'][sub_idx] # get the date of the subject from the dataframe  
    
    # assign specific empty room recording     
    ncm_path = ER_path / f"ER_{ER_date}_raw_sss.fif"

    # specific mri paths
    src_path        = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}-src.fif'
    bem_path        = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}_bem.h5'
    epoch_path      = data_path / f'{sub}' / f'{ses}' / 'meg' / 'rest'
    
    trans_path      = fs_path   / f'{sub}' / 'bem'  / f'{sub}-{ses}-trans.fif'  
   
    
    # set up output paths
    out_path = data_path / f'{sub}' / f'{ses}' / 'meg' / 'rest' / 'source' 
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f'{sub}_{ses}_src_rest-all_{inv_method}-d{int(src_d*10)}-l{int(src_l*10)}-snr{int(snr)}.h5'
   
    # Check if output file exists and whether to overwrite
    if out_file.exists():
        print(f"Output file {out_file} exists. Skipping Subject...")
        return
        
    # load relevant files 
    src  = mne.read_source_spaces(src_path) # load source space file
    bem  = mne.read_bem_solution(bem_path) # load boundary elements model
    
    # Detect existing blocks and loop over them 
    block_files = list(epoch_path.glob("*_clean_epo.fif"))
    available_blocks = []
    for block in range(12):
        file_pattern = f"{sub}_{ses}_rest-{block}_clean_epo.fif"
        file_path = epoch_path / file_pattern
        if file_path in block_files:
            available_blocks.append(block)
    
    # Initialize MEG source data handler
    meg_handler = RestSourceData(sub, ses)
    
    # Load and prepare empty room data once for all blocks
    print("Loading and preparing empty room data...")
    ncm_data = mne.io.read_raw(ncm_path)
    
    # Function to process a single block
    def process_block(block_idx):
        block = available_blocks[block_idx]
        try:
            print(f"Processing block {block}...")
            
            # load data for given block    
            data = mne.read_epochs(epoch_path / f'{sub}_{ses}_rest-{block}_clean_epo.fif') 
            
            # select the same channels as in the loaded data 
            common_channels = list(set(ncm_data.info['ch_names']) & set(data.info['ch_names']))
            ncm_data_block = ncm_data.copy().pick(common_channels)
            
            # compute forward model - this can benefit from parallelization
            fwd = mne.make_forward_solution(data.info, trans=trans_path, src=src, bem=bem, 
                                          meg=True, eeg=False, mindist=5.0, n_jobs=n_jobs)
                    
            # make noise covariance matrix from processed empty room recording
            # CuPy (GPU) will be used automatically if enabled
            ncov = mne.compute_raw_covariance(ncm_data_block, method='ledoit_wolf', rank='info')
            
            # compute inverse operator
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                data.info, fwd, ncov, loose=src_l, depth=src_d, fixed=src_f)
                 
            # compute source estimates - this benefits from GPU acceleration
            stc = mne.minimum_norm.apply_inverse_epochs(
                data, inverse_operator, lambda2=inv_lambda, method=inv_method)
            
            return block, data, stc, inverse_operator
            
        except Exception as e:
            print(f"Error processing block {block}: {str(e)}")
            return None
    
    # Process first block sequentially to initialize the file
    print("Processing first block to initialize file...")
    first_result = process_block(0)
    if first_result is not None:
        first_block, first_data, first_stc, first_inv_op = first_result
        meg_handler.initialize_file(out_file, first_data, first_stc[0], first_inv_op)
        meg_handler.add_block(out_file, first_block, first_stc)
        print(f"Successfully processed and saved block {first_block}")
    
    # Process remaining blocks in parallel
    if len(available_blocks) > 1:
        print(f"Processing remaining {len(available_blocks)-1} blocks in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, len(available_blocks)-1)) as executor:
            future_to_block = {executor.submit(process_block, i): i for i in range(1, len(available_blocks))}
            for future in concurrent.futures.as_completed(future_to_block):
                result = future.result()
                if result is not None:
                    block, _, stc, _ = result
                    meg_handler.add_block(out_file, block, stc)
                    print(f"Successfully processed and saved block {block}")
    
    print("Source localization complete!")

#%% Parallel processing of multiple subjects
def make_source_parallel(subjects, session, src_d, src_l, src_f, inv_method, max_subjects=3):
    """
    Process multiple subjects in parallel
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    session : str
        Session ID (e.g., 'ses-1')
    src_d, src_l, src_f : source localization parameters
        Depth, loose, and fixed parameters for source localization
    inv_method : str
        Inverse method to use for the source localization (e.g., 'dSPM', 'sLORETA')
    max_subjects : int
        Maximum number of subjects to process in parallel
    """
    print(f"Processing {len(subjects)} subjects with up to {max_subjects} in parallel")
    
    # Instead of using ProcessPoolExecutor, we'll use a ThreadPoolExecutor
    # This avoids the pickling issues and might be more stable for your use case
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_subjects) as executor:
        # Submit jobs for each subject
        future_to_subject = {}
        for sub in subjects:
            # Submit each subject as a separate job
            future = executor.submit(make_source_rest, sub, session, src_d, src_l, src_f, inv_method)
            future_to_subject[future] = sub
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_subject):
            sub = future_to_subject[future]
            try:
                # Get the result (or exception if it failed)
                future.result()
                print(f"Subject {sub} completed successfully")
            except Exception as e:
                print(f"Error processing subject {sub}: {str(e)}")