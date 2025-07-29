import mne 
import pandas as pd
from pathlib import Path
import os
import concurrent.futures
import numpy as np
import time
import gc  # Add garbage collection

# Enable GPU acceleration for MNE
os.environ['MNE_USE_CUDA'] = 'false'

from meg_analysis.Scripts.Rest_Analyses.Source_Class import RestSourceData

def make_source_rest(sub, ses, src_d, src_l, src_f, inv_method='dSPM', n_jobs=None, max_block_workers=3, use_baseline_cov=False):
    """Create the source localised files for a given subject and session containing all the blocks.
    
    Parameters:
    -----------
    sub : str
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
        Number of jobs for MNE operations. If None, uses conservative default.
    max_block_workers : int
        Maximum number of blocks to process in parallel (default: 2)
    use_baseline_cov : bool
        If True, use baseline block for noise covariance instead of empty room recording
    """
    

    if n_jobs is None:
        n_jobs = min(12, max(1, os.cpu_count() * 3 // 4))  # Use 75% of cores
            
    print(f"Using {n_jobs} CPU cores for MNE operations")
    print(f"Using {max_block_workers} workers for block processing")
    print(f"Using {'baseline' if use_baseline_cov else 'empty room'} for noise covariance")
    
    snr = 3.0  
    inv_lambda = 1.0 / snr ** 2 
    
    # Setup paths (same as before)
    script_dir = Path(__file__).resolve()
    base_path = script_dir.parent.parent.parent.parent
    data_path = base_path / 'Data' 
    ER_path = data_path / 'empty-room'
    data = pd.read_csv(data_path / 'Subject_Information.csv', encoding='latin1') 
    
    sub_idx = data[data['sub'] == sub].index[0]
    ER_date = data['ER'][sub_idx]
    
    ncm_path = ER_path / f"ER_{ER_date}_raw_sss.fif"
    src_path = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}-src.fif'
    bem_path = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}_bem.h5'
    trans_path = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}-{ses}-trans.fif'  
    epoch_path = data_path / f'{sub}' / f'{ses}' / 'meg' / 'rest'
    
    # Path for baseline continuous data
    baseline_raw_path = epoch_path / f'{sub}_{ses}_task-SRT_rest-bl-srt_meg.fif'
    
    out_path = data_path / f'{sub}' / f'{ses}' / 'meg' / 'rest' / 'source' 
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create distinct filenames based on covariance method
    if use_baseline_cov:
        cov_suffix = "baseline"
        out_file = out_path / f'{sub}_{ses}_src_rest-all_{inv_method}-d{int(src_d*10)}-l{int(src_l*10)}-snr{int(snr)}-{cov_suffix}.h5'
        inv_op_file = out_path / f'{sub}_{ses}_inv-d{int(src_d*10)}-l{int(src_l*10)}-snr{int(snr)}-{cov_suffix}-inv.fif'
    else:
        out_file = out_path / f'{sub}_{ses}_src_rest-all_{inv_method}-d{int(src_d*10)}-l{int(src_l*10)}-snr{int(snr)}.h5'
        inv_op_file = out_path / f'{sub}_{ses}_inv-d{int(src_d*10)}-l{int(src_l*10)}-snr{int(snr)}-inv.fif'
            
    if out_file.exists():
        print(f"Output file {out_file} exists. Skipping Subject...")
        return
        
    # Load files
    src = mne.read_source_spaces(src_path)
    bem = mne.read_bem_solution(bem_path)
    
    # Find available blocks
    block_files = list(epoch_path.glob("*_clean_epo.fif"))
    available_blocks = []
    for block in range(12):
        file_pattern = f"{sub}_{ses}_rest-{block}_clean_epo.fif"
        file_path = epoch_path / file_pattern
        if file_path in block_files:
            available_blocks.append(block)
    
    print(f"Found {len(available_blocks)} blocks to process")
    
    # Initialize MEG source data handler
    meg_handler = RestSourceData(sub, ses)
    
    # Load and prepare noise covariance data - but release it after inverse operator
    print("Computing inverse operator...")
    inverse_operator = None
    if inv_op_file.exists():
        try:
            print(f"Loading existing inverse operator from {inv_op_file}")
            inverse_operator = mne.minimum_norm.read_inverse_operator(inv_op_file)
            print("Successfully loaded existing inverse operator")
        except Exception as e:
            print(f"Error loading existing inverse operator: {str(e)}")
            inverse_operator = None
    
    # Load first epoch file for info
    first_block = available_blocks[0]
    first_data = mne.read_epochs(epoch_path / f'{sub}_{ses}_rest-{first_block}_clean_epo.fif')
    
    if inverse_operator is None:
        # Compute noise covariance based on method choice
        if use_baseline_cov:
            print("Using baseline block for noise covariance...")
            
            # Check if baseline continuous file exists
            if baseline_raw_path.exists():
                print(f"Loading baseline continuous data from {baseline_raw_path}")
                baseline_raw = mne.io.read_raw(baseline_raw_path)
                
                # Match channels between baseline and experimental data
                common_channels = list(set(baseline_raw.info['ch_names']) & set(first_data.info['ch_names']))
                baseline_filtered = baseline_raw.copy().pick(common_channels)
                
                # Compute noise covariance from baseline continuous data
                ncov = mne.compute_raw_covariance(baseline_filtered, method='ledoit_wolf', rank='info')
                
                # Clean up baseline data
                del baseline_raw, baseline_filtered
                gc.collect()
                
            else:
                # Fallback: try to find baseline epoched data
                baseline_epoch_pattern = f"{sub}_{ses}_rest-bl*_clean_epo.fif"
                baseline_epoch_files = list(epoch_path.glob(baseline_epoch_pattern))
                
                if baseline_epoch_files:
                    print(f"Loading baseline epoched data from {baseline_epoch_files[0]}")
                    baseline_epochs = mne.read_epochs(baseline_epoch_files[0])
                    
                    # Match channels
                    common_channels = list(set(baseline_epochs.info['ch_names']) & set(first_data.info['ch_names']))
                    baseline_epochs = baseline_epochs.copy().pick(common_channels)
                    
                    # Compute noise covariance from epoched baseline data
                    ncov = mne.compute_covariance(baseline_epochs, method='ledoit_wolf', rank='info')
                    
                    # Clean up baseline data
                    del baseline_epochs
                    gc.collect()
                    
                else:
                    raise FileNotFoundError(f"No baseline data found for {sub}. "
                                          f"Expected either {baseline_raw_path} or matching *rest-bl*_clean_epo.fif files")
        else:
            print("Using empty room recording for noise covariance...")
            # Original empty room processing
            ncm_data = mne.io.read_raw(ncm_path)
            common_channels = list(set(ncm_data.info['ch_names']) & set(first_data.info['ch_names']))
            ncm_data_filtered = ncm_data.copy().pick(common_channels)
            
            # Compute noise covariance
            ncov = mne.compute_raw_covariance(ncm_data_filtered, method='ledoit_wolf', rank='info')
            
            # Clean up empty room data
            del ncm_data, ncm_data_filtered
            gc.collect()
        
        # Compute forward model
        fwd = mne.make_forward_solution(first_data.info, trans=trans_path, src=src, bem=bem, 
                                      meg=True, eeg=False, mindist=5.0, n_jobs=n_jobs)
        
        # Compute inverse operator
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            info=first_data.info, forward=fwd, noise_cov=ncov, 
            loose=src_l, depth=src_d, fixed=src_f)
        
        # Clean up more objects
        del fwd, ncov
        gc.collect()
        
        # Save inverse operator
        try:
            print(f"Saving inverse operator to {inv_op_file}")
            mne.minimum_norm.write_inverse_operator(inv_op_file, inverse_operator)
            print("Successfully saved inverse operator")
        except Exception as e:
            print(f"Warning: Could not save inverse operator: {str(e)}")
    
    # Process blocks with memory management
    def process_single_block(block):
        """Process a single block with explicit memory cleanup"""
        try:
            print(f"Processing block {block}...")
            
            # Load data for this block only
            epoch_file = epoch_path / f'{sub}_{ses}_rest-{block}_clean_epo.fif'
            data = mne.read_epochs(epoch_file)
            
            # Compute source estimates
            stc = mne.minimum_norm.apply_inverse_epochs(
                data, inverse_operator, lambda2=inv_lambda, method=inv_method)
            
            # Save immediately and clean up
            result = (block, stc)
            
            # Explicit cleanup
            del data, stc
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"Error processing block {block}: {str(e)}")
            gc.collect()  # Clean up even on error
            return None
    
    # Process first block to initialize file
    print("Processing first block to initialize file...")
    first_result = process_single_block(available_blocks[0])
    if first_result is not None:
        first_block, first_stc = first_result
        
        # Need to reload first data just for initialization
        first_data = mne.read_epochs(epoch_path / f'{sub}_{ses}_rest-{first_block}_clean_epo.fif')
        meg_handler.initialize_file(out_file, first_data, first_stc[0], inverse_operator)
        meg_handler.add_block(out_file, first_block, first_stc)
        
        # Clean up
        del first_data, first_stc
        gc.collect()
        
        print(f"Successfully processed and saved block {first_block}")
    
    # Process remaining blocks with parallel processing
    if len(available_blocks) > 1:
        remaining_blocks = available_blocks[1:]
        print(f"Processing remaining {len(remaining_blocks)} blocks...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_block_workers) as executor:
            future_to_block = {executor.submit(process_single_block, block): block 
                            for block in remaining_blocks}
            
            for future in concurrent.futures.as_completed(future_to_block):
                result = future.result()
                if result is not None:
                    block, stc = result
                    meg_handler.add_block(out_file, block, stc)
                    print(f"Successfully processed and saved block {block}")
            
            gc.collect()
    
    print("Source localization complete!")
    
    # Final cleanup
    del inverse_operator, meg_handler
    gc.collect()

