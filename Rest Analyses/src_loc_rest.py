import os
import re
import mne 
import h5py
import glob
import numpy as np
from datetime import datetime
from pathlib import Path

# %%
def make_source_rest(sub, ses):
    # creat the source localised files for a given subject and session containing all the blocks.

    #input examples: 
    #sub = 'sub-02'
    #ses = 'ses-1'
    
    # source loc parameters
    src_d = 0.8 # depth
    src_l = 0.2 # loose
    src_f = False # Fixed? (True/False)
    
    snr         = 3.0  # Signal-to-noise ratio
    inv_method  = 'dSPM' #'sLORETA'
    inv_lambda  = 1.0 / snr ** 2 # Regularization parameter
    
    # general paths
    fs_path         = Path('C:/fs_data/')
    script_dir      = Path(__file__).resolve() # Location of current scripts
    base_path       = script_dir.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' 
    ER_path         = data_path / 'empty-room'
    
    sub_num = int(sub.split('-')[1])
    
    # assign specific empty room recording 
    ER_pattern = os.path.join(ER_path, "ER_*_raw_sss.fif")
    ER_names   = sorted(glob.glob(ER_pattern))
    ER_assign  = {'ER1' : '240927',
                  'ER2' : '241018'
                  } 
    
    date_files = {group: [f for f in ER_names if date in f][0] for group, date in ER_assign.items()}
    
    
    if sub_num <= 33:
        ncm_path = Path(date_files['ER1'])     
    elif 34 <= sub_num :
        ncm_path = Path(date_files['ER2'])
        
    
    # specific mri paths
    src_path        = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}-src.fif'
    bem_path        = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}_bem.h5'
    fwd_path        = data_path / f'{sub}' / 'anat' / f'{sub}-fwd.fif'
    epoch_path      = data_path / f'{sub}' / f'{ses}' / 'meg' / 'rest'
    
    trans_path      = fs_path   / f'{sub}' / 'bem' / f'{sub}-{ses}-trans.fif'  
   
    
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
    
    first_block = True
    
    for block in available_blocks:
        
        # load data for given block    
        data = mne.read_epochs(epoch_path / f'{sub}_{ses}_rest-{block}_clean_epo.fif') 
        
        # output path for given block
        out_path    = data_path / f'{sub}' / f'{ses}' / 'meg' / 'rest' / 'source' 
        out_path.mkdir(parents=True, exist_ok=True)
        out_file    = out_path /  f'{sub}_{ses}_src_rest-all_{inv_method}-d{src_d}-l{src_l}-snr{snr}.h5'
        
        # read empty room data 
        ncm_data = mne.io.read_raw(ncm_path)
        
        # select the same channels as in the loaded data 
        common_channels = list(set(ncm_data.info['ch_names']) & set(data.info['ch_names']))
        ncm_data.pick(common_channels)
        
        #load or create forward model, depending on whether it exists already. Only need to run it on first session
        # # compute forward model 
        fwd = mne.make_forward_solution(data.info, trans=trans_path, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=1)
                
        # make noise covariance matrix from processed empty room recording
        ncov = mne.compute_raw_covariance(ncm_data, method = 'ledoit_wolf', rank='info')
        
        # compute inverse operator
        inverse_operator = mne.minimum_norm.make_inverse_operator(data.info, fwd, ncov, loose=src_l, depth= src_d, fixed= src_f)
         
        # # Save the inverse operator for future use
        # mne.minimum_norm.write_inverse_operator(f'{subject}-inv.fif', inverse_operator)
             
        stc = mne.minimum_norm.apply_inverse_epochs(data, inverse_operator, lambda2 = inv_lambda, method= inv_method)
                    
        # # # Plot the source activity on the brain
        # # stc.plot(subject=subject, subjects_dir=subjects_dir, hemi='both', initial_time=0.1)
        # t = 11 # trial
        # brain = stc[t].plot(hemi='split', subjects_dir=fs_path, initial_time=0.1, time_viewer=True, views='lateral') # plot trials
        # # # Visualize the source estimates on an inflated brain surface
        # brain = stc[t].plot(subject=sub, subjects_dir=fs_path, hemi='both',
        #                    surface='inflated', initial_time=0.1, time_viewer=True)
        
        # # You can explore the source activity over time using the time_viewer and specify which hemisphere (hemi='both' for both hemispheres).