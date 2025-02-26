import os
import mne 
import glob
from pathlib import Path
import logging
from Source_Class import RestSourceData


def make_source_rest(sub, ses, src_d, src_l, src_f):
    """Create the source localised files for a given subject and session containing all the blocks."""
    
    snr         = 3.0  # Signal-to-noise ratio
    inv_method  = 'dSPM' #'sLORETA'
    inv_lambda  = 1.0 / snr ** 2 # Regularization parameter
    
    # general paths
    fs_path         = Path('C:/fs_data/')
    script_dir      = Path(__file__).resolve() # Location of current scripts
    base_path       = script_dir.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' 
    ER_path         = data_path / 'empty-room'
    data = pd.read_csv[data_path / 'Subject_Information.csv'] 
    
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
    elif 34 <= sub_num:
        ncm_path = Path(date_files['ER2'])
    
    # specific mri paths
    src_path        = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}-src.fif'
    bem_path        = data_path / f'{sub}' / 'anat' / 'bem' / f'{sub}_bem.h5'
    epoch_path      = data_path / f'{sub}' / f'{ses}' / 'meg' / 'rest'
    
    trans_path      = fs_path   / f'{sub}' / 'bem'  / f'{sub}-{ses}-trans.fif'  
   
    
    # set up output paths
    out_path = data_path / f'{sub}' / f'{ses}' / 'meg' / 'rest' / 'source' 
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f'{sub}_{ses}_src_rest-all_{inv_method}-d{int(src_d*10)}-l{int(src_l*10)}-snr{int(snr)}.h5'
   
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
     
    # Check if output file exists and whether to overwrite
    if out_file.exists():
        print(f"Output file {out_file} exists. Processing will add any missing blocks.")
    
    for i, block in enumerate(available_blocks):
        print(f"Processing block {block}...")
        
        # load data for given block    
        data = mne.read_epochs(epoch_path / f'{sub}_{ses}_rest-{block}_clean_epo.fif') 
        
        # read empty room data 
        ncm_data = mne.io.read_raw(ncm_path)
        
        # select the same channels as in the loaded data 
        common_channels = list(set(ncm_data.info['ch_names']) & set(data.info['ch_names']))
        ncm_data.pick(common_channels)
        
        # compute forward model 
        fwd = mne.make_forward_solution(data.info, trans=trans_path, src=src, bem=bem, 
                                      meg=True, eeg=False, mindist=5.0, n_jobs=1)
                
        # make noise covariance matrix from processed empty room recording
        ncov = mne.compute_raw_covariance(ncm_data, method='ledoit_wolf', rank='info')
        
        # compute inverse operator
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            data.info, fwd, ncov, loose=src_l, depth=src_d, fixed=src_f)
             
        # compute source estimates
        stc = mne.minimum_norm.apply_inverse_epochs(
            data, inverse_operator, lambda2=inv_lambda, method=inv_method)
        
        #try:
        # Initialize file with first block or add to existing
        if i == 0:
            meg_handler.initialize_file(out_file, data, stc[0], inverse_operator)
        
        # Add block data
        meg_handler.add_block(out_file, block, stc)
        print(f"Successfully processed and saved block {block}")
        
        # except Exception as e:
        #     print(f"Error processing block {block}: {str(e)}")
        #     continue
    
    print("Source localization complete!")

if __name__ == "__main__":
    # subject and session
    subs = ['sub-31','sub-32','sub-33','sub-35','sub-36']
    for sub  in subs:
        ses     = 'ses-1'
    
        # source loc parameters
        src_d   = 0.8 # depth
        src_l   = 0.2 # loose
        src_f   = False # Fixed? (True/False)

        # Example usage
        make_source_rest(sub, ses, src_d, src_l, src_f)