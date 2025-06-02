import os
import re
import shutil
import pandas as pd
import logging

from pathlib import Path

from meg_analysis.Scripts.Import.Import_Preproc import MEG_import_process, MEG_extract_destination, MakeBoundaryElementModel, check_path, save_file_to_path

# %% Begin 
def Import_Data(sub_codes):
    # -*- coding: utf-8 -*-
    # %% Setting up the Environment 

    # %% Setting up the Paths and relevant legend files
    script_dir = Path(__file__).resolve() # Location of current scripts
    base_path  = script_dir.parent.parent.parent.parent # Basepath that is convenient to move from throughout the tree
    
    raw_path  = Path(r'//raw/Project0407')  # Adjust this path as necessary
    anat_path = base_path / 'Anats' 
    
    sub_key  = pd.read_csv(base_path / 'Data/Subject_Keys.csv', dtype={0: str, 1: str, 2: str, 3: float, 4: str})
    sub_info = pd.read_csv(base_path / 'Data' / 'Subject_Information.csv', encoding='latin1')
    
    # check code for sub 12
    ses_name    = 'subject number'
    ses_n_lbl   = 'session number (1 = seq, 2 = control)' 
    hash_id     = 'subject hash'
    ses_dat     = 'session date'
    
    # %%
    log_file = base_path / "Data" / 'meg-proc_errors.log'
    logging.basicConfig(filename=log_file, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # %% find the raw MEG files and copy the behavioral files
    for list_idx, sub_code in enumerate(sub_codes):
        rowidx = sub_key[sub_key[ses_name] == sub_code]
        ses_n =  rowidx.shape[0]
        sessions_dum = rowidx.iloc[0]
           
        hash_up = sessions_dum[hash_id].upper() + "_RAW_32_ADNI" # MRI filename
    
        ses_name_str = str(int( sessions_dum[ses_name])).zfill(2) # ses folder with leading zeros
        
        mri_path = anat_path / hash_up 
        sub_path = base_path / "Data" / f"sub-{ses_name_str}"
        
        # Do MRI stuff first since it's session independent 
        MakeBoundaryElementModel(mri_path, sub_path )       
        
    # %%
        for x, sesidx in zip(range(ses_n), range(1,  ses_n + 1)):
            row_ses = rowidx.iloc[x] 
            out_path = sub_path / f"ses-{sesidx}"
                   
            # Building paths
            data_meg_path = raw_path / row_ses[hash_id] / row_ses[ses_dat]
            data_beh_path = raw_path / row_ses[hash_id] / 'SRT'
            out_path      = base_path / "Data" / f"sub-{ses_name_str}" / f"ses-{sesidx}"
            
            check_path(out_path / "beh") # Create the folders for the behavioral data
          
            for txt_file in data_beh_path.glob("*.txt"):# Copy only the text files for the corresponding session
                # split the path name into it's constiuent parts to check which session they belong to
                filename = txt_file.stem 
                parts = filename.split('_')
               
                if len(parts) > 1 and parts[1][1] == str(int(sessions_dum[ses_n_lbl])):          
                    destination_file = out_path / 'beh' / txt_file.name
                    if not os.path.exists(destination_file):
                        shutil.copy(txt_file, destination_file)
                        print(f'Copied {txt_file} to {destination_file}')
                    else:
                        print(f'{destination_file} already exists')
            
            # %%
            for meg_file in data_meg_path.glob("*.fif"):
                #make sure it doesn't look at extensions
                if meg_file.suffix == '.fif' and not meg_file.stem.endswith('-1'):
                    # find the last number to make the right filename
                    meg_stem = meg_file.stem
                    numbers = re.findall(r'\d+', meg_stem)
                    last_number = numbers[-1]
                
                    if "WL" in meg_stem:
                        task = "WL"
                    elif "SRT" in meg_stem:
                        task = "SRT"
                        if int(last_number) == 1:
                            dest_mat = {} 
                            dest_mat[sub_code] = MEG_extract_destination(meg_file)
                    
                    out_path_meg = out_path / "meg"
                    out_file_meg = f"sub-{ses_name_str}"+ f"_ses-{sesidx}_task-"+ task + f"_run-{last_number}_meg_tsss.fif"
                
                    full_out = out_path_meg / out_file_meg
                
                    check_path(out_path_meg) # Create the folders for the behavioral data               
                    try:
                        MEG_import_process(meg_file,full_out,dest_mat[sub_code])
                        print(f"Successfully saved sub-{ses_name_str}"+ f"_ses-{sesidx}_task-"+ task + f"_run-{last_number}_meg")
                    except Exception as e:
                        #Log the error with the file path
                        logging.error(f"Failed to process {meg_file}: {e}")
                          
                        print('#################### MEG ERROR FOR ' + str(meg_file) + ' ####################')
                        print(f"Error: Logged to {log_file}.")
