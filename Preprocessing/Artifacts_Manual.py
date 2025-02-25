import mne
import numpy as np
import os
import re
from pathlib import Path

from mne.annotations import Annotations

def mark_artifacts_interactive(raw, annot_path = None):     
    # check if annotations already exist. If yes load those, otherwise create new ones   
    if Path(annot_path).exists():
        artf_annot = mne.read_annotations(annot_path)  
        
    else:  
        # Initialize new annotation category for artefacts
        # also get's rid of the other annotations to make life with the gui easier
        artf_annot = mne.Annotations(
            onset=[raw.first_time],
            duration=[0],
            description=['bad_artifact'],
            orig_time=raw.annotations.orig_time if raw.annotations is not None else raw.info['meas_date']
        )
    # set annotation of raw file to artefacts only
    raw.set_annotations(artf_annot)
  
    # Launch interactive plotter
    fig = raw.plot(
        duration=20.0,  # Show 20 seconds at a time
        n_channels=50,  # Show 50 channels at a time
        block=True,
        show_scrollbars=True
    )
    
    # save the annotations without altering the old annotation.
    if annot_path is not None:
        raw.annotations.save(annot_path)
        print(f"Annotations saved to: {annot_path}")
    else: # more for debug than anything, since annot_path should never be None
        annot_path = 'C:\Annot_Backup.fif'
        raw.annotations.save(annot_path)
        print(f"No Path provided, Annotations saved to: {annot_path}")

def Artifacts_Manual(all_flag = 0, rstate = 100, start_sub = None, single_sub= False):
    '''
    Function for marking all artefacts throughout the experiment in a given subject. 
    Resulting annotation file is saves as an fif in the respective subject folder. 
    Still needs to be applied and combined if the user wants to use these in future scripts
    
    parameters: 
        all_flag: indicates whether all subjects should be processed/re-processed (1) or not (0)
        rstate: indicates which file of a specific ICA seed should be used. Different processing runs may have different seeds so this is used to distinguish them (default = 100)
        start_sub: optional parameter to specify starting subject (format: "XX" where XX is the subject number, e.g., "05")
        single_sub: if True, only process the specified start_sub. If False, continue processing subsequent subjects (default: False)
    '''
    
    rstring = f'r{rstate}'
    script_dir = Path('Z:\Scripts\Preprocessing')
    base_path       = script_dir.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' # Folder containing the Data
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # creat regular expression pattern to identify folders of processed subjects
    
    sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and reg_pattern.match(d)] # find the folders starting with sub-XX where XX is the subject number with a leading 0
    sub_folders.sort() 
#%% 
    # If start_sub is specified, find its index in sub_folders
    start_index = 0
    if start_sub is not None:
        start_sub_folder = f'sub-{start_sub}'
        try:
            start_index = sub_folders.index(start_sub_folder)
        except ValueError:
            raise ValueError(f"Subject folder {start_sub_folder} not found in {data_path}")
   
    # Adjust sub_folders based on start_sub and single_sub parameters
    if single_sub:
        sub_folders = sub_folders[start_index:start_index + 1]
    else:
        sub_folders = sub_folders[start_index:]
    
    # %% Loop over the processed subjects and sessions
    for sub in sub_folders:
        folder_path = data_path / sub
               
        ses_folders = ['ses-1','ses-2'] # give options for two sessions
        
        for ses in ses_folders:
            ses_path = folder_path / ses / 'meg'
             
            # check if subject and session have been processed
            if ses_path.exists():  
                data_pattern    = f'{ses_path}/{sub}_{ses}_{rstring}_PostICA_Full.fif'    
                annot_path      = f'{ses_path}/{sub}_{ses}_Artf.fif'
                
                if Path(annot_path).exists() and all_flag != 1:
                    continue 
               
                # Load preprocessed data
                raw = mne.io.read_raw_fif( data_pattern, preload=True)     
                
                # mark artefacts manually
                # if subjects have already been marked, load that previous marking file first
                mark_artifacts_interactive(raw, annot_path)
          
        
