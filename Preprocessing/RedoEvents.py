#%% import the necessary modules
import os
import re
import gc
import mne
import glob
import logging
import numpy as np 
from datetime import datetime, timedelta
from pathlib import Path
from mne.preprocessing import ICA, read_ica
import meg_analysis.Scripts.Preprocessing.Preproc_Functions
from meg_analysis.Scripts.Preprocessing.Preproc_Functions import block_annotate, extract_sort_key, read_events

def Events_Fix():
# %% Set up Paths 
    script_dir = Path(__file__).resolve() # Location of current scripts
    base_path  = script_dir.parent.parent.parent # Root folder
    data_path  = base_path / 'Data' # Folder containing the Data
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # creat regular expression pattern to identify folders of processed subjects
    sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and reg_pattern.match(d)] # find the folders starting with sub-XX where XX is the subject number with a leading 0
    
    
    # %% Loop over the processed subjects 
    for sub in sub_folders:
        folder_path = data_path / sub
        
        sub_nr = sub.split('-')[1]
        
        ses_folders = ['ses-1', 'ses-2'] # give options for two sessions
        
        for ses in ses_folders:
            ses_path = folder_path / ses / 'meg'
            
            if os.path.isdir(ses_path):
                # find the number of the session to make sure the correct pattern is matched for the file names
                ses_nr = ses.split('-')[1]
                
                # select all the relevant fif files and make sure they are processes and later concatinated in the correct order
                fif_pattern = f"sub-{sub_nr}_ses-{ses_nr}_task-*_run-*_meg_tsss.fif"
                fif_files = ses_path.glob(fif_pattern)
                fif_files_sorted = sorted(fif_files, key=extract_sort_key) # sort the files in the folder to have the runs in ascending order and load the SRT files first and only then the WL files
                
              #try:
                print(f"Processing file: subject {sub} -- ses {ses}")
                for fif_file in fif_files_sorted:          
                    data = mne.io.read_raw_fif(fif_file,preload=True)
                          
                    anot = block_annotate(data, fif_file)
                    data.set_annotations(anot)
                
                    read_events(data,fif_file)      # find and annotate relevant events in the signal 
                    
                       