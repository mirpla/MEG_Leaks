# Import necessary Packages
import mne
import pandas as pd
import os
import re
import glob
from pathlib import Path
from ast import literal_eval
from meg_analysis.Scripts.Preprocessing.Preproc_Functions import extract_sort_key

'''
Contains all steps done on the whole data-set for a given subject after running the ICA. 
Contains a script for applying ICA to the individual data files, and subsequent marking the artifacts in the data.
'''
def apply_ICA(rstate, start_sub=None, single_sub=False):
    """
    Apply ICA components to individual MEG data files.
    
    Parameters:
        rstate: Random state used for ICA processing
        start_sub: Optional parameter to specify starting subject (format: "XX" where XX is the subject number, e.g., "05")
        single_sub: If True, only process the specified start_sub. If False, continue processing subsequent subjects (default: False)
    """
    # %% Set up Paths 
    script_dir      = Path(__file__).resolve() # Location of current scripts
    base_path       = script_dir.parent.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' # Folder containing the Data
    ica_comp_path   = data_path / 'ICA_Components.csv'
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # create regular expression pattern to identify folders of processed subjects
    sub_folders = [d.name for d in data_path.iterdir() if d.is_dir() and reg_pattern.match(d.name)] # find the folders starting with sub-XX where XX is the subject number with a leading 0
    sub_folders.sort()
    
    # %% Handle subject selection (similar to Artifacts_Manual)
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
    
    # %% other parameters
    param_filt_lp   = 100   # filter parameters lowpass     
    icf             = pd.read_csv(ica_comp_path, delimiter = ',') 
    
    # %% Loop over the processed subjects 
    for sub in sub_folders:
        folder_path = data_path / sub
        
        sub_nr = sub.split('-')[1]
        
        ses_folders = ['ses-1', 'ses-2'] # give options for two sessions
        
        for ses in ses_folders:
            ses_path = folder_path / ses / 'meg'
            
            if ses_path.exists():
                # find the number of the session to make sure the correct pattern is matched for the file names
                ses_nr = ses.split('-')[1]
                
                # Create output directory for individual processed files
                output_dir = ses_path / 'individual_post_ica'
                output_dir.mkdir(exist_ok=True)
                
                # Find individual downsampled files
                downsampled_path    = ses_path / 'downsampled'
                data_pattern        = f"sub-{sub_nr}_ses-{ses_nr}_task-*_run-*_meg_tsss_notch-ds-500Hz.fif" 
                data_files          = downsampled_path.glob(data_pattern)
                data_files_sorted   = sorted(data_files, key=extract_sort_key) # sort the files in the folder to have the runs in ascending order and load the SRT files first and only then the WL files        
                
                # Check if any files exist to process
                if not data_files_sorted:
                    print(f"No data files found for Subject: {sub}, Session: {ses}")
                    continue
                
                # select the correct row for this subject and session
                ica_row = icf[(icf['subject'] == sub) & (icf['session'] == int(ses_nr))]
                if ica_row.empty:
                    print(f"No matching rows found for Subject: {sub}, Session: {ses}")
                    continue
                else:
                    # save the components as a list
                    ica_components_bad = literal_eval(ica_row['components'].iloc[0]) 
                    
                ica_dir     = ses_path / 'ica'
                ica_file    = ica_dir / f"ica_projsub-{sub_nr}_ses-{ses_nr}_rstate{rstate}.fif"
                
                # Check if ICA file exists
                if not ica_file.exists():
                    print(f"ICA file not found: {ica_file}")
                    continue
                
                # Load ICA
                ica = mne.preprocessing.read_ica(ica_file, verbose=None)
                
                # ICA component removal doesn't work for the string names. Converting to numbers instead where ICA001 == 1 etc.
                ica_ints = [int(comp[3:]) for comp in ica_components_bad]
                ica.exclude = ica_ints
                
                print(f"Processing individual files for subject {sub} -- ses {ses}")
                print(f"Removing ICA components: {ica_components_bad}")
                
                # Process each file individually
                for data_file in data_files_sorted:
                    # Create output filename
                    original_name = data_file.stem  # filename without extension
                    output_filename = f"{original_name}_post_ica_r{rstate}_.fif"
                    output_path = output_dir / output_filename
                    
                    # Skip if already processed
                    if output_path.exists():
                        print(f"  Skipping {data_file.name} (already processed)")
                        continue
                    
                    print(f"  Processing {data_file.name}")
                    
                    # Load individual file
                    raw = mne.io.read_raw_fif(data_file, preload=True)
                    
                    # Apply ICA
                    ica.apply(raw)
                    
                    # Apply low pass filter post-ICA
                    print(f"    Applying {param_filt_lp} Hz low pass filter")
                    raw.filter(l_freq=None, h_freq=param_filt_lp, fir_design='firwin')
                    
                    # Save processed file
                    raw.save(output_path, overwrite=True)
                    print(f"    Saved: {output_filename}")
                    
                    # Clear memory
                    del raw
                
                print(f"Completed processing for subject {sub} -- {ses}")
                
    print("ICA application to individual files completed!")
                
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
        raw.annotations.save(annot_path, overwrite=True)
        print(f"Annotations saved to: {annot_path}")
    else: # more for debug than anything, since annot_path should never be None
        annot_path = 'C:\Annot_Backup.fif'
        raw.annotations.save(annot_path)
        print(f"No Path provided, Annotations saved to: {annot_path}")


def Artifacts_Manual(redo_flag=0, rstate=100, start_sub=None, single_sub=False):
    '''
    Function for marking all artefacts throughout the experiment in individual files for a given subject. 
    Resulting annotation files are saved as fif files in the respective subject folder for each individual file. 
    
    parameters: 
        redo_flag: indicates whether all subjects should be processed/re-processed (1) or not (0)
        rstate: indicates which file of a specific ICA seed should be used. Different processing runs may have different seeds so this is used to distinguish them (default = 100)
        start_sub: optional parameter to specify starting subject (format: "XX" where XX is the subject number, e.g., "05")
        single_sub: if True, only process the specified start_sub. If False, continue processing subsequent subjects (default: False)
    '''
    
    rstring = f'r{rstate}'
    script_dir = Path(__file__).resolve()  # Use __file__ for consistency with apply_ICA
    base_path = script_dir.parent.parent.parent.parent  # Root folder
    data_path = base_path / 'Data'  # Folder containing the Data
    
    reg_pattern = re.compile(r'^sub-\d{2}$')  # create regular expression pattern to identify folders of processed subjects
    
    # Use pathlib consistently for directory operations
    sub_folders = [d.name for d in data_path.iterdir() if d.is_dir() and reg_pattern.match(d.name)]
    sub_folders.sort() 

    # Handle subject selection (same logic as apply_ICA)
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
    
    # Loop over the processed subjects and sessions
    for sub in sub_folders:
        folder_path = data_path / sub
        sub_nr = sub.split('-')[1]
               
        ses_folders = ['ses-1', 'ses-2']  # give options for two sessions
        
        for ses in ses_folders:
            ses_path = folder_path / ses / 'meg'
             
            # check if subject and session have been processed
            if ses_path.exists():  
                ses_nr = ses.split('-')[1]
                
                # Look for individual post-ICA files
                individual_post_ica_dir = ses_path / 'individual_post_ica'
                            
                if not individual_post_ica_dir.exists():
                    print(f"No individual post-ICA directory found for Subject: {sub}, Session: {ses}")
                    continue
                
                # Find individual post-ICA files
                data_pattern = f"sub-{sub_nr}_ses-{ses_nr}_task-*_run-*_meg_tsss_notch-ds-500Hz_post_ica_{rstring}_.fif"
                data_files = individual_post_ica_dir.glob(data_pattern)
                data_files_sorted = sorted(data_files, key=extract_sort_key)  # sort the files consistently
                
                # Check if any files exist to process
                if not data_files_sorted:
                    print(f"No individual post-ICA files found for Subject: {sub}, Session: {ses}")
                    continue
                
                # Create output directory for artifact annotations
                annot_output_dir = ses_path / 'individual_annotations'
                annot_output_dir.mkdir(exist_ok=True)
                
                print(f"Processing artifact marking for subject {sub} -- ses {ses}")
                
                # Process each individual file
                for data_file in data_files_sorted:
                    # Create annotation filename based on the data file
                    original_name = data_file.stem  # filename without extension
                    annot_filename = f"{original_name}artf_annot.fif"
                    annot_path = annot_output_dir / annot_filename
                    
                    # Skip if already processed and redo_flag is not set
                    if annot_path.exists() and redo_flag != 1:
                        print(f"  Skipping {data_file.name} (already annotated)")
                        continue
                    
                    print(f"  Processing artifact marking for {data_file.name}")
                    
                    # Load individual post-ICA file
                    raw = mne.io.read_raw_fif(data_file, preload=True)     
                    
                    # Mark artifacts manually for this individual file
                    mark_artifacts_interactive(raw, annot_path)
                    
                    # Clear memory
                    del raw
                
                print(f"Completed artifact marking for subject {sub} -- ses {ses}")
                
    print("Individual file artifact marking completed!")