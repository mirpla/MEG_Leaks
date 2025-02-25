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
from meg_analysis.Scripts.Preprocessing.Preproc_Functions import block_annotate, extract_sort_key, read_events, manual_artifact_check, detect_artifacts

def Preprocess_Data_Artf():
    # %% Set up Paths 
    script_dir = Path(__file__).resolve() # Location of current scripts
    base_path  = script_dir.parent.parent.parent # Root folder
    data_path  = base_path / 'Data' # Folder containing the Data
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # creat regular expression pattern to identify folders of processed subjects
    sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and reg_pattern.match(d)] # find the folders starting with sub-XX where XX is the subject number with a leading 0
    
    
    # %% Set up logging of errors
    log_file = data_path / 'pre-proc_errors.log'
    
    logging.basicConfig(filename=log_file, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    # %% other parameters
    param_res_Fs    = 500   # frequency that data is downsampled to
    param_filt_hp   = 1     # filter parameters highpass
    #param_filt_lp   = 100   # filter parameters lowpass | better after ICA
    
    rstate = 99 # Seed for the ICA (see below)
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
                
                # make directory for all the downampled and filtered annotated files before the ICA
                downsampled_files = []
                downsampled_dir = ses_path / 'downsampled'
                downsampled_dir.mkdir(exist_ok=True)
                
                ica_dir = ses_path / 'ica'
                ica_dir.mkdir(exist_ok = True)
                ica_file = ica_dir / f"ica_projsub-{sub_nr}_ses-{ses_nr}_artf_{rstate}.fif"
                
                # select all the relevant fif files and make sure they are processes and later concatinated in the correct order
                fif_pattern = f"sub-{sub_nr}_ses-{ses_nr}_task-*_run-*_meg_tsss.fif"
                fif_files = ses_path.glob(fif_pattern)
                fif_files_sorted = sorted(fif_files, key=extract_sort_key) # sort the files in the folder to have the runs in ascending order and load the SRT files first and only then the WL files
                
                # if ICA has already been performed skip thee participant
                if ica_file.exists():
                    print(f"Skipping: subject {sub} -- ses {ses}; ICA exists. Remove ICA files from folder if you want to rerun them with these parameters")
                    continue
                
                #try:
                print(f"Processing file: subject {sub} -- ses {ses}")
                for fif_file in fif_files_sorted:
                    downsampled_file = downsampled_dir / fif_file.name.replace('_meg_tsss.fif', f'_meg_tsss_ds-notch-artf-{param_res_Fs}Hz.fif')
                    if not downsampled_file.exists():
                        data = mne.io.read_raw_fif(fif_file, preload=True)
                        
                        # Add block annotations and events (keeping these per-file as they're file-specific)
                        anot = block_annotate(data, fif_file)
                        data.set_annotations(anot)
                        read_events(data, fif_file)
                        
                        # Basic preprocessing
                        data.pick(['mag', 'grad'])
                        data.resample(param_res_Fs)
                        data.filter(param_filt_hp, None)
                        data.notch_filter(np.arange(50, 201, 50))
                        
                        data.save(downsampled_file, overwrite=True)
                    
                    else:
                        print(f"File already exists: {downsampled_file}")
                    
                    downsampled_files.append(downsampled_file)
                
                print(f"Loading and concatenating files for {sub} {ses}")
                data_list        = [mne.io.read_raw_fif(file, preload=True) for file in downsampled_files]
                data_combined    = mne.concatenate_raws(data_list, on_mismatch='warn')
                
                print("Detecting artifacts on concatenated data...")
                artifact_annot = detect_artifacts(data_combined, reject_speech=False)
                data_combined.set_annotations(data_combined.annotations + artifact_annot)
                
                manual_annot = manual_artifact_check(data_combined)
              
                # Standardize origin times and combine annotations
                current_annot = data_combined.annotations
                if manual_annot is not None and len(manual_annot) > 0:
                    # Set the origin time of new annotations to match existing ones
                    manual_annot.orig_time = current_annot.orig_time if current_annot.orig_time is not None else None
                    all_annotations = current_annot + manual_annot
                    data_combined.set_annotations(all_annotations)
              
                
                del data_list
                gc.collect() # free up memory just in case
                
                # run ICA
                ica = ICA(method='picard',  # robust newer ICA based on infoMax 
                          noise_cov = None,  # No empty room recordings so PCA based default whitening is used
                          n_components=.98, # percentage of variance of the data extracted from PCA that is used for the ICA (based on Chris)      
                          max_iter='auto',  # max n of iterations to try and have the algorithm converge 
                          random_state=rstate,  # integer that sets the seed for the random number generator of the algorithm (for reproducability). Using same as Gaby, but should be arbitrary
                          fit_params = {'extended': True}, # uses the extended Picard algorithm that takes longer but performs better
                          verbose=True)
                
                ica.fit(data_combined, 
                       reject_by_annotation=True,
                       reject={'grad': 8000e-13, 'mag': 8000e-15}) # , reject=reject, decim=10, verbose=True)        
                
                del data_combined 
                gc.collect() 
                
                # save files 
                ica.save(ica_dir / f"ica_projsub-{sub_nr}_ses-{ses_nr}_artf_{rstate}.fif")    
                          
                print(f"Successfully performed filtering and ICA on: {fif_file}")
                #except Exception as e:
                    # Log the error with the file path
                #    logging.error(f"Failed to process {fif_file}: {e}")
                 #   print(f"Error processing {fif_file}. Logged to {log_file}.")
                
    # %% 
    # Filter settings
    #l_freq = 0 # no low filter
    #h_freq = 270 # below cHPI 
    #notch_freq = np.arange(50, 251, 50)
        