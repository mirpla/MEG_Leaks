import os
import re
import mne 
import glob
import numpy as np
from pathlib import Path
from meg_analysis.Scripts.Preprocessing.Preproc_Functions import find_anot_onset

def Epoch_SRT(script_dir, sub_folders = None, Overwrite = False):
    '''
    Crop the full length data files into the relevant rest periods.
    Include pre-annotated bad sections from annotation files.
    
    input: 
        script_dir: 
            path to the script directory 
        sub_folders: list of subjects to process (with the format ['sub-XX','sub-YY']). 
            If None, all subjects in the data folder will be processed.
        Overwrite: 
            boolean indicating whether to overwrite existing files (default = False).
    '''
    seed = 100 # seed for the ICA run used to process the data
    base_path       = script_dir.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' # Folder containing the Data
    warning_path    = data_path /"event_warnings_SRT.txt"
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # creat regular expression pattern to identify folders of processed subjects
    if sub_folders is None:
        sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and reg_pattern.match(d)] # find the folders starting with sub-XX where XX is the subject number with a leading 0
     
    # %% Loop over the processed subjects 
    for sub in sub_folders:
        folder_path = data_path / sub
               
        ses_folders = ['ses-1']#,'ses-2'] # give options for two sessions; stick to ses 1 for now
         
        for ses in ses_folders:
            ses_path = folder_path / ses / 'meg'
             
            if ses_path.exists():                
                orig_path       = f"{ses_path}/{sub}_{ses}_task-SRT_run-1_meg_tsss.fif"
                data_pattern    = f"{ses_path}/{sub}_{ses}_r{seed}_PostICA_Full.fif"
                
                # Load annotation file for bad segments
                annot_file      = f"{ses_path}/{sub}_{ses}_Artf_annot.fif"
                
                event_path_srt  = ses_path / 'events' / f"{sub}_{ses}_task-SRT_run-1_events.npy"# first base-line rest period
                event_path_wl   = ses_path / 'events' / f"{sub}_{ses}_task-WL_run*_events.npy" # load in the rest of the rest-periods
                out_dir         = ses_path / 'srt'
                
                out_dir.mkdir(exist_ok = True)
                
                # Load the data
                data = mne.io.read_raw_fif(data_pattern, preload=False)
                data_anot = data.annotations
                
                # Load and apply pre-annotated bad segments if they exist
                if Path(annot_file).exists():
                    bad_annotations = mne.read_annotations(annot_file)
                    data.set_annotations(data.annotations + bad_annotations)
                    print(f"Applied bad segment annotations from {annot_file}")
                else:
                    raise ValueError(f"No bad segment annotations found in {annot_file}")
                # find the original sampling frequency by reading info of first instance of pre-downsampled data
                info_orig   = mne.io.read_info(orig_path)
                orig_sf     = info_orig['sfreq']
                
                # %% SRT Baseline
                # correct the srt events for sampling rate and locate 217 as the end of the first rest period
                event_array     = np.load(event_path_srt, allow_pickle=True) # 473 - 1024 column 2
                event_dict_srt  = event_array.item()
                time_srt        = event_dict_srt['STI101'][:, 0] / orig_sf
                       
                srt_onset = find_anot_onset(data_anot,'SRT_run-1')
                
                out_name    = Path(event_path_srt).name[:-16]
            
                # %%
                srt_start   = np.where(event_dict_srt['STI101'][:, 2] == 473)
                srt_end     = np.where(event_dict_srt['STI101'][:, 2] == 1024)
                if  len(srt_start[0]) == 0 :
                    srt_start = np.where(event_dict_srt['STI101'][:, 2] == 1241) # trigger value is different in subject 25 for some reason. Adjusting for them
                    
                idx_srt         = srt_start[0][0]
                idx_srt_end     = srt_end[0][0]
                
                start_time      = time_srt[idx_srt] 
                end_time        = start_time + 180      # theoretical endtime
                end_time_sc     = time_srt[idx_srt_end] # measured endtime (as sanity check)
                
                srt_rest_file    = out_dir / Path(out_name+ 'rest-bl-srt_meg.fif') # path-name for the rest period for the corresponding block
                
                if srt_rest_file.exists() and Overwrite == False:
                    print(f'Skipping: {sub}-{ses}-SRT_BL; file already exists')                       
                
                else:
                    if end_time > end_time_sc: # make sure recording wasn't terminated prematurely in rest period
                        warn_message = f"Warning: {sub}_{ses}_SRT_BL is shorter than 180 seconds by {end_time-end_time_sc} s.\n"
                        print(warn_message)
                        with open(warning_path, "a") as f:
                            f.write(warn_message)
                            f.close()
                            
                    data.crop(tmin= start_time, tmax = end_time_sc)
                    data.save(srt_rest_file, overwrite=True)
                    del data
                                             
# %% --------------------------------------------------- Original Epoch script
                data_pattern = [f'{sub}_{ses}_task-SRT_rest-bl-srt_meg.fif']
                
                # Loop across files
                for fidx in range(1, 11):
                    if fidx == 10:
                        filename = f'{sub}_{ses}_task-WL_rrest-{fidx}_meg.fif'
                    else:
                        filename = f'{sub}_{ses}_task-WL_rest-{fidx}_meg.fif'
    
                    data_pattern.append(filename)
                    
                for fidx, filename in enumerate(data_pattern): 
                    out_file = rest_dir / f'{sub}_{ses}_rest-{fidx}_clean_epo.fif'
                    
                    if not out_file.exists() or Overwrite == True:       
                        data_file = rest_dir / filename
                        
                        if data_file.exists():
                            print(f"Processing {data_file}")
                            data = mne.io.read_raw_fif(data_file)
                            
                            # Annotations should already be applied from the cropping step
                            # No need to reapply them here
                            
                            sfreq = data.info['sfreq']
                            epoch_len = int(epoch_dur * sfreq)
                            epoch_n = data.n_times // epoch_len
                            events = np.array([[i * epoch_len, 0, 1] for i in range(epoch_n)])
                            events[:,0] = events[:,0] + data.first_samp  # correct for sample offset due to cropping
                            
                            # Create epochs, using reject_by_annotation=True to skip bad segments
                            data_epoch = mne.Epochs(data, events, 
                                                   event_id=1, 
                                                   tmin=0, 
                                                   tmax=epoch_dur-(1/sfreq),
                                                   baseline=None, 
                                                   preload=True,
                                                   reject_by_annotation=True)
                            
                            data_epoch.save(out_file, overwrite=True)
                            print(f"Saved {out_file} with {len(data_epoch)}/{epoch_n} epochs")
                        else: 
                            print(f'File {data_file.name} not found, skipping')
                    else:
                        print(f'{out_file.name} already exists, skipping!')