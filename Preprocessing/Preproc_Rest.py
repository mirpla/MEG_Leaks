#%%
import os
import re
import mne 
import glob
import numpy as np
from pathlib import Path
from meg_analysis.Scripts.Preprocessing.Preproc_Functions import find_anot_onset
# %% 
def Crop_Rest_Events(script_dir, sub_folders = None, Overwrite = False):
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
    warning_path    = data_path /"event_warnings_rest.txt"
    
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
                out_dir         = ses_path / 'rest'
                
                out_dir.mkdir(exist_ok = True)
                
                events_WL_list = glob.glob(str(event_path_wl))
                events_WL_sort = sorted(events_WL_list, key=lambda x: int(re.search(r'run-(\d+)', x).group(1)))
    
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
                
                try: 
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
                    
                except: # in case there is no 473 because of a late press on record
                    srt_rest_file    = out_dir / Path(out_name+ 'rest-bl-srt_meg.fif') # path-name for the rest period for the corresponding block
                    
                    if srt_rest_file.exists() and Overwrite == False:
                        print(f'Skipping: {sub}-{ses}-SRT_BL; file already exists')                       
                    else: 
                        idx_srt     = np.where(event_dict_srt['STI101'][:, 2] == 1024)[0][0]
                        end_time    = time_srt[idx_srt]
                        start_time  = srt_onset[2]
                        
                        warn_message = f"Warning: {sub}_{ses}_SRT_BL was started late and is shorter than 180 seconds by {end_time-start_time} s.\n"
                        print(warn_message)
                        with open(warning_path, "a") as f:
                            f.write(warn_message)
                            f.close()
                        
                        data.crop(tmin= start_time, tmax = end_time)
                        data.save(srt_rest_file, overwrite=True)
                        del data
                                             
                # %% WL
                # do the same but for the WL. the first block contains a second rest period, for the rest extract the rest period at the end of the blocks
                wl_pattern  = r'WL_run-\d+'
                for blck, event_file, in enumerate(events_WL_sort):
                    # load in data again after wl or srt cropping; Not time efficient but done for RAM reasons
                    data = mne.io.read_raw_fif(data_pattern, preload=False)
                    
                    bad_annotations = mne.read_annotations(annot_file)
                    data.set_annotations(data.annotations + bad_annotations)
                    
                    event_array     = np.load(event_file, allow_pickle=True) # WL 1 = start - 217 | WL rest = 256 - 217/end
                    event_dict_wl   = event_array.item()
                    time_wl         = event_dict_wl['STI101'][:, 0] / orig_sf
                   
                    # extract the pattern, compare it to the current event-file and extract the corresponding annotation
                    wl_target   = re.search(wl_pattern, event_file)
                    wl_onset    = find_anot_onset(data_anot,wl_target.group(0))
                    
                    out_name    = Path(event_file).name[:-16] # make basis for the path-names       
                    # %% WL Baseline 
                    # locate first wl rest period
                    if Path(event_file) == (ses_path / 'events' / f'{sub}_{ses}_task-WL_run-1_events.npy'):
                        end_wl =  np.where(event_dict_wl['STI101'][:, 2] == 217)
                        if len(end_wl[0]) == 0:
                            end_wl = np.where(event_dict_wl['STI101'][:, 2] == 1024)
                        idx_first_wl    = end_wl[0][0]
                        end_time        = time_wl[idx_first_wl]
                        start_time      = end_time - 180      
                        
                        wl_bl_file      = out_dir / Path(out_name + 'rest-bl-wl_meg.fif') #path-name for the wl baseline / post-SRT measure 
                        if wl_bl_file.exists() and Overwrite == False:
                            print(f'Skipping: {sub}-{ses}-block{blck+1}-WL_BL; file already exists')                       
                        else:
                            if start_time <0:
                                start_time = 0.0
                                print(f"Warning: The first segment is shorter than 180 seconds for {sub}, {ses}.")
                                with open(warning_path, "a") as f:
                                    f.write(f"Warning: WL PRE is shorter than 180 seconds for {sub}, {ses}.\n")
                                    f.close()
                            data.crop(tmin= start_time + wl_onset[0], tmax = end_time+wl_onset[0])    
                            data.save(wl_bl_file, overwrite=True)                           
                            del data
                            
                        
                    # %% WL Rest
                    #Reload the original data to avoid memory issues
                    data = mne.io.read_raw_fif(data_pattern, preload=False)
                    
                    bad_annotations = mne.read_annotations(annot_file)
                    data.set_annotations(data.annotations + bad_annotations)
                   
                    if time_wl.size != 0:
                        idx_wl          = np.where(event_dict_wl['STI101'][:, 2] == 256)[0][0]
                        start_time      = time_wl[idx_wl]-wl_onset[2]
                        end_time        = start_time + 180   
                    else: # account cases where recording started in the middle of rest period
                        start_time      = 0.0
                        if wl_onset[1] > 180: # if there is no start trigger the file HAS to be shorter than 180, so longer end-file is meaningless
                            end_time        = 179
                        else:
                            end_time        = wl_onset[1]
                    wl_rest_file    = out_dir / Path(out_name+ f'rest-{blck+1}_meg.fif') # path-name for the rest period for the corresponding block
                    
                    if wl_rest_file.exists() and Overwrite == False:
                        print(f'Skipping: {sub}-{ses}-block{blck+1}-WL_BL; file already exists')                       
                        continue
                    
                    if end_time > wl_onset[1]: # make sure recording wasn't terminated prematurely in rest period
                        warn_message = f"Warning: {sub}_{ses}_{wl_target[0]} is shorter than 180 seconds by {end_time-wl_onset[1]} s.\n"
                        print(warn_message)
                        with open(warning_path, "a") as f:
                            f.write(warn_message)
                            f.close()
                        if start_time< wl_onset[1]:
                            end_time = wl_onset[1]
                        else:
                           warn_message = f"Warning: {sub}_{ses}_{wl_target[0]} CANNOT BE COMPUTED.\n"
                           print(warn_message)
                           with open(warning_path, "a") as f:
                               f.write(warn_message)
                               f.close()
                               
                           continue
                    
                    if end_time+wl_onset[0] > data.times[-1]:
                        warn_message = f"Warning: {sub}_{ses}_{wl_target[0]} is shorter than 180 seconds by {end_time + wl_onset[0] - data.times[-1]} s.\n"
                        print(warn_message)
                        with open(warning_path, "a") as f:
                            f.write(warn_message)
                            f.close()
                        data.crop(tmin= (start_time+wl_onset[0]), tmax = (data.times[-1]))
                    
                    else:
                        data.crop(tmin= (start_time+wl_onset[0]), tmax = (end_time+wl_onset[0]))
                    
                    data.save(wl_rest_file, overwrite=True)
                    del data
                    
                    print(f"next {event_file}")
                    

# %% Epoch the Data
def Epoch_Rest(script_dir, epoch_dur, sub_folders = None, ses_folders=None, Overwrite = False):
    """
    Epoch the rest data, using pre-annotated bad sections to reject artifacts.
    
    Parameters:
    -----------
    script_dir : Path
        Path to the script directory.
    epoch_dur : float
        Duration of epochs in seconds
    sub_folders : list or None
        List of subject folders to process. If None, processes all subjects in the data folder.
    ses_folders : list or None
        List of session folders to process. If None, processes both 'ses-1' and 'ses-2'.
    Overwrite : bool
        If True, overwrite existing files. Default is False.
    """
    base_path       = script_dir.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' # Folder containing the Data
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # creat regular expression pattern to identify folders of processed subjects    
    if sub_folders is None:
        sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and reg_pattern.match(d)] # find the folders starting with sub-XX where XX is the subject number with a leading 0

    if ses_folders is None:
        ses_folders = ['ses-1', 'ses-2']
    
    # %% Loop over the processed subjects 
    for sub in sub_folders:
        folder_path = data_path / sub
                 
        for ses in ses_folders:
            ses_path = folder_path / ses / 'meg'
             
            if ses_path.exists():               
                rest_dir = ses_path / 'rest'
                
                if not rest_dir.exists() and Overwrite == False:
                    print(f"Rest directory does not exist for {sub}/{ses}, skipping.")
                    continue
                
                data_pattern = [f'{sub}_{ses}_task-SRT_rest-bl-srt_meg.fif', f'{sub}_{ses}_task-WL_rest-bl-wl_meg.fif']
                
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