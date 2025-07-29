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
    Crop the individual block files into the relevant rest periods.
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
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # create regular expression pattern to identify folders of processed subjects
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
                individual_data_path = ses_path / 'individual_post_ica'
                individual_annot_path = ses_path / 'individual_annotations'
                
                if not individual_data_path.exists():
                    print(f"Individual data path does not exist for {sub}/{ses}, skipping.")
                    continue
                    
                if not individual_annot_path.exists():
                    print(f"Individual annotations path does not exist for {sub}/{ses}, skipping.")
                    continue
                
                event_path_srt  = ses_path / 'events' / f"{sub}_{ses}_task-SRT_run-1_events.npy"# first base-line rest period
                event_path_wl   = ses_path / 'events' / f"{sub}_{ses}_task-WL_run*_events.npy" # load in the rest of the rest-periods
                out_dir         = ses_path / 'rest'
                
                # Check if event files exist before proceeding
                if not event_path_srt.exists():
                    print(f"SRT event file not found for {sub}/{ses}: {event_path_srt}")
                    continue
                
                events_WL_list = glob.glob(str(event_path_wl))
                if len(events_WL_list) == 0:
                    print(f"No WL event files found for {sub}/{ses}: {event_path_wl}")
                    continue
                
                out_dir.mkdir(exist_ok = True)
                
                events_WL_sort = sorted(events_WL_list, key=lambda x: int(re.search(r'run-(\d+)', x).group(1)))
    
                # find the original sampling frequency by reading info of first instance of pre-downsampled data
                if not Path(orig_path).exists():
                    print(f"Original data file not found for {sub}/{ses}: {orig_path}")
                    continue
                    
                info_orig   = mne.io.read_info(orig_path)
                orig_sf     = info_orig['sfreq']
                
                # %% SRT Baseline
                # Load SRT data and annotations
                srt_data_file = individual_data_path / f"{sub}_{ses}_task-SRT_run-1_meg_tsss_notch-ds-500Hz_post_ica_r{seed}_.fif"
                srt_annot_file = individual_annot_path / f"{sub}_{ses}_task-SRT_run-1_meg_tsss_notch-ds-500Hz_post_ica_r{seed}_artf_annot.fif"
                
                if not srt_data_file.exists():
                    print(f"SRT data file not found: {srt_data_file}")
                    continue
                    
                if not srt_annot_file.exists():
                    print(f"SRT annotation file not found: {srt_annot_file}")
                    continue
                
                # Load the SRT data
                data = mne.io.read_raw_fif(srt_data_file, preload=False)
                data_anot = data.annotations
                
                # Load and apply pre-annotated bad segments with proper handling
                try:
                    bad_annotations = mne.read_annotations(srt_annot_file)
                    print(f"Loaded {len(bad_annotations)} annotations from {srt_annot_file}")
                    print(f"Original annotation descriptions: {list(set(bad_annotations.description))}")
                    
                    # Ensure annotations are marked as "bad" for epoching to work properly
                    bad_descriptions = []
                    for desc in bad_annotations.description:
                        if 'bad' not in desc.lower():
                            bad_descriptions.append('BAD_' + desc)
                        else:
                            bad_descriptions.append(desc)
                    
                    # Create new annotations with proper "bad" labels and matching orig_time
                    corrected_annotations = mne.Annotations(
                        onset=bad_annotations.onset,
                        duration=bad_annotations.duration,
                        description=bad_descriptions,
                        orig_time=data.info['meas_date']  # Match the data's measurement date
                    )
                    
                    data.set_annotations(data.annotations + corrected_annotations)
                    print(f"Applied {len(corrected_annotations)} bad annotations to SRT data")
                    print(f"Total annotations after loading: {len(data.annotations)}")
                    
                except Exception as e:
                    print(f"Error loading SRT annotations from {srt_annot_file}: {e}")
                    print(f"Trying alternative annotation loading approach...")
                    
                    # Alternative approach: replace existing annotations entirely
                    try:
                        bad_annotations = mne.read_annotations(srt_annot_file)
                        
                        # Ensure annotations are marked as "bad"
                        bad_descriptions = []
                        for desc in bad_annotations.description:
                            if 'bad' not in desc.lower():
                                bad_descriptions.append('BAD_' + desc)
                            else:
                                bad_descriptions.append(desc)
                        
                        # Create new annotations with data's orig_time
                        corrected_annotations = mne.Annotations(
                            onset=bad_annotations.onset,
                            duration=bad_annotations.duration,
                            description=bad_descriptions,
                            orig_time=data.info['meas_date']
                        )
                        
                        # Replace annotations instead of adding
                        data.set_annotations(corrected_annotations)
                        print(f"Replaced annotations with {len(corrected_annotations)} bad annotations")
                        
                    except Exception as e2:
                        print(f"Failed alternative approach: {e2}")
                        print("Proceeding without annotations for this file")
                        continue
                
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
                        
                        print(f"Cropping SRT from {start_time:.2f}s to {end_time_sc:.2f}s")
                        data.crop(tmin= start_time, tmax = end_time_sc)
                        
                        # Check annotations after cropping
                        print(f"Annotations after cropping SRT: {len(data.annotations)}")
                        bad_annots_after = [desc for desc in data.annotations.description if 'bad' in desc.lower() or 'BAD' in desc]
                        print(f"Bad annotations after cropping: {len(bad_annots_after)}")
                        if len(bad_annots_after) > 0:
                            print(f"Bad annotation descriptions: {set(bad_annots_after)}")
                        
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
                        
                        print(f"Cropping SRT (late start) from {start_time:.2f}s to {end_time:.2f}s")
                        data.crop(tmin= start_time, tmax = end_time)
                        
                        # Check annotations after cropping
                        print(f"Annotations after cropping SRT (late): {len(data.annotations)}")
                        bad_annots_after = [desc for desc in data.annotations.description if 'bad' in desc.lower() or 'BAD' in desc]
                        print(f"Bad annotations after cropping: {len(bad_annots_after)}")
                        
                        data.save(srt_rest_file, overwrite=True)
                        del data
                                             
                # %% WL
                # do the same but for the WL. the first block contains a second rest period, for the rest extract the rest period at the end of the blocks
                wl_pattern  = r'WL_run-\d+'
                for blck, event_file, in enumerate(events_WL_sort):
                    # Extract run number from event file
                    run_match = re.search(r'run-(\d+)', event_file)
                    if not run_match:
                        print(f"Could not extract run number from {event_file}")
                        continue
                    run_num = run_match.group(1)
                    
                    # Load corresponding WL data and annotation files
                    wl_data_file = individual_data_path / f"{sub}_{ses}_task-WL_run-{run_num}_meg_tsss_notch-ds-500Hz_post_ica_r{seed}_.fif"
                    wl_annot_file = individual_annot_path / f"{sub}_{ses}_task-WL_run-{run_num}_meg_tsss_notch-ds-500Hz_post_ica_r{seed}_artf_annot.fif"
                    
                    if not wl_data_file.exists():
                        print(f"WL data file not found: {wl_data_file}")
                        continue
                        
                    if not wl_annot_file.exists():
                        print(f"WL annotation file not found: {wl_annot_file}")
                        continue
                    
                    # load in data for current WL run
                    data = mne.io.read_raw_fif(wl_data_file, preload=False)
                    data_anot = data.annotations
                    
                    # Load and apply pre-annotated bad segments with proper handling
                    try:
                        bad_annotations = mne.read_annotations(wl_annot_file)
                        print(f"Loaded {len(bad_annotations)} annotations from {wl_annot_file}")
                        print(f"Original annotation descriptions: {list(set(bad_annotations.description))}")
                        
                        # Ensure annotations are marked as "bad" for epoching to work properly
                        bad_descriptions = []
                        for desc in bad_annotations.description:
                            if 'bad' not in desc.lower():
                                bad_descriptions.append('BAD_' + desc)
                            else:
                                bad_descriptions.append(desc)
                        
                        # Create new annotations with proper "bad" labels and matching orig_time
                        corrected_annotations = mne.Annotations(
                            onset=bad_annotations.onset,
                            duration=bad_annotations.duration,
                            description=bad_descriptions,
                            orig_time=data.info['meas_date']  # Match the data's measurement date
                        )
                        
                        data.set_annotations(data.annotations + corrected_annotations)
                        print(f"Applied {len(corrected_annotations)} bad annotations to WL run-{run_num}")
                        print(f"Total annotations after loading WL run-{run_num}: {len(data.annotations)}")
                        
                    except Exception as e:
                        print(f"Error loading WL annotations from {wl_annot_file}: {e}")
                        print(f"Trying alternative annotation loading approach...")
                        
                        # Alternative approach: replace existing annotations entirely
                        try:
                            bad_annotations = mne.read_annotations(wl_annot_file)
                            
                            # Ensure annotations are marked as "bad"
                            bad_descriptions = []
                            for desc in bad_annotations.description:
                                if 'bad' not in desc.lower():
                                    bad_descriptions.append('BAD_' + desc)
                                else:
                                    bad_descriptions.append(desc)
                            
                            # Create new annotations with data's orig_time
                            corrected_annotations = mne.Annotations(
                                onset=bad_annotations.onset,
                                duration=bad_annotations.duration,
                                description=bad_descriptions,
                                orig_time=data.info['meas_date']
                            )
                            
                            # Replace annotations instead of adding
                            data.set_annotations(corrected_annotations)
                            print(f"Replaced annotations with {len(corrected_annotations)} bad annotations")
                            
                        except Exception as e2:
                            print(f"Failed alternative approach: {e2}")
                            print("Proceeding without annotations for this file")
                            continue
                    
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
                            
                            print(f"Cropping WL baseline from {start_time + wl_onset[0]:.2f}s to {end_time+wl_onset[0]:.2f}s")
                            data.crop(tmin= start_time + wl_onset[0], tmax = end_time+wl_onset[0])
                            
                            # Check annotations after cropping
                            print(f"WL baseline annotations after cropping: {len(data.annotations)}")
                            bad_annots_after = [desc for desc in data.annotations.description if 'bad' in desc.lower() or 'BAD' in desc]
                            print(f"Bad annotations after cropping: {len(bad_annots_after)}")
                            if len(bad_annots_after) > 0:
                                print(f"Bad annotation descriptions: {set(bad_annots_after)}")
                            
                            data.save(wl_bl_file, overwrite=True)                           
                            del data
                            
                        
                    # %% WL Rest
                    #Reload the WL data for the rest processing
                    data = mne.io.read_raw_fif(wl_data_file, preload=False)
                    
                    # Reapply annotations for the rest processing
                    try:
                        bad_annotations = mne.read_annotations(wl_annot_file)
                        
                        # Ensure annotations are marked as "bad" for epoching to work properly
                        bad_descriptions = []
                        for desc in bad_annotations.description:
                            if 'bad' not in desc.lower():
                                bad_descriptions.append('BAD_' + desc)
                            else:
                                bad_descriptions.append(desc)
                        
                        # Create new annotations with proper "bad" labels and matching orig_time
                        corrected_annotations = mne.Annotations(
                            onset=bad_annotations.onset,
                            duration=bad_annotations.duration,
                            description=bad_descriptions,
                            orig_time=data.info['meas_date']  # Match the data's measurement date
                        )
                        
                        data.set_annotations(data.annotations + corrected_annotations)
                        
                    except Exception as e:
                        print(f"Error reloading WL annotations: {e}")
                        # Try alternative approach
                        try:
                            bad_annotations = mne.read_annotations(wl_annot_file)
                            
                            bad_descriptions = []
                            for desc in bad_annotations.description:
                                if 'bad' not in desc.lower():
                                    bad_descriptions.append('BAD_' + desc)
                                else:
                                    bad_descriptions.append(desc)
                            
                            corrected_annotations = mne.Annotations(
                                onset=bad_annotations.onset,
                                duration=bad_annotations.duration,
                                description=bad_descriptions,
                                orig_time=data.info['meas_date']
                            )
                            
                            data.set_annotations(corrected_annotations)
                            
                        except Exception as e2:
                            print(f"Failed to load annotations for WL rest processing: {e2}")
                            continue
                   
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
                        print(f'Skipping: {sub}-{ses}-block{blck+1}-WL_REST; file already exists')                       
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
                        
                        print(f"Cropping WL rest (short) from {start_time+wl_onset[0]:.2f}s to {data.times[-1]:.2f}s")
                        data.crop(tmin= (start_time+wl_onset[0]), tmax = (data.times[-1]))
                    
                    else:
                        print(f"Cropping WL rest from {start_time+wl_onset[0]:.2f}s to {end_time+wl_onset[0]:.2f}s")
                        data.crop(tmin= (start_time+wl_onset[0]), tmax = (end_time+wl_onset[0]))
                    
                    # Check annotations after cropping
                    print(f"WL rest annotations after cropping: {len(data.annotations)}")
                    bad_annots_after = [desc for desc in data.annotations.description if 'bad' in desc.lower() or 'BAD' in desc]
                    print(f"Bad annotations after cropping: {len(bad_annots_after)}")
                    if len(bad_annots_after) > 0:
                        print(f"Bad annotation descriptions: {set(bad_annots_after)}")
                    
                    data.save(wl_rest_file, overwrite=True)
                    del data
                    
                    print(f"Processed {event_file}")
                    

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
    
    reg_pattern = re.compile(r'^sub-\d{2}$') # create regular expression pattern to identify folders of processed subjects    
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
                
                if not rest_dir.exists():
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
                            
                            # Check what annotations are present before epoching
                            print(f"Annotations before epoching: {len(data.annotations)}")
                            bad_annots = [desc for desc in data.annotations.description if 'bad' in desc.lower() or 'BAD' in desc]
                            print(f"Bad annotations before epoching: {len(bad_annots)}")
                            if len(bad_annots) > 0:
                                print(f"Bad annotation descriptions: {set(bad_annots)}")
                            
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
                            print(f"Saved {out_file} with {len(data_epoch)}/{epoch_n} epochs (rejected {epoch_n - len(data_epoch)} epochs)")
                        else: 
                            print(f'File {data_file.name} not found, skipping')
                    else:
                        print(f'{out_file.name} already exists, skipping!')