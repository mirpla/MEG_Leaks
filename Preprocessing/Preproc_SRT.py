#%% 
import os
import re
import mne 
import numpy as np
import pandas as pd
from pathlib import Path
from meg_analysis.Scripts.Preprocessing.Preproc_Functions import find_anot_onset

def Epoch_SRT(script_dir, seed,lock = 'resp' , pre_window=None, post_window = None, bl_size = 0.2,sub_folders=None, ses_folders=['ses-1'], overwrite_epochs=False):
    """    
    Epoch the SRT data for each subject and session.
    Args:
        script_dir (Path): 
            Path to the script directory. (MANDATORY)
        seed (int): 
            Seed for the ICA run used to process the data. (MANDATORY)
        lock (str, optional): 
            'resp' for response-locked or 'stim' for stimulus-locked epochs. Defaults to 'resp' as 'stim is not implemented yet.
        pre_window (float, optional): 
            Pre-stimulus time in seconds. Defaults depend on lock type. (see code)
        post_window (float, optional): 
            Post-stimulus time in seconds. Defaults depend on lock type. (see code)
        bl_size (float, optional): 
            Baseline size in seconds. Defaults to 0.2.
        sub_folders (list, optional): 
            List of subject folders. If None, all sub-XX folders in the data path will be used.
        ses_folders (list, optional): 
            List of session folders. Defaults to ['ses-1'].
        overwrite_epochs (bool, optional): If True, overwrite existing epochs files. Defaults to False.
    """
    
    # set default pre and post windows if not provided depending on the lock type
    if lock == 'resp':
        if pre_window is None:
            pre_window = 0.5   
        if post_window is None: 
            post_window = 0.4
            

    # set the path structure
    base_path       = script_dir.parent.parent.parent # Root folder
    data_path       = base_path / 'Data' # Folder containing the Data

    if sub_folders is None:
        reg_pattern = re.compile(r'^sub-\d{2}$') # creat regular expression pattern to identify folders of processed subjects
        sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and reg_pattern.match(d)] # find the folders starting with sub-XX where XX is the subject number with a leading 0
        
    # %% Loop over the processed subjects 
    for sub in sub_folders:
        if sub == 'sub-41' or sub == 'sub-42' or sub == 'sub-55' or sub == 'sub-64' or sub == 'sub-69':  # skip sub-41 
            print(f"Skipping {sub}.")
            continue
        folder_path = data_path / sub
        sub_num     = re.search(r'\d+', sub)
        if sub_num:
            sub_num = int(sub_num.group(0))
            for ses in ses_folders:
                ses_path = folder_path / ses / 'meg'
                    
                if ses_path.exists():                
                    # Output directory
                    out_dir = ses_path / 'srt'                
                    out_dir.mkdir(exist_ok = True)
                        
                    for run_num in [1, 2, 3]:
                        # Define the epochs file name and only continue if it does not exist if overwrite_epochs is False
                        epochs_fname = out_dir / f"{sub}_{ses}_task-SRT_run-{run_num}_{lock}_epo.fif"
                        if epochs_fname.exists() and not overwrite_epochs:
                            print(f"Skipping {sub} {ses} run {run_num} - epochs file already exists: {epochs_fname}")
                            continue
                        print(f"Processing {sub} {ses} run {run_num}")
                        
                                    # Path to individual post-ICA file
                        data_pattern = ses_path / 'individual_post_ica' / f"{sub}_{ses}_task-SRT_run-{run_num}_meg_tsss_notch-ds-500Hz_post_ica_r{seed}_.fif"
                        
                        # Path to the original data file for sample rate             
                        orig_path   = f"{ses_path}/{sub}_{ses}_task-SRT_run-1_meg_tsss.fif"
                        info_orig   = mne.io.read_info(orig_path)
                        orig_sf     = info_orig['sfreq']
                        
                        # Path to corresponding annotation file
                        annot_file = ses_path / 'individual_annotations' / f"{sub}_{ses}_task-SRT_run-{run_num}_meg_tsss_notch-ds-500Hz_post_ica_r{seed}_artf_annot.fif"
                        
                        # Check if data file exists
                        if not data_pattern.exists():
                            print(f"Warning: Data file not found: {data_pattern}")
                #                       continue
                        
                        # Load the individual block data
                        data = mne.io.read_raw_fif(str(data_pattern), preload=False)
                        
                        # Load and apply pre-annotated bad segments if they exist
                        if annot_file.exists():
                            bad_annotations = mne.read_annotations(str(annot_file))
                            data.set_annotations(data.annotations + bad_annotations)
                            print(f"Applied bad segment annotations from {annot_file}")
                        else:
                            print(f"Warning: No bad segment annotations found at {annot_file}")
                        
                        # Load the corresponding SRT events 
                        event_path_srt = ses_path / 'events' / f"{sub}_{ses}_task-SRT_run-{run_num}_events.npy" 
                        
                        if not event_path_srt.exists():
                            print(f"Warning: Event file not found: {event_path_srt}")
                            #continue
                            
                        event_array = np.load(str(event_path_srt), allow_pickle=True)
                        event_dict_srt = event_array.item()
                        
                        # Load behavioral data
                        beh_pattern = f'*SRTT_blocks{run_num}.txt'
                        beh_path = folder_path / ses / 'beh'
                        beh_file = list(beh_path.glob(beh_pattern))
                        
                        if not beh_file:
                            print(f"Warning: Behavioral file not found for run {run_num}")
                            #continue
                        
                        beh_data = pd.read_csv(beh_file[0], sep='\t', header=None)
                        beh_data.columns = [
                            'subject_id',
                            'block_number',
                            'trial_number',
                            'target_trial',
                            'trial_code',
                            'rt_index',
                            'rt_ring',
                            'rt_middle',
                            'rt_pinky',
                            'session'
                        ]
                        
                        # Since we're working with individual blocks, we don't need to crop
                        # The data file already contains only the relevant time window
                        raw_run = data.copy()
                        
                        triggers = {} # pinky, ring, middle, index respectively
                        triggers['fingers'] = mne.pick_events(event_dict_srt['STI102']) # pick the events for the button presses
                       
                        # Calculate relative trigger values based on trigger differences 
                        sti101_events = event_dict_srt['STI101']
                        trigger_diff = sti101_events[:, 2] - sti101_events[:, 1]
                        
                        # Find the actual trigger values that correspond to diff == 11 and diff == 26
                        screen_value = sti101_events[trigger_diff == 11, 2][0]  # screen trigger
                        resp_value = sti101_events[trigger_diff == 26, 2][0]    # response trigger
                        
                        # Use mne.pick_events with the dynamically determined trigger values
                        triggers['screen_trig'] = mne.pick_events(event_dict_srt['STI101'], include=[screen_value, resp_value])

                        # Find indices for screen triggers and response triggers
                        index_screen = np.where(triggers['screen_trig'][:, 2] == screen_value)[0]
                        index_resp = np.where(triggers['screen_trig'][:, 2] == resp_value)[0]
                       
                        if lock == 'resp':
                            # find the indices of the 1317 events that are preceded by a 1291 event
                            preceding_samples = []
                            for idx in index_screen:
                                # Find the largest 1317 index that's smaller than current 1291 index
                                valid_resp_indices = index_resp[index_resp < idx]
                                if len(valid_resp_indices) > 0:
                                    preceding_idx = valid_resp_indices[-1]  # Get the most recent one
                                    preceding_samples.append(triggers['screen_trig'][preceding_idx, 0])

                            # Add the very last 1317 sample value
                            if len(index_resp) > 0:
                                preceding_samples.append(triggers['screen_trig'][index_resp[-1], 0])

                            # match the screen triggers with the correct response triggers AND get their event IDs
                            trigger_events = []
                            finger_samples = triggers['fingers'][:, 0]  # Extract the sample indices for button presses
                            finger_event_ids = triggers['fingers'][:, 2]  # Extract the event IDs for button presses

                            for sample in preceding_samples:
                                # Use searchsorted to find insertion point, then get the element just before
                                insert_idx = np.searchsorted(finger_samples, sample, side='left')
                                
                                if insert_idx > 0:  # There's at least one element before the insertion point                                
                                    closest_sample = finger_samples[insert_idx - 1]
                                    closest_event_id = finger_event_ids[insert_idx - 1]
                                    
                                    # Create MNE events array entry: [sample, prev_id, event_id]
                                    trigger_events.append([closest_sample, 0, closest_event_id])

                            # Convert to numpy array for MNE
                            trigger_events = np.array(trigger_events)
                            
                            # Since we're working with individual files, we need to adjust sample indices
                            # to account for the first sample of this specific block
                            trigger_events[:,0] = np.round((trigger_events[:,0] / orig_sf)*raw_run.info['sfreq']).astype(int)
                            
                            n_trials = len(trigger_events)
                            trial_indices = np.arange(n_trials)  # Original trial indices
                            
                            epochs = mne.Epochs(
                                raw_run,
                                events=trigger_events,  # or trl_onset for stim-locked
                                tmin=-pre_window,
                                tmax=post_window,
                                baseline=(-pre_window, -pre_window + bl_size),
                                reject_by_annotation=True,  # Don't reject yet
                                flat=dict(mag=1e-20, grad=1e-20), 
                                preload=True
                            )
                                        
                            n_original_trials = len(trigger_events)
                            n_kept_trials = len(epochs)

                            # check which trials were kept
                            kept_indices = epochs.selection
                            all_indices = np.arange(len(trigger_events))
                            dropped_indices = np.setdiff1d(all_indices, kept_indices)
                            print(f"Kept {len(kept_indices)} trials, dropped {len(dropped_indices)} trials")
                            print(f"Dropped trial indices: {dropped_indices}")
                            
                            # Now sync the behavioral data
                            if len(beh_data) == n_original_trials:
                                epochs.metadata = beh_data.iloc[kept_indices].copy()
                                print(f"Attached behavioral metadata for {n_kept_trials} trials (dropped {n_original_trials - n_kept_trials})")
                            elif len(beh_data) > n_original_trials:
                                KeyError(f"Warning: Behavioral data has {len(beh_data)} trials, but only {n_original_trials} events found. Metadata will not be attached.")  
                            elif len(beh_data) < n_original_trials:
                                KeyError(f"Warning: Behavioral data only has {len(beh_data)} trials, but {n_original_trials} events found. Metadata will not be attached.")
                        
                        #elif lock == 'stim':  Not yet implemented         
                        #    print('stim locked processing not implemented yet')
                        #    trl_onset = mne.pick_events(event_dict_srt['STI101'], include=[1291])
                        #    trl_onset[:,0] = np.round((trl_onset[:,0] / orig_sf)*raw_run.info['sfreq']).astype(int)

                        # Save epochs
                        epochs.save(epochs_fname, overwrite=True)
                        if overwrite_epochs and epochs_fname.exists():
                            print(f"Overwrote existing epochs file: {epochs_fname}")
                        else:
                            print(f"Saved new epochs file: {epochs_fname}")         
                    
                        # keep log of how many trials were removed/kept
                        print(f"Final epochs: {len(epochs)} trials kept out of {len(trigger_events)} original triggers")
                        if hasattr(epochs, 'drop_log'):
                            n_dropped = sum(len(log) > 0 for log in epochs.drop_log)
                            print(f"Dropped {n_dropped} trials due to artifacts/annotations")
