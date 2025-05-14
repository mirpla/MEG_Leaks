# %% 
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
import os
import warnings

os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.2'
# %% Import R's LOESS implementation via rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
stats = importr('stats')

# %%
def import_all_srt(data_path, ses=1):
    '''
    Imports SRT data for all subjects into DataFrames with proper structure
    
    parameters:
        data_path: path to folder containing all the subjects
        ses: optional, session to be analysed (default = 1 since everyone has 1 session)
    
    returns:
        subject_dfs: dictionary with subject IDs as keys and DataFrames as values
        all_data_df: combined DataFrame with all subjects
    '''
    # Paths and sequence loading
    pattern = f"sub-*/ses-{ses}/beh/*SRTT_blocks*.txt"
    data_files = list(data_path.rglob(pattern))
   
    # Column names for the data
    column_names = [
        'subject', 'block', 'b_trial', 'target', 'condition',
        'rt_button1', 'rt_button2', 'rt_button3', 'rt_button4', 'sequence_file'
    ]

    # preallocate data structures
    subject_dfs = {}
    
    for file_path in data_files: 
        subject_match = re.search(r'sub-(\d+)', str(file_path))
        if not subject_match:
           print(f"Could not extract subject ID from path: {file_path}")
           continue
        
        subject_id = int(subject_match.group(1))          
      
        try:
            # Read the data file
            block_data = pd.read_csv(
                file_path, 
                delimiter='\t', 
                names=column_names,
                dtype={
                    'subject': int,
                    'block': int,
                    'b_trial': int,
                    'target': int,
                    'condition': int,
                    'rt_button1': float,
                    'rt_button2': float,
                    'rt_button3': float,
                    'rt_button4': float,
                    'sequence_file': str
                }
            )
            
            # Add session information for reference
            block_data['session'] = ses
             
            # If subject already exists in the dictionary, append the data
            if subject_id in subject_dfs:
                subject_dfs[subject_id] = pd.concat([subject_dfs[subject_id], block_data], ignore_index=True)
            else:
                subject_dfs[subject_id] = block_data.copy()
                
            print(f"Processed file for subject {subject_id}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Add condition name for better readability
    for subject_id in subject_dfs:
        subject_dfs[subject_id]['condition_name'] = subject_dfs[subject_id]['condition'].map({
            82: 'random',
            84: 'sequence'
        })
    
    return subject_dfs
   
#%% LOESS R implementation

def r_loess(data, fit_limit, poly_degree=2):
    """
    Implementation of LOESS (LOcal regrESSion) smoothing using R's loess function
    
    Parameters:
    -----------
    data : numpy array
        2D array with x values in first column, y values in second column
    fit_limit : float
        Fraction of data used for smoothing (span parameter in R's loess)
    poly_degree : int
        Degree of polynomial to fit locally (1=linear, 2=quadratic)
    
    Returns:
    --------
    smoothed_data : numpy array
        Array with smoothed values
    f_lim_lower : numpy array
        Lower limit for filtering
    f_lim_upper : numpy array
        Upper limit for filtering
    """
    x, y = data[:, 0], data[:, 1].copy()
    
    #check for severe outliers that have to be removed before the loess filter
    y_mean = np.mean(y)
    y_std = np.std(y) * 2.5
    outlier_mask = y > y_mean + y_std

    # Create outlier-free data for LOESS filter 
    x_clean = x[~outlier_mask]
    y_clean = y[~outlier_mask]
    
    # Create an R dataframe
    df = pd.DataFrame({'x': x_clean, 'y': y_clean})
    r_df = pandas2ri.py2rpy(df)
    
    # Set up the formula
    formula = ro.Formula('y ~ x')
    
    # Run R's loess function (span is equivalent to fit_limit)
    # degree must be 0, 1, or 2 in R's implementation
    r_poly_degree = min(poly_degree, 2)  # R's loess only supports up to degree 2
    
    loess_fit = stats.loess(formula, data=r_df, span=fit_limit, degree=r_poly_degree)
    
    # merge original and masked data
    full_df = pd.DataFrame({'x': x})
    r_full_df = pandas2ri.py2rpy(full_df)
    
    # nan values may cause warnings but are not a problem and will simply propagate
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
        warnings.filterwarnings("ignore", message="invalid value encountered in add")
    
        # Predict smoothed values for all points (with outliers)
        smoothed = np.array(stats.predict(loess_fit, newdata=r_full_df))
        
        # Calculate residuals only for non-outlier points
        clean_residuals = y_clean - smoothed[~outlier_mask]
        std_dev = np.std(clean_residuals)
        
        # Calculate limits
        f_lim_lower = smoothed - 2 * std_dev
        f_lim_upper = smoothed + 2 * std_dev
    
    # Create result array with all information
    smoothed_data = np.column_stack((x, y, smoothed))
    
    return smoothed_data, f_lim_lower, f_lim_upper

# %% Lowess filter implementation
def lowess(data, fit_limit, order=1):
    """
    Implementation of LOWESS smoothing using statsmodels' lowess function
    
    Parameters:
    -----------
    data : numpy array
        2D array with x values in first column, y values in second column
    fit_limit : float
        Fraction of data used for smoothing (frac parameter in statsmodels)
    order : int
        Not used directly, kept for compatibility with original function signature
    
    Returns:
    --------
    smoothed_data : numpy array
        Array with smoothed values
    f_lim_lower : numpy array
        Lower limit for filtering
    f_lim_upper : numpy array
        Upper limit for filtering
    """
    x, y = data[:, 0], data[:, 1].copy()
    
    #check for severe outliers that have to be removed before the loess filter
    y_mean = np.mean(y)
    y_std = np.std(y) * 2.5
    y[y > y_mean + y_std] = np.nan
    
    
    # Apply LOWESS smoothing from statsmodels
    # The frac parameter is equivalent to fit_limit - fraction of data used for smoothing
    smoothed = sm_lowess(y, x, frac=fit_limit, it=3, return_sorted=False)
    
    # Calculate limits 
    residuals = y - smoothed
    std_dev = np.std(residuals)
    
    f_lim_lower = smoothed - 2.5 * std_dev
    f_lim_upper = smoothed + 2.5 * std_dev
    
    smoothed_data = np.column_stack((x, y, smoothed))
    
    return smoothed_data, f_lim_lower, f_lim_upper

# %% srt import and fitting
def srt_import_fit(base_path, fit_limit, sl_window,method='loess',poly_degree=2):
    """
    Import SRT data, apply filtering to remove outliers, calculate error rates and skill learning    
    Parameters:
    -----------
    base_path : str
        Base directory path
    fit_limit : list
        Limits for filtering outliers [random_limit, sequence_limit]
    sl_window : int
        Window size for skill learning calculation
    
    Returns:
    --------
    results : dict
        Dictionary containing the following
        - skill_learning : DataFrame with skill learning values
        - filtered_rts : DataFrame with filtered RT values along with single trial information
        - error_rates : DataFrame with error rates for each subject
    """
    # Paths and sequence loading
    seq_path    = base_path / 'meg_analysis' / 'Sequence_files'
    data_path   = base_path / 'Data'
    
    # sequence files of the respective sequences/trial structure for the congruent and incongruent Condition
    seq_ori = {}
    seq_ori[1] = pd.read_csv(seq_path / 'seq-1.txt', sep='\t', names=['block', 'target', 'ID'])
    seq_ori[2] = pd.read_csv(seq_path / 'seq-2.txt', sep='\t', names=['block', 'target', 'ID'])
    seq_ori[3] = pd.read_csv(seq_path / 'seq-3.txt', sep='\t', names=['block', 'target', 'ID']) # incongruent 64 <=
    
    # Load subject information
    sub_info = pd.read_csv(base_path / 'Data' / 'Subject_Information.csv', encoding='latin1')
    
    sub_info['SubNr'] = sub_info['ID'] % 1000  # Extract subject ID
    
    subject_dfs = import_all_srt(data_path, ses=1)
    
    results = {
        'skill_learning':   pd.DataFrame(columns=['subject', 'session', 'block', 'condition', 'skill_learning_value']),
        'filtered_rts':     pd.DataFrame(), #pd.DataFrame(columns=['subject_id', 'block', 'random', 'upper_limit', 'lower_limit','error_trials','error_button' , 'error_rt', 'error_value', 'method', 'sequence']),
        'error_rates':      pd.DataFrame(columns=['subject','block', 'error_value', 'total_trials','error_rate'])
    }
    
    # Choose the appropriate smoothing function
    if method.lower() == 'lowess':
        smoothing_function = lowess
    elif method.lower() == 'loess':
        # Use lambda to bind poly_degree parameter while keeping the same function signature
        smoothing_function = lambda data, fit_lim, order: r_loess(data, fit_lim, poly_degree)
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Choose 'lowess' or 'loess'.")
    
    for subject_id, subject_df in subject_dfs.items():
        cond = subject_df['sequence_file'].unique().astype(int)[0] # get the condition 1 = congruent 2 = incongruent
        sub_pos = sub_info.index[sub_info['sub'] == f'sub-{int(subject_id):02d}'].tolist() 
        # For each row, get the RT for the correct button based on target and detect the errors
        subject_df['correct_rt']    = np.nan
        subject_df['error_button']  = np.nan
        subject_df['error_value']   = np.nan 
        
        subject_df = subject_df.astype({'error_button': 'object', 'error_value': 'object'})
        
        for idx, row in subject_df.iterrows():
            target = int(row['target'])
            subject_df.at[idx, 'correct_rt'] = row[f'rt_button{target}']
            
            other_buttons   = [f'rt_button{i}' for i in range(1, 5) if i != target] # determine which non-target buttons were pressed
            error_buttons   = [btn for btn in other_buttons if row[btn] != 0]  # Collect button names
            if error_buttons: # store error buttons and their values if present 
                subject_df.at[idx, 'error_button']   = error_buttons
                subject_df.at[idx, 'error_value']    = [row[btn] for btn in error_buttons] 
                                   
        # Calculate error rates by dividing error by total trials
        total_trials = subject_df.groupby(['block', 'condition_name']).size().reset_index(name='total_trials')   
        grouped_errors = subject_df.groupby(['block', 'condition_name'])
        error_count = grouped_errors['error_value'].count().reset_index()
        error_rate_df = error_count.merge(total_trials, on=['block', 'condition_name'])
        error_rate_df['error_rate'] = error_rate_df['error_value'] / error_rate_df['total_trials']

        error_rate_df['subject'] = subject_id
        
        # Append to results
        results['error_rates'] = pd.concat([results['error_rates'],error_rate_df], ignore_index=True)
               
        is_sub_rand = bool(sub_info.loc[sub_pos[0], 'Random']) # check whether subject had random or sequence control
        is_control = 2 in subject_df['sequence_file'].unique().astype(int) # check which condition the SRT file is in 
        # For control session (session 2) who were only random, just apply the filter to random trials
        if is_control & is_sub_rand: # check if it's a control trial and whether a random control was used
            for block in subject_df['block'].unique():
                block_data = subject_df[subject_df['block'] == block].sort_values('b_trial') # select specific block
                
                if len(block_data) > 0: # make sure there is actual data for this block
                    # Prepare data for LOWESS
                    rt_array = block_data['correct_rt'].values
                    trial_nums = np.arange(1, len(rt_array) + 1)
                    data_array = np.column_stack((trial_nums, rt_array))
                    
                    # Apply filter
                    filtered_data, lower_limit, upper_limit = smoothing_function(data_array, fit_limit[0], 1)
                    
                    # Store results - start with DF and add the subject and block information
                    lowess_df = pd.DataFrame(filtered_data, columns=['trial', 'original_rt', 'smoothed_rt'])
                    lowess_df['subject_id']     = subject_id
                    lowess_df['block']          = block
                    lowess_df['random']         = 0 # 0 = all random; 1 = pre-seq random 2 = post-seq random
                    lowess_df['upper_limit']    = upper_limit
                    lowess_df['lower_limit']    = lower_limit
                    lowess_df['error_trials']   = block_data['error_button'].values
                    lowess_df['error_rt']       = block_data['error_value'].values
                    lowess_df['method']         = method
                    lowess_df['target']         = block_data['target'].values
                    
                    # Set MultiIndex to make data more readable
                    lowess_df.set_index(['subject_id', 'block','random'], inplace=True)
                    
                    results['filtered_rts'] = pd.concat([results['filtered_rts'], lowess_df], ignore_index=False)
                    results['filtered_rts'] = results['filtered_rts'].sort_index()
                else:
                    print(f'No Data for subject {subject_id} - Block: {block}....')
                    print('Skipping!') 
        # For experimental session (session 1), split random trials and apply filters
        else: 
            for block in subject_df['block'].unique():
                block_data = subject_df[(subject_df['session'] == 1) & (subject_df['block'] == block)]
                if not block_data['b_trial'].is_unique:
                    print("-----------Error-----------") # find if trial log has duplicates 
                    print(f"Subject {subject_id} - Block {block} has duplicate trial numbers")                      
                    return 
                # Split random trials into random & non-random segments
                random_data     = block_data[block_data['condition'] == 82].sort_values('b_trial')
                sequence_data   = block_data[block_data['condition'] == 84].sort_values('b_trial')
                
                filtered_rand_data  = {}
                lower_limit_r       = {}
                upper_limit_r       = {}
                if len(random_data) > 0: 
                    half_idx        = len(random_data) // 2 # find midpoint of data
                    
                    # Split random data into pre- and post sequence segments
                    random_split    = {} 
                    random_split[0] = random_data.iloc[:half_idx].reset_index(drop=True) 
                    random_split[1] = random_data.iloc[half_idx:].reset_index(drop=True)
                    
                    #Apply Lowess filter on Random data
                    # Prepare data for LOWESS
                    for dat in enumerate(random_split):
                        rt_array    = random_split[dat[0]]['correct_rt'].values
                        trl_nums    = np.arange(1, len(rt_array) + 1)
                        data_array  = np.column_stack((trl_nums, rt_array))
                        
                        # Apply filter
                        filtered_rand_data[dat[0]], lower_limit_r[dat[0]], upper_limit_r[dat[0]] = smoothing_function(data_array, fit_limit[0], 1)
                                                     
                        # Store results - start with DF and add the subject and block information
                        lowess_df = pd.DataFrame(filtered_rand_data[dat[0]], columns=['trial', 'original_rt', 'smoothed_rt'])
                        lowess_df['subject_id']     = subject_id
                        lowess_df['block']          = block
                        lowess_df['random']         = dat[0]+1 # 0 = all random; 1 = pre-seq random 2 = post-seq random
                        lowess_df['upper_limit']    = upper_limit_r[dat[0]]
                        lowess_df['lower_limit']    = lower_limit_r[dat[0]]
                        lowess_df['error_trials']   = random_split[dat[0]]['error_button'].values
                        lowess_df['error_rt']       = random_split[dat[0]]['error_value'].values
                        lowess_df['method']         = method
                        lowess_df['target']         = random_split[dat[0]]['target'].values
                        
                        # Set MultiIndex to make data more readable
                        lowess_df.set_index(['subject_id', 'block','random'], inplace=True)
                        
                        results['filtered_rts'] = pd.concat([results['filtered_rts'], lowess_df], ignore_index=False)
                        results['filtered_rts'] = results['filtered_rts'].sort_index()
                # Apply filter to sequence data
                seq_len = len(sequence_data) 
                if len(sequence_data) > 0:
                    # Prepare data for LOWESS
                    rt_array = sequence_data['correct_rt'].values
                    trial_nums = np.arange(1, len(rt_array) + 1)
                    data_array = np.column_stack((trial_nums, rt_array))
                    
                    # Apply filter with sequence-specific limit
                    filtered_data, lower_limit, upper_limit = smoothing_function(data_array, fit_limit[1], 1)
                    
                    # Store results - start with DF and add the subject and block information
                    lowess_df = pd.DataFrame(filtered_data, columns=['trial', 'original_rt', 'smoothed_rt'])
                    lowess_df['subject_id']     = subject_id
                    lowess_df['block']          = block
                    lowess_df['random']         = 3 # 0 = all random; 1 = pre-seq random 2 = post-seq random 3 = sequence
                    lowess_df['upper_limit']    = upper_limit
                    lowess_df['lower_limit']    = lower_limit
                    lowess_df['error_trials']   = sequence_data['error_button'].values
                    lowess_df['error_rt']       = sequence_data['error_value'].values
                    lowess_df['method']         = method
                    lowess_df['target']         = sequence_data['target'].values
                    lowess_df['seq_pos']        = np.tile(np.arange(1, 13), seq_len // 12*26)[:seq_len]
                    
                    # Set MultiIndex to make data more readable
                    lowess_df.set_index(['subject_id', 'block','random',], inplace=True)
                    
                    results['filtered_rts'] = pd.concat([results['filtered_rts'], lowess_df], ignore_index=False) 
                    
                # Calculate skill learning
                if (len(sequence_data) >= sl_window and 
                    len(random_split[1]) >= sl_window): # make sure there are enough trials
                    
                    seq_data    = results['filtered_rts'].loc[(subject_id,block,3),:] # sequence data
                    rand_data   = results['filtered_rts'].loc[(subject_id,block,2),:] # post-sequence random data
                    first_block = results['filtered_rts'].loc[(subject_id,1,1),:] # pre-sequence random data
                                 
                    # Calculate skill learning: mean of first SLWindow random trials - mean of last SLWindow sequence trials
                    # Skill learning is normalised based on performance in the first block to account for starting differences
                    rand_mean   = rand_data['smoothed_rt'].iloc[:sl_window].mean()
                    seq_mean    = seq_data['smoothed_rt'].iloc[-sl_window:].mean()
                    norm_mean   = first_block['smoothed_rt'].iloc[:sl_window].mean()
                    
                    # Add to results
                    skill_learning_row = {
                        'subject': subject_id,
                        'session': 1,
                        'block': block,
                        'condition': cond,
                        'skill_learning_value': (rand_mean - seq_mean) * 100,
                        'skill_learning_norm': ((rand_mean - seq_mean)/norm_mean)*100,
                        'method': method
                    }
                        
                    results['skill_learning'] = pd.concat(
                        [results['skill_learning'], pd.DataFrame([skill_learning_row])], 
                        ignore_index=True
                    )    
    return results

# %%
