#%% 
import os 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_WL_data(data1, data2, m_flag):
    figsize = (12, 6)
    plt.style.use('seaborn-v0_8-paper')  # Use a clean style for publication
    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=300)  # Higher DPI for better SVG quality

    # Define colors with transparency for shading
    colors = ['#0000FF', '#FF0000']  # Blue and Red
    alpha_fill = 0.2  # Transparency for the shading

    # Serial Recall Implicit 
    if m_flag == 0:
        # Calculate means and standard errors
        avg_data1 = np.nanmean(data1, axis=0)
        avg_data2 = np.nanmean(data2, axis=0)        
    elif  m_flag ==1:
        avg_data1 = np.nanmedian(data1, axis=0)
        avg_data2 = np.nanmedian(data2, axis=0)
    else: 
        print('Invalid averaging flag! Choose 0 for mean, or 1 for median')
        return
        
    # Calculate standard errors
    se_data1 = np.nanstd(data1, axis=0) / np.sqrt(len(data1))
    se_data2 = np.nanstd(data2, axis=0) / np.sqrt(len(data2))

    # Create x-axis values
    x = np.arange(10)
    
    s1 = len(data1)
    s2 = len(data2)
    # Plot means with standard error bands
    mean1, = axs.plot(x, avg_data1, color=colors[0], linewidth=2, label= f'Congruent ({s1})')
    mean2, = axs.plot(x, avg_data2, color=colors[1], linewidth=2, label=f'Incongruent ({s2})')

    # Add standard error shading
    axs.fill_between(x, 
                     avg_data1 - se_data1, 
                     avg_data1 + se_data1, 
                     color=colors[0], 
                     alpha=alpha_fill)
    axs.fill_between(x, 
                     avg_data2 - se_data2, 
                     avg_data2 + se_data2, 
                     color=colors[1], 
                     alpha=alpha_fill)

    # Customize the plot
    axs.set_ylim([0, 12])
    axs.set_xlim([-0.2, 9.2])  # Slightly extended for better visibility
    axs.set_xticks(x)
    axs.set_xticklabels([f'{int(i+1)}' for i in x])
    axs.set_xlabel('Trial', fontsize=11, labelpad=10)
    axs.set_ylabel('Serial Recall', fontsize=11, labelpad=10)

    # Add grid for better readability
    axs.grid(True, linestyle='--', alpha=0.3)

    # Customize legend
    axs.legend(handles=[mean1, mean2], 
              loc='lower right', 
              frameon=True, 
              framealpha=0.95,
              edgecolor='none')

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Save as SVG
    # plt.savefig('learning_curves.svg', format='svg', bbox_inches='tight')
    # plt.close()
    return fig, axs
  
def find_all_segments(sequence, min_length):
    """
    Find all segments of consecutive numbers in ascending order 
        Parameters:
            sequence: 1D array
                  WL performance in a vector for a given WL block
                min_length: int
                    minimum accepted lenght for a given segment
                    
        Output: 
            segments: array of np.floats
                array containing arrays of each valid segment that fullfills the min_length condition
    """
    segments = []
    current_segment = []
    
    for i in range(len(sequence)):
        if len(current_segment) == 0: # check for first element in segment
            current_segment.append(sequence[i])
        elif sequence[i] == current_segment[-1] + 1: # check if new element follows the last segment or not
            current_segment.append(sequence[i])
        else: # if new element is not old element + 1 start new segment
            if len(current_segment) >= min_length: # ensure that segment fullfills minimum lenght
                segments.append(current_segment)
            current_segment = [sequence[i]]
    
    # Check last segment
    if len(current_segment) >= min_length:
        segments.append(current_segment)
        
    return segments

def calculate_exp_score(segments):
    """Calculate exponential score using log2(sum(2^len)) for segments"""
    if not segments:
        return 0
    
    segment_scores = [2 ** len(seg) for seg in segments]
    return np.log2(sum(segment_scores))

# %%
def process_WL_data(m = 0, min_seq_length=2, plot_flag=0):
    '''
    Calculate average subject wordlist performance for implicit and explicit subjects separately
    
    Parameters: 
        m: int
            type of averaging where m = 0 = mean; m = 1 median
        min_seq_length: int
            Minimum lenght of sequences to include in the analysis (default = 2) 
    '''
    script_dir = Path(__file__).resolve()
    data_path = script_dir.parent.parent.parent.parent / 'Data'

    print(f'Data path: {data_path}')
    
    # load relevant information for behavioral analysis from central csv
    sub_info = pd.read_csv(data_path / 'Subject_Information.csv',encoding='ISO-8859-1')
    rel_info = pd.DataFrame({
        'sub': sub_info['sub'],
        'ID': sub_info['ID'],
        'Exclusion': sub_info['Excluded'],
        'Explicitness': sub_info['Explicitness'],
        'Order': sub_info['ID'] // 1000,  # Extract the order; 1 = Exp first, 2 = Cont first
        'SubID': sub_info['ID'] % 1000    # Extract the corresponding subject IDs
        })

    # make sure to analyse the correct session as the first session
    rel_info['Order'][0]

    sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.startswith('sub-')]
    sub_names   = {} # initialize dict to keep track of all the respective sub names
    sub_conds   = {}
    sub_nums    = {}
    data        = [[None] for _ in range(len(sub_folders))] # initialise variable where data will go in

    for i, sub in enumerate(sub_folders):
        ses_folders = ['ses-1']#,'ses-2'] # only analyse 'ses-1' data   
        sub_number = int(sub.split('-')[1]) # Extract subject number
        try:
            idx = np.where(rel_info['sub'] == sub )[0][0] # find the condition
        except:
            print(f'Subject {sub} not found. Skipping.')
            continue
        
        ses_number = rel_info['Order'][idx] # find the right index to extract condition
    
        if sub_number < 10:# first 9 had a different process with 2 session etc 
            ses = f'ses-{ses_number}'
        else:
            ses = 'ses-1'
        
        beh_path = data_path / sub / ses / 'beh'
        
        if beh_path.exists(): 
            file_name = f'{ses_number}{sub_number:03d}.csv'
            file_path = beh_path / file_name

            # Check if the file exists
            if file_path.exists():
                # Load the content of the .csv file
                temp_data = pd.read_csv(file_path, usecols=range(1, 11), header=None)
                temp_data = temp_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # ensure that there are not trailing spaces that might cause issues
                data[i] = temp_data.to_numpy() # convert the data to a float and making the empty spaces NaN.
                
                sub_names[i]    = file_name[:-4]
                sub_conds[i]    = int(sub_names[i][0])
                sub_nums[i]     = int(sub_names[i][-2:])
            else:
                print(f'No file for subject {sub_number} - Session {ses_number}.')

    #%% Separate Data by explicitness, exclusion, and condition
    sr_con          = []
    sr_incon        = []
    sr_con_exp      = []
    sr_incon_exp    = []
    
    # Create lists to track subject names for each condition
    sub_con         = []
    sub_incon       = []
    sub_con_exp     = []
    sub_incon_exp   = []
    
    sr = {}
    exp_trck = {}
    for i, sub in enumerate(data): # go through processed subjects
        if sub[0] is None:
            continue
                
        exp_idx = rel_info.SubID == sub_nums[i]
        exp_trck[i] = rel_info['Explicitness'][exp_idx] 
        
        if rel_info['Exclusion'][exp_idx].iloc[0] == 1:  # Fixed missing colon and added .iloc[0]
            continue # skip excluded subjects
        
        sr[i] = []    
        for t in range(10): # go through iterations           
            data[i][np.isnan(data[i][:, t]), t] = 0 # select specific Subject WL recall and set NaN to 0
            x = data[i][:, t] # Extract specific wordlist occurance
            
            # Find all valid segments with specified minimum length
            segments = find_all_segments(x[x > 0], min_seq_length)
            exp_score = calculate_exp_score(segments)
            sr[i].append(exp_score)

        # Determine condition and add data and subject name to appropriate lists
        if (sub_conds[i] == 1) & (exp_trck[i].iloc[0] == 0):
            sr_con.append(sr[i])
            sub_con.append(sub_names[i])  
        elif (sub_conds[i] == 2) & (exp_trck[i].iloc[0] == 0):
            sr_incon.append(sr[i])
            sub_incon.append(sub_names[i])  
        elif (sub_conds[i] == 1) & (exp_trck[i].iloc[0] == 1):
            sr_con_exp.append(sr[i])
            sub_con_exp.append(sub_names[i])  
        elif (sub_conds[i] == 2) & (exp_trck[i].iloc[0] == 1):
            sr_incon_exp.append(sr[i])
            sub_incon_exp.append(sub_names[i])  
    
    # %% 
    if plot_flag == 1:
        # Plot 1 
        fig_imp, axs_imp = plot_WL_data(sr_con, sr_incon, m)

        # Plot 2
        fig_exp, axs_exp = plot_WL_data(sr_con_exp, sr_incon_exp, m)
    
        #%% Format title and layout   
        fig_imp.suptitle('Serial Recall Learning Curves - Implicit', y=0.98, fontsize=10, fontweight='bold')
        fig_exp.suptitle('Serial Recall Learning Curves - Explicit', y=0.98, fontsize=10, fontweight='bold')
        
        fig_imp.tight_layout()
        fig_exp.tight_layout()
        
        plt.show()
    
    # Create dictionaries to return data and subject names by condition
    wl_data = {
        'con_imp': sr_con, 
        'incon_imp': sr_incon, 
        'con_exp': sr_con_exp, 
        'incon_exp': sr_incon_exp
    }
    
    sub_names = {
        'con_imp': sub_con, 
        'incon_imp': sub_incon, 
        'con_exp': sub_con_exp, 
        'incon_exp': sub_incon_exp
    }
    
    return wl_data, sub_names  # Return both data and subject names