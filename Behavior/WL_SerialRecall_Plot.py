import os 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_WL_Data(data1, data2, m_flag):
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
    
# %% 
script_dir = Path(__file__).resolve()
data_path = script_dir.parent.parent.parent / 'Data'
figsize = (12, 6)

#load wordist items
wl = pd.read_csv(data_path / 'MEG_WL_ITEMS.csv').to_numpy()

# load relevant information for behavioral analysis from central csv
sub_info = pd.read_csv(data_path / 'Subject_Information.csv',encoding='ISO-8859-1')
rel_info = pd.DataFrame({
    'sub': sub_info['sub'],
    'ID': sub_info['ID'],
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
   
    idx = np.where(rel_info['sub'] == sub )[0][0] # find the condition
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

#%% Separate Data by explicitness and condition

sr_con          = []
sr_incon        = []
sr_con_exp      = []
sr_incon_exp    = []
sr = {}
exp_trck = {}
for i, sub in enumerate(data): # go through processed subjects
    exp_idx = rel_info.SubID == sub_nums[i]
    exp_trck[i] = rel_info['Explicitness'][exp_idx] 
    
    sr[i] = []    
    for t in range(10): # go through iterations           
        data[i][np.isnan(data[i][:, t]), t] = 0
        x = data[i][:, t]
        d = np.diff(x) == 1
        indices = np.where(np.diff(np.concatenate(([0], d, [0]))) != 0)[0]
        if len(indices) > 1:
            segment_lengths = indices[1::2] - indices[::2]
            breakpoint()
            sr[i].append(np.max(segment_lengths) + 1 if len(segment_lengths) > 0 else 0)
        else: 
            sr[i].append(0)
    if (sub_conds[i] == 1) & (exp_trck[i].iloc[0] == 0):
        sr_con.append(sr[i])
    elif (sub_conds[i] == 2) & (exp_trck[i].iloc[0] == 0):
        sr_incon.append(sr[i])
    elif (sub_conds[i] == 1) & (exp_trck[i].iloc[0] == 1):
        sr_con_exp.append(sr[i])
    elif (sub_conds[i] == 2) & (exp_trck[i].iloc[0] == 1):
        sr_incon_exp.append(sr[i])
# %% Plot 1 
m = 1 # flag for average type: 0 = mean; 1 = median
fig_imp, axs_imp = plot_WL_Data(sr_con,sr_incon,m)
axs_imp.set_title('Serial Recall Learning Curves - Implicit', pad=20, fontsize=12, fontweight='bold')

# %% Plot 2
fig_exp, axs_exp = plot_WL_Data(sr_con_exp,sr_incon_exp,m)
axs_exp.set_title('Serial Recall Learning Curves - Explicit', pad=20, fontsize=12, fontweight='bold')
