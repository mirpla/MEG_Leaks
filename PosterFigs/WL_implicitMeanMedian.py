import os 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



def calculate_se(data):
    """Calculate mean/median and standard error"""
    mean = np.nanmean(data, axis=0)
    median = np.nanmedian(data, axis=0)
    se = np.nanstd(data, axis=0) / np.sqrt(len(data))
    return mean, median, se

# fonts
title_font  = 8
axis_font   = 7
tick_font   = 5
legend_font = 5 
ss_font     = 5
ln_width    = 1.5   
 
#padding
title_pad   = 8
lbl_pad     = 5


script_dir = Path(__file__).resolve()
data_path = script_dir.parent.parent.parent / 'Data'
figsize = (5, 3)  # More vertical orientation

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
            data[i] = pd.read_csv(file_path, usecols=range(1, 11),header=None).to_numpy()
            sub_names[i]    = file_name[:-4]
            sub_conds[i]    = int(sub_names[i][0])
            sub_nums[i]     = int(sub_names[i][-2:])
        else:
            print(f'No file for subject {sub_number} - Session {ses_number}.')

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
            sr[i].append(np.max(segment_lengths) + 1 if len(segment_lengths) > 0 else 0)
        else: 
            sr[i].append(0)
    if (sub_conds[i] == 1) & (int(exp_trck[i]) == 0):
        sr_con.append(sr[i])
    elif (sub_conds[i] == 2) & (int(exp_trck[i]) == 0):
        sr_incon.append(sr[i])
    elif (sub_conds[i] == 1) & (int(exp_trck[i]) == 1):
        sr_con_exp.append(sr[i])
    elif (sub_conds[i] == 2) & (int(exp_trck[i]) == 1):
        sr_incon_exp.append(sr[i])
# %%

# Create figure with two subplots
plt.style.use('seaborn-v0_8-paper')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=300)

# Define colors with transparency for shading
colors = ['#0000FF', '#FF0000']  # Blue and Red
alpha_fill = 0.2  # Transparency for the shading

# titles 
ax1.set_title('Mean Serial Recall', 
              pad=title_pad, fontsize=title_font, fontweight='bold')

# Calculate statistics
mean_sr_con, median_sr_con, se_sr_con = calculate_se(sr_con)
mean_sr_incon, median_sr_incon, se_sr_incon = calculate_se(sr_incon)

x = np.arange(10)

mean1, = ax1.plot(x, mean_sr_con, color=colors[0], linewidth= ln_width, label='Congruent')
mean2, = ax1.plot(x, mean_sr_incon, color=colors[1], linewidth= ln_width, label='Incongruent')

# Add SE shading for means
shade1 = ax1.fill_between(x, 
                mean_sr_con - se_sr_con, 
                mean_sr_con + se_sr_con, 
                color=colors[0], 
                alpha=alpha_fill,
                label='±1 SE')
ax1.fill_between(x, 
                mean_sr_incon - se_sr_incon, 
                mean_sr_incon + se_sr_incon, 
                color=colors[1], 
                alpha=alpha_fill)

# Plot 2: Medians with SE
ax2.set_title('Median Serial Recall', pad=title_pad, fontsize=title_font, fontweight='bold')

median1, = ax2.plot(x, median_sr_con, color=colors[0], linewidth= ln_width, label='Congruent')
median2, = ax2.plot(x, median_sr_incon, color=colors[1], linewidth= ln_width, label='Incongruent')

# Add SE shading for medians
ax2.fill_between(x, 
                median_sr_con - se_sr_con, 
                median_sr_con + se_sr_con, 
                color=colors[0], 
                alpha=alpha_fill,
                label='±1 SE')
ax2.fill_between(x, 
                median_sr_incon - se_sr_incon, 
                median_sr_incon + se_sr_incon, 
                color=colors[1], 
                alpha=alpha_fill)

# Customize both plots
for ax in [ax1, ax2]:
    ax.set_ylim([0, 12])
    ax.set_xlim([-0.2, 9.2])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(i+1)}' for i in x], fontsize= tick_font)
    ax.set_xlabel('Trial', fontsize=axis_font, labelpad=lbl_pad)
    ax.set_ylabel('Serial Recall', fontsize=axis_font, labelpad=lbl_pad)
    ax.grid(True, linestyle='--', alpha=0.3)
    
# legend
lines = [mean1, mean2]
labels = ['Congruent', 'Incongruent']
fig.legend(lines, labels, 
          loc='center', 
          bbox_to_anchor=(0.5, 0.02),
          ncol=3,
          fontsize=8,
          frameon=True,
          framealpha=0.95)
# fig.text(0.5, 0.12, f'n={len(sr_con)} per group', ha='center', fontsize=legend_font)

# Adjust layout
plt.tight_layout()

plt.savefig('C:/Users/mirceav/Desktop/MEGUKIPoster/learning_curves_mean_median_se_compact_shared.svg', format='svg', bbox_inches='tight', dpi=300)
