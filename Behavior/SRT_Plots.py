# %% Import packages
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import numpy as np
import pandas as pd

def srt_plot_sl( srt_data, base_path, norm=True, save_path=None, separate_conditions=True):
    """
    Plot skill learning changes across blocks for average data and individual trajectories
    
    Parameters:
    -----------
    srt_data : dict
        Dictionary containing analysis results from srt_import_fit function
        - Must contain 'skill_learning' DataFrame
    norm : bool, optional
        Whether to plot normalised skill learning values or not
    base_path : Path or str
        Path to the base directory of containing subject data
    save_path : Path or str, optional
        Path to save the figures. If None, figures are displayed but not saved
    separate_conditions : bool, optional
        Whether to separate plots by condition (congruent vs incongruent)
        
    Returns:
    --------
    figs : dict
        Dictionary containing the created figure objects
    """
    
    # check the field of interest for normalised or original data
    if norm:
        foi = 'skill_learning_norm'
    else:
        foi = 'skill_learning_value'
    
    sub_info = pd.read_csv(base_path / 'Data' / 'Subject_Information.csv', encoding='latin1')
    sl_data_all = srt_data['skill_learning'].copy()

    # figure style 
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10

    conditions      = sl_data_all['condition'].unique()

    # Define condition names
    condition_names = {
        1: 'Congruent',
        2: 'Incongruent'
    }

    explicit_names = {
        0: 'Implicit',
        1: 'Explicit'
    }

    # Define colors for plotting
    colors = {
        'Congruent': 'royalblue',
        'Incongruent': 'firebrick'
    }

    # Plot settings
    figsize = (10, 6)
    alpha_individual = 0.3
    linewidth_individual = 1
    linewidth_average = 2.5
    marker_style = 'o'
    marker_size = 6

    # %% find subject population of interest (implicit vs explicit) 
    sub_exp_ID = sub_info.loc[(sub_info['Explicitness'] == 0) & (sub_info['Excluded'] == 0), 'ID'].tolist()
    sub_exp     = [id_num % 100 for id_num in sub_exp_ID]

    sl_data = sl_data_all[sl_data_all['subject'].isin(sub_exp)]
    # %% Plotting

    figs = {}
    # %% Create separate plots for each condition
    if separate_conditions:
        for cond in conditions:
            # Filter data for current condition
            cond_data = sl_data[sl_data['condition'] == cond]
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot individual trajectories
            for subject in cond_data['subject'].unique():
                subj_data = cond_data[cond_data['subject'] == subject]
                ax.plot(subj_data['block'], subj_data[foi], 
                        '-', color=colors[condition_names[cond]], alpha=alpha_individual, 
                        linewidth=linewidth_individual)
            
            # Calculate and plot average trajectory
            avg_data = cond_data.groupby('block')[foi].agg(['mean', 'sem']).reset_index()
            ax.errorbar(avg_data['block'], avg_data['mean'], yerr=avg_data['sem'], 
                        color=colors[condition_names[cond]], linewidth=linewidth_average, 
                        marker=marker_style, markersize=marker_size, 
                        label=f'Average ({condition_names[cond]})')
            
            # Set axis labels and title
            ax.set_xlabel('Block')
            ax.set_ylabel('Skill Learning in %')
            ax.set_title(f'Skill Learning Trajectory - {condition_names[cond]} Condition')
            
            # Set x-axis ticks to only show integers
            ax.set_xticks(sorted(cond_data['block'].unique()))
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            fig.tight_layout()
            
            # Save figure if path is provided
            if save_path:
                fig.savefig(f"{save_path}/skill_learning_condition_{cond}.png", dpi=300, bbox_inches='tight')
            
            # Store figure in dictionary
            figs[f'condition_{cond}'] = fig

    # %% Create combined plot with all conditions
    else:
        fig_combined, ax_combined = plt.subplots(figsize=figsize)

        # Plot average trajectories for each condition
        for cond in conditions:
            cond_data = sl_data[sl_data['condition'] == cond]
            avg_data = cond_data.groupby('block')[foi].agg(['mean', 'sem']).reset_index()
            ax_combined.errorbar(avg_data['block'], avg_data['mean'], yerr=avg_data['sem'], 
                                color=colors[condition_names[cond]], linewidth=linewidth_average, 
                                marker=marker_style, markersize=marker_size, 
                                label=f'{condition_names[cond]}')

        # Set axis labels and title
        ax_combined.set_xlabel('Block')
        ax_combined.set_ylabel('Skill Learning in %')
        ax_combined.set_title('Average Skill Learning Trajectory by Condition')

        # Set x-axis ticks to only show integers
        blocks = sorted(sl_data['block'].unique())
        ax_combined.set_xticks(blocks)

        # Add horizontal line at y=0
        ax_combined.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Add legend
        ax_combined.legend()

        # Add grid
        ax_combined.grid(True, linestyle='--', alpha=0.7)

        # Tight layout
        fig_combined.tight_layout()

        # Save figure if path is provided
        if save_path:
            fig_combined.savefig(f"{save_path}/skill_learning_combined.png", dpi=300, bbox_inches='tight')

        # Store figure in dictionary
        figs['combined'] = fig_combined

        #%%  Create additional violin plot to show distribution of skill learning by block and condition
        fig_violin, ax_violin = plt.subplots(figsize=figsize)

        # Prepare data for the violin plot
        # Melt the dataframe to long format for seaborn
        violin_data = sl_data.copy()
        violin_data['condition_name'] = violin_data['condition'].map(condition_names)

        # Create violin plot
        sns.violinplot(x='block', y=foi, hue='condition_name', 
                        data=violin_data, palette=colors, split=True, 
                        inner='quartile', ax=ax_violin)

        # Set axis labels and title
        ax_violin.set_xlabel('Block')
        ax_violin.set_ylabel('Skill Learning in %')
        ax_violin.set_title('Distribution of Skill Learning by Block and Condition')

        # Add horizontal line at y=0
        ax_violin.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Tight layout
        fig_violin.tight_layout()

        # Save figure if path is provided
        if save_path:
            fig_violin.savefig(f"{save_path}/skill_learning_violin.png", dpi=300, bbox_inches='tight')

        # Store figure in dictionary
        figs['violin'] = fig_violin

    return figs
# %%
def plot_trial_performance(srt_data, base_path, y_lim = [0, 1.5], sub_exp=None, save_flag=False):
    """
    Plot individual trial performance from filtered_rts data
    
    Parameters:
    -----------
    srt_data : dict
        Dictionary containing analysis results from srt_import_fit function
        - Must contain 'filtered_rts' DataFrame with MultiIndex (subject_id, block, random)
    y_lim : list, optional
        List of two values specifying the y-axis limits for the plots
    base_path : Path or str
        Path to the base directory of containing subject data
    sub_exp : list or array, optional
        List of subject IDs to include in the analysis. If None, all implicit subjects are included
    save_flag : boolean, optional
        if True, saves the figures to the subject folders

    Returns:
    --------
    figs : dict
        Dictionary containing the created figure objects
    """
    # %%
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    #%% Load data and set figure aesthetics
    sub_info = pd.read_csv(base_path / 'Data' / 'Subject_Information.csv', encoding='latin1')
    
    if sub_exp is None:
        sub_exp_ID  = sub_info.loc[(sub_info['Excluded'] == 0), 'ID'].tolist()
        sub_exp     = [id_num % 100 for id_num in sub_exp_ID]
    
    filtered_rts = srt_data['filtered_rts'].copy().reset_index()
        
    # Set figure aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    
#%% Plot individual trial performance     
    # Filter data for experimental subjects
    filtered_rts = filtered_rts[filtered_rts['subject_id'].isin(sub_exp)]
    if len(filtered_rts) == 0:
        raise ValueError("No data found for the specified subjects")
    
    # Get unique subjects
    subjects = filtered_rts['subject_id'].unique()
    
    # Create dictionary to store figures
    figs = {}
    
    # Define colors for different trial types
    colors = {
        1: 'green',     # Pre-sequence random
        2: 'red',       # Post-sequence random
        3: 'blue'  # Sequence
    }
    
    # Define labels for different trial types
    labels = {
        1: 'Pre-Sequence Random',
        2: 'Post-Sequence Random',
        3: 'Sequence'
    }
    
    trl_types = [1, 3, 2]
    
    # If plot_type is 'individual' or 'both', create individual subject plots
    for subject in subjects:
        # Get data for current subject
        subject_data = filtered_rts[filtered_rts['subject_id'] == subject]
        
        # Get unique blocks for this subject
        blocks = subject_data['block'].unique()
        
        # Create figure with subplots for each block
        fig, axes = plt.subplots(len(blocks), 1, figsize=(12, 4 * len(blocks)), 
                                    sharex=False, sharey=True)
        
        # Handle case where there's only one block
        if len(blocks) == 1:
            axes = [axes]
        
        # Plot each block
        for i, block in enumerate(sorted(blocks)):
            # Get data for current block
            block_data = subject_data[subject_data['block'] == block]
            
            # Create a cumulative trial counter for x-axis
            cum_trial = 0
            trial_offsets = {}
            
            # Plot each trial type; random 1, sequence, random 2
            for random_val in trl_types:
                # Get data for current trial type
                type_data = block_data[block_data['random'] == random_val]
                
                if len(type_data) == 0:
                    continue
                
                # Store the trial offset for this type
                trial_offsets[random_val] = cum_trial
                
                # Create x values with the cumulative offset
                x_vals = type_data['trial'] + cum_trial
                axes[i].set_ylim(y_lim)
                
                # Plot the data points
                axes[i].scatter(x_vals, type_data['original_rt'], 
                                color=colors[random_val], alpha=0.6, 
                                label=f"{labels[random_val]} Trials" if i == 0 else "")
    
                out_of_bounds = type_data['original_rt'] > y_lim[1]
                if out_of_bounds.any():
                    x_out = x_vals[out_of_bounds]
                    y_out = type_data['original_rt'][out_of_bounds]
                    axes[i].scatter(x_out, [y_lim[1]-0.03] * len(x_out), 
                                    marker='^', color=colors[random_val], edgecolors='black', 
                                    s=80, linewidth=1.5, alpha=0.7)
    
                    for x, y in zip(x_out, y_out):
                        axes[i].text(x, y_lim[1]-0.1, f"{y:.2f}", ha='center', va='top', fontsize=9, color=colors[random_val])
                    
                # Plot the smoothed line
                axes[i].plot(x_vals, type_data['smoothed_rt'], 
                            color=colors[random_val], linewidth=2)
                
                # Plot upper and lower filter bounds
                axes[i].fill_between(x_vals, 
                                    type_data['lower_limit'], 
                                    type_data['upper_limit'], 
                                    color=colors[random_val], alpha=0.1)
                
                # Highlight error trials with a red circle
                error_trials = type_data[type_data['error_trials'].notna()]
                if len(error_trials) > 0:
                    axes[i].scatter(error_trials['trial'] + cum_trial, 
                                    error_trials['original_rt'], 
                                    facecolors='none', edgecolors='black', 
                                    s=80, linewidth=1.5, alpha=0.7)
                
                # Update cumulative trial counter
                cum_trial += len(type_data)
            
            # Set plot title and labels
            axes[i].set_title(f'Block {block}')
            axes[i].set_ylabel('Reaction Time (ms)')
            axes[i].set_xlim([0, cum_trial])
            
            # Add grid
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            # Add vertical lines between trial types
            for offset in list(trial_offsets.values())[1:]:
                axes[i].axvline(x=offset + 0.5, color='black', linestyle='--', alpha=0.5)
        
        # Add common x-label
        fig.text(0.5, 0.04, 'Trial Number', ha='center', va='center', fontsize=14)
        
        # Add legend to the first subplot only (to avoid duplicates)
        if len(axes) > 0:
            handles, labels_text = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels_text, loc='upper center', 
                        bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Add main title
        fig.suptitle(f'Subject {subject} - Trial Performance', fontsize=16, y=0.99)
        
        # Save figure if path is provided
        if save_flag:
            save_dir = base_path / 'Data' / f'sub-{subject:02d}' / 'ses-1' / 'beh'
            fig.savefig(save_dir / f"subject_{subject}_trial_performance.png", 
                        dpi=300, bbox_inches='tight')
        
        # Store figure in dictionary
        figs[f'subject_{subject}'] = fig   
    return figs

# %%
