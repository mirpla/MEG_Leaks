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
                                    s=80, linewidth=2, alpha=0.8)
                
                # Update cumulative trial counter
                cum_trial += len(type_data)
            
            # Set plot title and labels
            axes[i].set_title(f'Block {block}')
            axes[i].set_ylabel('Reaction Time (s)')
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

# %% plot sequence and button specific reaction times for each subject and block
def plot_buttons(srt_data, base_path=None, sub_exp=None, use_median=False):
    """
    Plot detailed analysis of sequence position effects and button-specific reaction times
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results from srt_import_fit function
        - Must contain 'filtered_rts' DataFrame with MultiIndex (subject_id, block, random)
        - Should also reference the original data with button-specific information
    base_path : Path, optional
        Path the Project folder for saving the figures. If None, figures are displayed but not saved
    sub_exp : list or array, optional
        List of subject IDs to include in the analysis. If None, all subjects are included
    use_median : bool, optional
        If True, use median instead of mean for the sequence position plot. Default is False (use mean)
    
    Returns:
    --------
    figs : dict
        Dictionary containing the created figure objects
    """
    # %%
    # Set figure aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    
    # Create a copy of the data to avoid modifying the original
    filtered_rts = srt_data['filtered_rts'].copy().reset_index()
    
    ymax = 1    
    
    # Get unique subjects
    if sub_exp is None:
        subjects = filtered_rts['subject_id'].unique()
    else: 
        subjects = sub_exp
        
    # Create dictionary to store figures
    figs = {}
    
    # Define colors for different buttons
    button_colors = {
        1: 'royalblue',
        2: 'forestgreen', 
        3: 'firebrick',
        4: 'darkorange'
    }
    
    # Define colors for pre/post random
    random_colors = {
        1: 'lightblue',  # Pre-sequence random
        2: 'coral'       # Post-sequence random
    }
    
    # Define which measure to use in plot titles
    plot_title_prefix = 'Median' if use_median else 'Mean'
    
    #%%
    # Iterate through each subject
    for subject in subjects:
        subject_data = filtered_rts[filtered_rts['subject_id'] == subject]
        
        # Get blocks for this subject
        blocks = subject_data['block'].unique()
        
        for block in sorted(blocks):
            #%%
            # Get data for current block
            block_data = subject_data[subject_data['block'] == block]          
            
            # Split data into sequence and random segments
            sequence_data   = block_data[block_data['random'] == 3]
            random_pre      = block_data[block_data['random'] == 1]
            random_post     = block_data[block_data['random'] == 2]
            
            # extract sequence used for this subject
            first_seq = sequence_data.sort_values('trial').head(12)
            
            target_seq = {}
            for _, row in first_seq.iterrows():
                pos = row['seq_pos']
                button = row['target']
                if pos not in target_seq:
                    target_seq[pos] = button
            
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            
            # 1. Plot sequence position analysis
            # -----------------------------------------------------
            # Calculate central tendency and error measure per sequence position
            if use_median:
                # For median, compute median and MAD
                seq_position_stats = sequence_data.groupby('seq_pos')['original_rt'].agg([
                    ('median', 'median'),
                    ('mad', lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))  # MAD with consistency correction
                ]).reset_index()
                # These define which columns to use from seq_position_stats
            else:
                # For mean, compute mean and SEM
                seq_position_stats = sequence_data.groupby('seq_pos')['original_rt'].agg([
                    ('mean', 'mean'),
                    ('sem', lambda x: x.sem() if len(x) > 1 else 0)
                ]).reset_index()
                # These define which columns to use from seq_position_stats
            
            for seq_pos in range(1, 13):
                # Get individual trials for this position
                pos_data = sequence_data[sequence_data['seq_pos'] == seq_pos]
                
                if len(pos_data) > 0:
                    # Determine button color
                    button = target_seq.get(seq_pos, 1)  # Default to button 1 if not found
                    dot_color = button_colors.get(button, 'gray')
                    
                    # Add jitter to x-position for better visualization
                    jitter = (np.random.rand(len(pos_data)) - 0.5) * 0.3
                    
                    # Plot individual dots
                    ax1.scatter(pos_data['seq_pos'] + jitter, pos_data['original_rt'], 
                              s=20, alpha=0.3, color=dot_color, edgecolor='none')
            
            for i, row in seq_position_stats.iterrows():
                seq_pos = row['seq_pos']
                button = target_seq.get(seq_pos, 1)  # Default to button 1 if not found
                dot_color = button_colors.get(button, 'gray')
                
                # Plot the central tendency dot (larger)
                ax1.scatter(seq_pos, row['median' if use_median else 'mean'], s=100, color=dot_color, 
                          edgecolor='black', linewidth=1.5, zorder=5, 
                          label=f"Button {button}" if button not in [b for _, b in target_seq.items()][:i] else "")
                
                # Add error bars if using median
                if use_median:
                    ax1.errorbar(seq_pos, row['median'], yerr=row['mad'], 
                                fmt='none', ecolor='black', capsize=3)
            
            # Connect central tendency dots with black dotted line
            ax1.plot(seq_position_stats['seq_pos'], seq_position_stats['median' if use_median else 'mean'], 
                   'k--', linewidth=1.5, alpha=0.7, zorder=4)
                        
            # Set fixed y-axis range (0-1000ms)
            ax1.set_ylim(0, ymax)
            
            # Mark outliers with triangles
            for seq_pos in range(1, 13):
                # Get outliers for this position
                pos_data = sequence_data[sequence_data['seq_pos'] == seq_pos]
                outliers = pos_data[pos_data['original_rt'] > ymax]
                
                if len(outliers) > 0:
                    # For each outlier, add a triangle at the top of the plot
                    for _, outlier in outliers.iterrows():
                        # Add upward-pointing triangle at the top
                        ax1.scatter(outlier['seq_pos'], ymax-0.011, marker='^', s=100, 
                                  color='red', edgecolor='black', zorder=6)
                        
                        # Add text with the actual value
                        ax1.text(outlier['seq_pos'], ymax-0.06, f' {outlier["original_rt"]}', 
                               va='bottom', ha='center', fontsize=8)
            
            # Set plot title and labels
            ax1.set_title(f'Sequence Position Effect ({plot_title_prefix})')
            ax1.set_xlabel('Position in Sequence (1-12)')
            ax1.set_ylabel('Reaction Time (s)')
            ax1.set_xticks(range(1, 13))
            
            # Add grid
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend (only showing each button once)
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys(), loc='best')
            
            # 2. Plot button-specific reaction times
            # -----------------------------------------------------
            
            # Calculate button-specific statistics for each segment
            button_stats = {}
            
            for segment_name, segment_data in [('Pre-Random', random_pre), 
                                                ('Sequence', sequence_data), 
                                                ('Post-Random', random_post)]:
                button_stats[segment_name] = {}
                
                # Group by target button and calculate stats
                for button in range(1, 5):
                    # Get trials for this button
                    button_trials = segment_data[segment_data['target'] == button]
                    
                    if len(button_trials) > 0:
                        if use_median:
                            central_value = button_trials['original_rt'].median()
                            error_value = 1.4826 * np.median(np.abs(button_trials['original_rt'] - central_value)) if len(button_trials) > 1 else 0
                        else:
                            central_value = button_trials['original_rt'].mean()
                            error_value = button_trials['original_rt'].sem() if len(button_trials) > 1 else 0
                            
                        button_stats[segment_name][button] = {
                            'central_value': central_value,
                            'error': error_value,
                            'count': len(button_trials)
                        }
            
            # Plot button-specific RTs
            bar_width = 0.25
            segment_positions = {
                'Pre-Random': np.array([1, 2, 3, 4]) - bar_width,
                'Sequence': np.array([1, 2, 3, 4]),
                'Post-Random': np.array([1, 2, 3, 4]) + bar_width
            }
            
            # Define hatches for different segments
            segment_hatches = {
                'Pre-Random': '///',     # Diagonal stripes
                'Sequence': '',          # Solid fill (no hatch)
                'Post-Random': '...'     # Dots
            }
            
            # Set fixed y-axis range to match the other plot (0-1000ms)
            ax2.set_ylim(0, ymax)
            
            # Create bar chart
            for segment, positions in segment_positions.items():
                if segment in button_stats:
                    for i, button in enumerate(range(1, 5)):
                        if button in button_stats[segment]:
                            stats = button_stats[segment][button]
                            
                            # Use the same button colors as in the first plot
                            bar_color = button_colors[button]
                            
                            # Plot the bar with appropriate hatch
                            ax2.bar(positions[i], stats['central_value'], width=bar_width, 
                                    color=bar_color, 
                                    hatch=segment_hatches[segment],
                                    alpha=0.7 if segment == 'Sequence' else 0.5,
                                    edgecolor='black',
                                    linewidth=1)
                            
                            # Add error bars
                            ax2.errorbar(positions[i], stats['central_value'], yerr=stats['error'], 
                                        fmt='none', ecolor='black', capsize=5)
            
            # Mark outliers in the bar plot - these are means/medians that exceed the limit
            for segment, positions in segment_positions.items():
                if segment in button_stats:
                    for i, button in enumerate(range(1, 5)):
                        if button in button_stats[segment]:
                            stats = button_stats[segment][button]
                            
                            # Check if the central value is above the limit
                            if stats['central_value'] > ymax:
                                # Add an upward-pointing triangle at the top
                                ax2.scatter(positions[i], ymax, marker='^', s=100, 
                                            color='red', edgecolor='black', zorder=6)
                                
                                # Add text with the actual value
                                ax2.text(positions[i], ymax, f' {stats["central_value"]:.3f}', 
                                        va='bottom', ha='center', fontsize=8)
            
            # Add legend items for segments
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gray', hatch='///', edgecolor='black', alpha=0.5, label='Pre-Random'),
                Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Sequence'),
                Patch(facecolor='gray', hatch='...', edgecolor='black', alpha=0.5, label='Post-Random')
            ]
            
            # Add a second legend for segments (positioned at the top)
            ax2.legend(handles=legend_elements, loc='upper center', 
                        bbox_to_anchor=(0.5, 1.15), ncol=3)
            
            # Set plot title and labels for button plot
            ax2.set_title(f'Button-Specific Reaction Times ({plot_title_prefix})')
            ax2.set_xlabel('Button Number')
            ax2.set_ylabel('Reaction Time (s)')
            ax2.set_xticks(range(1, 5))
            ax2.set_xticklabels(['Button 1', 'Button 2', 'Button 3', 'Button 4'])
                
            # Add legend for button plot
            ax2.legend()
            
            # Add grid
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Set main figure title
            fig.suptitle(f'Subject {subject} - Block {block} Analysis', fontsize=16)
            
            # Adjust layout
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
            
            # Save figure if path is provided
            if base_path is not None:
                save_dir = base_path / 'Data' / f'sub-{subject:02d}' / 'ses-1' / 'beh'
                measure_suffix = 'median' if use_median else 'mean'
                fig.savefig(save_dir / f"subject_{subject}-buttons_block-{block}_{measure_suffix}.png", 
                            dpi=300, bbox_inches='tight')
                #close figures if saving to avoid memory/display issues
                plt.close(fig)
                
            # Store figure in dictionary
            measure_suffix = 'median' if use_median else 'mean'
            figs[f'subject_{subject}_block_{block}_{measure_suffix}'] = fig
    
    #%%
    return figs