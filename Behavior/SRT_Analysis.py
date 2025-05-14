# %% Import packages
from pathlib import Path

from meg_analysis.Scripts.Behavior.SRT_Performance import srt_import_fit
from meg_analysis.Scripts.Behavior.SRT_Plots import srt_plot_sl, plot_trial_performance, plot_buttons

# %% Run analysis
base_path =  Path('Z:/')
fit_limit = [0.24, 0.14]  # Example filter limits for [random, sequence]
sl_window = 50  # Window size for skill learning calculation
srt_data = srt_import_fit(base_path, fit_limit, sl_window, method='loess', poly_degree=1)
    
# Display results
print("Analysis complete!")

# %% Plot learning rate results, if new window do: %matplotlib qt
norm = True # flag for normalized data
separate_conditions = False # flag for separate conditions or both together
save_path = None # Path to save the figure (if None, the figure is not saved)
figs = srt_plot_sl( srt_data, base_path, norm, save_path, separate_conditions)

# %% Plot individual SRT performances
save = True
sub = [67] # list of subjects to plot (for example [2,4]) (can be None)
y_lim = [0, 1] # y-axis limits
plot_trial_performance(srt_data, base_path, y_lim, sub, save) 

# %% Plot individual buttons 
# plot individual buttons presses for sequence and random trials. If no Path is given plots are output immediately.
sub = [67] # list of subjects to plot (for example [2,4]). If None processes all subjects
use_median  = True # using median or mean  
plot_buttons(srt_data, base_path, sub, use_median)

# %%
