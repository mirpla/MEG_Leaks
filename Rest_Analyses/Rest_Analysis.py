# %%Rest Analysis
import pandas as pd
import sys
from meg_analysis.Scripts.Rest_Analyses.Source_Data_Rest import make_source_rest
from meg_analysis.Scripts.Rest_Analyses.Source_Analysis_FQ_Blocks_Subs import motor_FFT_analysis
from meg_analysis.Scripts.Rest_Analyses.Source_Analysis_MC_Spectra import plot_source_spectra

#%% see Source_Loc_Cluster.py for running the source localization for all subjects
# Run source localisation for one subjects for the resting state data

subs = 'sub-76'  # 'sub-XX'
ses = 'ses-1'  
#%% Source locl parallel processing
# Go to Source_Loc_Cluster.py for the parallel processing of the source localization on the cluster for proper performance and the latest version of the script
# source loc parameters
src_d = 0.8  # depth
src_l = 0.4  # loose
src_f = False  # Fixed? (True/False)
src_method = 'eLORETA' # inverse method to use for the source localization (e.g., 'dSPM', 'sLORETA','eLORETA')
make_source_rest(subs, ses, src_d, src_l, src_f, src_method, max_subjects=2)

#%% run FFT on source localised data: Source_PSD_HPC.py/
# example parameters  
#python Source_PSD_HPC.py \ 
#   --data_root /analyse/Project0407/Data \ 
#   --save_path /analyse/Project0407/Data/derivatives/spectral_analysis \ 
#   --method multitaper \ 
#   --fmin 1 \ 
#   --fmax 30 \ 
#   --max_parallel_subjects 2

#%% Plot the source localised data
Soure_Plot_FFT
# %% Specific Motor Cortex Analysis --------------------------------------------------
#Create the Motor Cortex data for each block
source_param = {
    'method': 'eLORETA', # inverse method to use for the source localization (e.g., 'dSPM', 'sLORETA','eLORETA')
    'depth': 8,  # depth
    'loose': 3,  # loose
    'fixed': False,  # Fixed? (True/False)
    'snr': 3,  # Signal-to-noise ratio
}
motor_FFT_analysis(source_param = source_param, Condition = ['Congruent', 'Incongruent'], LR = ['left','right'])

#%% Plot the Source Frequency Analysis
ss_flag = 0     # 1 = create and save individual subject spectra
ms_flag = 0     # 1 = create and save group spectra locked to behavior
msul_flag = 1   # 1 = create and save group behaviorally unlocked spectra
a_flag = 0     # 1 = create and save group spectra locked to behavior (unlocked)
plot_source_spectra(ss_flag, ms_flag, msul_flag, a_flag,
                        window_size = 3, 
                        Condition = ['Congruent','Incongruent'], 
                        LR  = ['left','right'],   
                        comp_t = ['periodic'],
                        method = 'eLORETA')
# %%
