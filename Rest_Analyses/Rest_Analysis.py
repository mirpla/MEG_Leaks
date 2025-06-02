# %%Rest Analysis
import pandas as pd
import sys
from meg_analysis.Scripts.Rest_Analyses.Source_Data_Rest import make_source_rest, make_source_parallel
from meg_analysis.Scripts.Rest_Analyses.Source_Analysis_FQ_Blocks_Subs import motor_FFT_analysis
from meg_analysis.Scripts.Rest_Analyses.Source_Analysis_MC_Spectra import plot_source_spectra

#%% 
# Run source localisation for all subjects for the resting state data
data = pd.read_csv('Z:/Data/Subject_Information.csv', encoding='latin1') 
Included = ~data['Excluded'].astype(bool)

subs = data['sub'][Included].tolist()
ses = 'ses-1'  
#%% Source locl parallel processing
# source loc parameters
src_d = 0.8  # depth
src_l = 0.4  # loose
src_f = False  # Fixed? (True/False)
src_method = 'sLORETA' # inverse method to use for the source localization (e.g., 'dSPM', 'sLORETA')
make_source_parallel(subs, ses, src_d, src_l, src_f, src_method, max_subjects=2)

''' For Sequential processing of individual subjects:
for sub in subs:
    make_source_rest(sub, ses, src_d, src_l, src_f, src_method, max_subjects=2)
'''
# %% 
#Create the Motor Cortex data for each block
source_param = {
    'method': 'sLORETA', # inverse method to use for the source localization (e.g., 'dSPM', 'sLORETA')
    'depth': 8,  # depth
    'loose': 4,  # loose
    'fixed': False,  # Fixed? (True/False)
    'snr': 3,  # Signal-to-noise ratio
}
motor_FFT_analysis(source_param = source_param, Condition = ['Congruent', 'Incongruent'], LR = ['left','right'])

#%% Plot the Source Frequency Analysis
ss_flag = 0     # 1 = create and save individual subject spectra
ms_flag = 1     # 1 = create and save group spectra locked to behavior
msul_flag = 1   # 1 = create and save group behaviorally unlocked spectra
a_flag = 0      # 1 = create and save group spectra locked to behavior (unlocked)
plot_source_spectra(ss_flag, ms_flag, msul_flag, a_flag,
                        window_size = 3, 
                        Condition = ['Congruent'], 
                        LR  = ['right'],   
                        comp_t = ['periodic'],
                        method = 'sLORETA')
# %%
