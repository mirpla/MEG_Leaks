# %%Rest Analysis
import pandas as pd
import sys
from meg_analysis.Scripts.Rest_Analyses.Source_Data_Rest import make_source_rest
from meg_analysis.Scripts.Rest_Analyses.Source_Analysis_FQ_Blocks_Subs import motor_FFT_analysis
from meg_analysis.Scripts.Rest_Analyses.Source_Analysis_MC_Spectra import plot_source_spectra

#%% 
# Run source localisation for all subjects for the resting state data
data = pd.read_csv('Z:/Data/Subject_Information.csv',encoding='latin1') 
Included = ~data['Excluded'].astype(bool)
subs = data['sub'][Included]
#for sub  in subs:
sub = 'sub-37' # select the subject folder to look at
ses     = 'ses-1'  
# source loc parameters
src_d   = 0.8 # depth
src_l   = 0.2 # loose
src_f   = False # Fixed? (True/False)

# Example usage
make_source_rest(sub, ses, src_d, src_l, src_f)
# %% 
#Create the Motor Cortex data for each block
motor_FFT_analysis(Condition = ['Congruent', 'Incongruent'], LR = ['left','right'])

#%% Plot the Source Frequency Analysis
ss_flag = 0     # 1 = create and save individual subject spectra
ms_flag = 0     # 1 = create and save group spectra locked to behavior
msul_flag = 0   # 1 = create and save group behaviorally unlocked spectra
a_flag = 1      # 1 = create and save group spectra locked to behavior (unlocked)
plot_source_spectra(ss_flag, ms_flag, msul_flag, a_flag,
                        window_size = 3, 
                        Condition = ['Congruent','Incongruent'], 
                        LR  = ['left','right'],   
                        comp_t = ['periodic'])
# %%
