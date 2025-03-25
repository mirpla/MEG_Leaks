# %%
#%load_ext autoreload # allows for reloads of functions without restarting kernel. It's an iPython command so not recognised, error can be ignored
#%autoreload 2
from pathlib import Path
from meg_analysis.Scripts.Import.Import_Data import Import_Data
from meg_analysis.Scripts.Import.fix_MRIs import re_run_BEM
from meg_analysis.Scripts.Import.Import_Preproc import import_ER, coreg_subs
from meg_analysis.Scripts.Preprocessing.Preprocess_Data import Preprocess_Data
#from Scripts.Preprocessing.Preprocess_Data_Artf import Preprocess_Data_Artf
from meg_analysis.Scripts.Preprocessing.ICA_Manual import check_ICA_comp
from meg_analysis.Scripts.Preprocessing.Post_ICA import apply_ICA, Artifacts_Manual
from meg_analysis.Scripts.Preprocessing.RedoEvents import Events_Fix
from meg_analysis.Scripts.Preprocessing.Preproc_Rest import  Crop_Rest_Events, Epoch_Rest
from meg_analysis.Scripts.Preprocessing.Preproc_Functions import read_events

from meg_analysis.Scripts.Behavior.WL_implicitSEplot_Extend import process_WL_data

from meg_analysis.Tools.Audio_Read import export_meg_audio

#%% 
# Import the Raw MEG data and MRI's and perform Tsss and other first run preprocessing 

# MAKE SURE TO RUN AS ADMINISTRATOR!
# give number as string 
sub_codes = ['56'] # 61  still need 1 and 9, 9 has a naming problem
Import_Data(sub_codes)

#%%
subs = ['sub-56'] # 'sub-XX'
coreg_subs(subs)

# %% 
# First run of preprocessing - HP filtering, downsampling and ICA
 
Preprocess_Data()
#Preprocess_Data_Artf()

# %% 
# Look at ICA components and write down which ones to exclude in a separate .csv (Data\ICA_Components.csv)
ses = 0 # select session for checking ICA (ses 1 == 0; ses 2 == 1)
sub = Path('//analyse7/project0407/Data/sub-63') # select the subject folder to look at
rstate = 100 # select the seed 100 is notchfiltered 97 was not 
check_ICA_comp(sub, ses, rstate)

# %% 
# Reject the ICA components and apply a 100 Hz lp filter now that ICA components have been rejected

apply_ICA(rstate)
# %% 
# Remove Artefacts left over after ICA
# optional parameters: 
#    redo_flag: indicates whether all subjects should be processed/re-processed (1) or not (0)
#    rstate: indicates which file of a specific ICA seed should be used. Different processing runs may have different seeds so this is used to distinguish them (default = 100)
#    start_sub: optional parameter to specify starting subject (format: "XX" where XX is the subject number, e.g., "05")
#    single_sub: if True, only process the specified start_sub. If False, continue processing subsequent subjects (default: False)

Artifacts_Manual(redo_flag=1, rstate=100, start_sub="63", single_sub=True)

# %% 
# Rest analyses:
Crop_Rest_Events(['sub-25','sub-29','sub-30','sub-31','sub-32','sub-33','sub-35','sub-36'])

# Make the Epochs
manual_rej = 1 # 0 = take previous artefacts; 1 = do artf rejection manually
epoch_dur = 4 # epoch window size in seconds
sessions = ['ses-1'] # give options for two sessions
Epoch_Rest(manual_rej,epoch_dur, sessions)

# %% Behavior
fig_imp, fig_exp = process_WL_data(m=0, min_seq_length=2)

# %% Other
import_ER() # import and process the empty room data to allow for NCM for source localisation, input is the data 'YYMMDD'
# Events_Fix Careful, running this will rerun all the event files and overwrite the existing ones
#%% Export audio of a specific subject and block to relisten in case of doubts 
meg_file = '//raw/Project0407/eir06/250303/MEG_2058_WL2.fif'
wav_file = 'C:/Users/mirceav/Desktop/audio.wav'
export_meg_audio(meg_file, wav_file)

# %% Change watershed algorithm parameters to fix intersections 

re_run_BEM('sub-56', pre_flood = 10, scale_factor = 0)