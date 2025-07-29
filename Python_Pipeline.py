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

from meg_analysis.Tools.Audio_Read import export_meg_audio

#%% Preprocessing --------------------------------------------------------------------
# Import the Raw MEG data and MRI's and perform Tsss and other first run preprocessing 
# MAKE SURE TO RUN AS ADMINISTRATOR!
# give number as string 
sub_codes = ['76'] # 61  still need 1 and 9, 9 has a naming problem
Import_Data(sub_codes)

# %%  
# First run of preprocessing - HP filtering, downsampling and ICA
 
Preprocess_Data()
#Preprocess_Data_Artf()

#%%
subs = ['sub-76'] # 'sub-XX'
coreg_subs(subs)

# %% 
# Look at ICA components and write down which ones to exclude in a separate .csv (Data\ICA_Components.csv)
ses = 0 # select session for checking ICA (ses 1 == 0; ses 2 == 1)
sub = Path('//analyse7/project0407/Data/sub-76') # select the subject folder to look at
rstate = 100 # select the seed 100 is notchfiltered 97 was not 
check_ICA_comp(sub, ses, rstate)

# %% 
# Reject the ICA components and apply a 100 Hz lp filter now that ICA components have been rejected

apply_ICA(rstate, start_sub='76', single_sub=True)
# %%
# Remove Artefacts left over after ICA
# optional parameters: 
#    redo_flag: indicates whether all subjects should be processed/re-processed (1) or not (0)
#    rstate: indicates which file of a specific ICA seed should be used. Different processing runs may have different seeds so this is used to distinguish them (default = 100)
#    start_sub: optional parameter to specify starting subject (format: "XX" where XX is the subject number, e.g., "05")
#    single_sub: if True, only process the specified start_sub. If False, continue processing subsequent subjects (default: False)

Artifacts_Manual(redo_flag=0, rstate=100, start_sub="76", single_sub=False)

#  --------------------------------------------------------------------
# Rest analyses:
script_path = Path('Z:\meg_analysis\Scripts\Preprocessing') # giving the path to the script manually, because the other way keeps defaulting to \\analyse7 instead of Z:
Crop_Rest_Events(script_path,['sub-76'],True)
# 
# Make the Epochs
epoch_dur = 4 # epoch window size in seconds
sessions = ['ses-1'] # give options for two sessions; session 2 not impmlemented yet though
Epoch_Rest(script_path, epoch_dur,None, sessions, True)

#%% Contains Python pipeline analyses for resting state data source reconstruction, analysis and plotting
# meg_analysis.Scripts.Rest_Analyses.Rest_Analysis --- Rest_Analysis

# %% --------------------------------------------------------------------
# MEG SRT analyses:
from pathlib import Path
from meg_analysis.Scripts.Preprocessing.Preproc_SRT import Epoch_SRT

script_path = Path('Z:\meg_analysis\Scripts\Preprocessing') # giving the path to the script manually, because the other way keeps defaulting to \\analyse7 instead of Z:
lock = 'resp' # stimulus locked or response locked processing 'stim' or 'resp'
if lock == 'stim':
    pre  = 0.4
    post = 1
elif lock == 'resp': 
    pre = 0.5   # pre-stimulus time in seconds    
    post = 0.4  # post-stimulus time in seconds
# weird 42 55 64
Epoch_SRT(script_dir = script_path, 
          seed = 100,
          lock = lock, 
          pre_window = pre, 
          post_window = post, 
          bl_size = 0.2,
          sub_folders = None, # None for all subjects or list of subjects  
          overwrite_epochs=False)

# %% Behavioral Analysis -----------------------------------------------
# WL Behavior -------------------------------------------------------
from meg_analysis.Scripts.Behavior.WL_implicitSEplot_Extend import process_WL_data

wl = {}
wl['data'] = []
wl['names'] = []
wl['data'], wl['names'] = process_WL_data(m=1, min_seq_length=2, plot_flag=1)

# %% SRT Behavior 
# for full SRT analysis see: meg_analysis.Scripts.Behavior.SRT_Analysis 

# %% Other --------------------------------------------------------------
# Import empty room data for NCM
script_path = Path('Z:\meg_analysis\Scripts\Preprocessing') # giving the path to the script manually, because the other way keeps defaulting to \\analyse7 instead of Z:
import_ER(script_path, date = None) # import and process the empty room data to allow for NCM for source localisation, input is the data 'YYMMDD'
#%% 
# Events_Fix Careful, running this will rerun all the event files and overwrite the existing ones
#%% Export audio of a specific subject and block to relisten in case of doubts 
meg_file = '//raw/Project0407/nfa05/250527/MEG_2069_WL5.fif'
wav_file = 'C:/Users/mirceav/Desktop/audio.wav'
export_meg_audio(meg_file, wav_file)

# %% Change watershed algorithm parameters to fix intersections 

re_run_BEM('sub-57', pre_flood = 10, scale_factor = 0)
# %%
