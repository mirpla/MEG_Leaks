# Import necessary Packages
import glob
import mne
import matplotlib
matplotlib.use('Qt5Agg')  

from pathlib import Path
from meg_analysis.Scripts.Preprocessing.Preproc_Functions import extract_sort_key

def check_ICA_comp(sub,ext_ses,rstate):
    # %% Select seed used for ICA
    #rstate = 100
    
    # %% Paths
    script_dir = Path(__file__).resolve() # Location of current scripts
    base_path  = script_dir.parent.parent.parent # Root folder
    data_path  = base_path / 'Data' # Folder containing the Data
      
    #for sub in sub_folders: # No loop, 
    ses_folders = ['ses-1', 'ses-2'] # give options for two sessions
    ses = ses_folders[ext_ses]
    
    
    folder_path = data_path / sub
    
    sub_nr = sub.name.split('-')[1]
        
    
    ses_path = folder_path / ses / 'meg'
            
    ses_nr = ses.split('-')[1]
         
    # set up ICA paths and check for existence of ICA files               
    ica_folder = ses_path / 'ica' 
    ica_file =  ica_folder / f"ica_projsub-{sub_nr}_ses-{ses_nr}_rstate{rstate}.fif"
    
    #if not ica_file.exists():
    #    print (f"Skipping: ICA file of sub{sub} - ses{ses} not found")
    #    continue
    
    # load the downsampeled files again so I can perform ICA 
    downsampled_files = []        
    data_path           = ses_path / 'downsampled'
    data_pattern        = f"sub-{sub_nr}_ses-{ses_nr}_task-*_run-*_meg_tsss_notch-ds-500Hz.fif" 
    data_files          = data_path.glob(data_pattern)
    data_files_sorted   = sorted(data_files, key=extract_sort_key) # sort the files in the folder to have the runs in ascending order and load the SRT files first and only then the WL files        
           
            
    for data_file in data_files_sorted:
        downsampled_files.append(data_file)
                
    data_list        = [mne.io.read_raw_fif(file, preload=True) for file in downsampled_files]
    data_combined    = mne.concatenate_raws(data_list, on_mismatch='warn')
    del data_list
            
    ica = mne.preprocessing.read_ica(ica_file, verbose=None)
       
    ica.plot_sources(data_combined)     
    ica.plot_components(inst=data_combined)
    
            
            
            
                