#%%
import os
import mne 
import numpy as np
import subprocess
import shutil
import pandas as pd
import glob
import logging
from pathlib import Path
from mne.preprocessing import find_bad_channels_maxwell
import matplotlib.pyplot as plt   

# %% Import Functions
# %% Check the paths and make it if it doesn't exist
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# %% Save Saves content to the specified file path. Ensures that the necessary directories exist.
def save_file_to_path(file_path, content):
    # Extract directory from file path
    directory = os.path.dirname(file_path)
    
    # Ensure directory exists
    check_path(directory)
    
    # Save the content to the file
    with open(file_path, 'w') as file:
        file.write(content)

# %% Runs freesurfer scripts in a virtual linux machine

def run_freesurfer(subject_id, input_file):
   
    # Create the shell script content
    shell_script = f"""#!/bin/bash
# Set FreeSurfer home directory
export FREESURFER_HOME=/usr/local/freesurfer  # Update this path if necessary

# Set license file path (FreeSurfer will look here by default)
export FS_LICENSE=$FREESURFER_HOME/.license

# Source FreeSurfer setup
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Set up the environment
export SUBJECTS_DIR=/mnt/c/fs_data/

# Run FreeSurfer command
recon-all -i /mnt/c/fs_data/{input_file} -s {subject_id} -all

# Run Watershed algorithm
mne watershed_bem -s {subject_id}

# Creat Head Surface
mkheadsurf -s {subject_id}

# Fix permissions for Windows access
chmod -R 755 /mnt/c/fs_data/{subject_id}
find /mnt/c/fs_data/{subject_id} -type f -exec chmod 644 {{}} \;
"""

    # Write the shell script to a file with Unix line endings
    with open('run_freesurfer.sh', 'w', newline='\n') as f:
        f.write(shell_script)

    # Make the shell script executable
    os.chmod('run_freesurfer.sh', 0o755)

    # Run the shell script using WSL
    wsl_command = 'wsl bash -c "/bin/bash ./run_freesurfer.sh"'
    result = subprocess.run(wsl_command, shell=True, capture_output=True, text=True)

    # Print the output and error (if any)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

    # Remove the temporary shell script
    os.remove('run_freesurfer.sh')

# %% Make the BEM from the processed MRIs
def MakeBoundaryElementModel(mri_path,meg_path):    
    log_file = meg_path / 'bem_error.log'
    logging.basicConfig(filename=log_file, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define relevant parameters
    conductivity = (0.3, 0.006, 0.3) 
    
    # set the folder structures I will need 
    local_dir = r'C:\fs_data' # Local folder to with read-write access for Freesurfer
    
    # output paths where all the data should be saved on the drives
    out_path_dum = meg_path / 'anat'
    out_path_dum.mkdir(parents=True, exist_ok=True)
    
    subject_id = str(meg_path.name)
       
    fs_path = Path(local_dir) / subject_id
    
    mri_file_nii = (subject_id + '_T1w.nii.gz')
    mri_file_nii_path = out_path_dum / mri_file_nii 
    
    bem_path = out_path_dum /  'bem'
    
    if not os.path.exists(mri_path):
        print('MRI for ' +  mri_path.name + ' not found! Skipping...')
        return
    
    if not os.path.isfile(mri_file_nii_path):
        print('Converting DCM MRI file for ' + subject_id + ' to NiFTI')
    
        # Construct the dcm2niix command
        command = [
            'dcm2niix',
            '-o', out_path_dum.as_posix(),
            '-f', subject_id + '_T1w',
            '-z', 'y',  # Optional: compress output files using gzip
            mri_path.as_posix()
        ]
    
        # Run the command
        subprocess.run(command, check=True)
    
    if not fs_path.exists():
        os.chdir(local_dir) # make sure the local directory is the current directly for wsl to work
        
        if not os.path.exists(Path(local_dir) / mri_file_nii):
            # copy files from Analyse to C: drive as an intermediary
            shutil.copy(mri_file_nii_path, local_dir)
      
        # Set up the WSL commands to use freesurfer     
        run_freesurfer(subject_id,  (subject_id + '_T1w.nii.gz'))
        
        # Fix permissions after FreeSurfer processing
        wsl_command = f'wsl chmod -R 755 /mnt/c/fs_data/{subject_id}'
        subprocess.run(wsl_command, shell=True, check=True)
        
        # creat folder for BEM model if it doesn't exist already
    bem_path.mkdir(parents = True, exist_ok=True)
              
    #make sure Boundary Element model hasn't been created yet for this Subject
    bem_file = bem_path / (subject_id + '_bem.h5')
    if os.path.isfile(bem_file):
        print(f'BEM file for {subject_id} already exists!')
        return  
        
    try: # BEM model and log if there are issues
        bem_model = mne.make_bem_model(subject=subject_id, ico=4, conductivity=conductivity, subjects_dir=local_dir)
        bem = mne.make_bem_solution(bem_model)
        
        mne.write_bem_solution(bem_file, bem)
        print(f"FreeSurfer recon-all completed for subject {subject_id}")   

    except Exception as e:
        #Log the error with the file path
        logging.error(f"Failed to process {subject_id}: {e}")
      
        print('#################### BEM ERROR FOR SUBJECT ' + subject_id + ' ####################')
        print(f"Error: Logged to {log_file}.")
        
        
    # Set up source space
    src = mne.setup_source_space(
        subject= subject_id,
        subjects_dir = local_dir,
        spacing='oct6', add_dist=False)
    
    src_name = os.path.join(bem_path, subject_id + '-src.fif')
    src.save(src_name,overwrite=True)
     
    # Visualize surfaces - works EVEN IF previous BEM step failed
    plot_bem_kwargs = dict(
        subject=subject_id, subjects_dir=local_dir,
        brain_surfaces='white', orientation='coronal',
        slices=[50,75, 100,125,130, 150,160,175])
    
    
    fig = mne.viz.plot_bem(**plot_bem_kwargs)
    figname = os.path.join(bem_path, subject_id + '_BEM_slices.png')
    
    fig.savefig(figname)
    plt.close('all')    

    
def MEG_extract_destination(first_file):
    raw = mne.io.read_raw_fif(first_file, preload=True,verbose=False)
    # Extract the transformation matrix from the first recording
    dest_mat = raw.info['dev_head_t']
    
    return dest_mat
# %% Copy files from one folder to another (used to run the linux stuff on my local machine)    
def copy_files(source, destination):
    """
    Copies files from the source directory to the destination directory.
    
    Parameters:
    source (Path): Source directory path.
    destination (Path): Destination directory path.
    """
    if not destination.exists():
        destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy(item, destination / item.name)  
    
# %% TSSS and head correction from imported MEG
def MEG_import_process(file, out, destination_mat):      
    if os.path.exists(out):
        return f'File already imported and Maxfiltered for {out}! ----- Skipping'
        
    # Read in data, use hpi information and apply maxwell filter
    root_path = out.parent.parent.parent.parent.parent
    raw_data = mne.io.read_raw_fif(file, preload=True,verbose=False)
    tools_path = root_path / 'meg_analysis' /'Tools' 
   
    file_path = Path(file) # file is a WindowsPath object that has to be converted first to allow for manipulation into an eventual string
    dum_name = file_path.with_suffix('').as_posix() + '-1.fif'
    
    if os.path.isfile(dum_name ): # check whether the file was split and append it if it was
        dum_file = mne.io.read_raw_fif(dum_name, preload=True,verbose=False)
    
        raw_data.append(dum_file)
        del dum_file
        
    # compute continuous hpi coil information
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw_data)
    
    check_cHPI = np.all(np.isnan(np.unique(chpi_amplitudes['slopes']))) # if true, all values are nan, therefore no cHPI used  

    if check_cHPI:
        print('cHPI was off')
        # create a dummy head position file that contains origin information 
        pos_file =  out.with_name(out.stem +'_nocHPI_pos.fif')
        chpi_pos = np.zeros((1,10))
        chpi_pos[:,4:7] = raw_data.info['dev_head_t']['trans'][:3,3] 
        mne.chpi.write_head_pos(pos_file,chpi_pos)

    else:
        print('cHPI was on! extracting head movements during the recording')
        chpi_locs = mne.chpi.compute_chpi_locs(raw_data.info,chpi_amplitudes,verbose=False)
        chpi_pos  = mne.chpi.compute_head_pos(raw_data.info, chpi_locs,verbose=False)
        pos_file = out.with_name(out.stem + '_pos.fif')
        mne.chpi.write_head_pos(pos_file,chpi_pos)

        fig = mne.viz.plot_head_positions(chpi_pos, mode='traces',show = False)
        
        # Ensure the directory exists; if not, create it
        check_fig_dir = out.parent / 'check_fig'
        check_fig_dir.mkdir(parents=True, exist_ok=True)
        figure_path = check_fig_dir / Path(str(out.stem) +  '_head_movement.png')

        fig.savefig(figure_path)    
        plt.close()
    
    raw_data.info['bads'] = []
   
    # Import csv with generally known bad channels
    bads_path = out.parent.parent.parent.parent / 'Bad_Channels.csv'  # Replace with your actual file path
    df = pd.read_csv(bads_path)
    channel_names = df['Bad_Channels'].tolist()

    raw_data.info['bads'] = channel_names

    raw_data_check = raw_data.copy()
    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(raw_data_check, return_scores=True, verbose = True)
    print(auto_noisy_chs)  # we should find them!
    print(auto_flat_chs)  

    bads = list(set(raw_data.info['bads'] + auto_noisy_chs + auto_flat_chs))
    raw_data.info['bads'] = bads

    
    raw_data_sss = mne.preprocessing.maxwell_filter(
       raw_data, 
       st_duration=50, 
       st_correlation=.9, 
       coord_frame='head', 
       destination= destination_mat, 
       head_pos= chpi_pos, 
       verbose=True,
       calibration = tools_path / 'sss_cal.dat',
       cross_talk = tools_path  / 'ct_sparse.fif'
       )
    
    raw_data_sss.save(out)
    
# %% Import and Process the empty room recording

def import_ER(script_dir, date = None):
    '''
    import and process the empty room data to allow for NCM for source localisation, input in the 'YYMMDD' format
    date: str optional
        'YYMMDD' (e.g., '240927' for 27th September 2024)
        If not provided, the function will process all unprocessed dates in the folder
        If date is provided function will proceed to process and overwrite the original if it exists
    '''
    #paths
    base_path   = script_dir.parent.parent.parent 
    data_folder = base_path / 'Data' / 'empty-room'
    tools_path  = base_path / 'meg_analysis' / 'Tools' 
    
    # Check if the date is provided and if the file exists
    files_to_process = []
    if date is not None:
        files_to_process = [data_folder / f'ER_{date}.fif']
        if not  files_to_process[0].exists():
            raise FileNotFoundError(f"File not found: {files_to_process[0]}")      
    else: 
        # Find all ER_*.fif files that are not already processed files (*_raw_sss.fif)
        all_er_files = list(data_folder.glob('ER_*.fif'))
        raw_files = [f for f in all_er_files if '_raw_sss' not in f.name]
        if not raw_files:
            print("No empty room files found to process.")
            return
    
        print(f"Found {len(raw_files)} files to potentially process.")
       
       # remove the files that have a corresponding _raw_sss file already from the list to avoid overwrite
        for file in raw_files:     
            expected_output = file.parent / (file.stem + '_raw_sss.fif')
            if not expected_output.exists():
                    files_to_process.append(file)
        if not files_to_process:
            print("All empty room files have already been processed.")
            return
        print(f"Found {len(files_to_process)} files that need processing.")
        
    for file in files_to_process:
        raw_data    = mne.io.read_raw_fif(file, preload=True,verbose=False)
        out_file    = data_folder / Path(file.stem +'_raw_sss.fif')
    
        raw_data.info['bads'] = []
        
        # Import csv with generally known bad channels
        bads_path = base_path / 'Data' / 'Bad_Channels.csv'  # Replace with your actual file path
        df = pd.read_csv(bads_path)
        channel_names = df['Bad_Channels'].tolist()
        
        raw_data.info['bads'] = channel_names
        
        raw_data_check = raw_data.copy()
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(raw_data_check, return_scores=True, verbose = True,coord_frame='meg') 
        
        bads = list(set(raw_data.info['bads'] + auto_noisy_chs + auto_flat_chs))
        raw_data.info['bads'] = bads
        
        raw_data_sss = mne.preprocessing.maxwell_filter(
        raw_data, 
        st_duration=50, 
        st_correlation=.9, 
        coord_frame='meg', 
        destination= None, 
        verbose=True,
        calibration = tools_path / 'sss_cal.dat',
        cross_talk = tools_path  / 'ct_sparse.fif'
        )
        
        raw_data_sss.resample(500)    # resample data to make the data more manageable             
        raw_data_sss.filter(1, 100)  # filter the data to remove drifts and other possible artefacts. No high-pass filter this this might affect detection of muscle and eye artefacts
        raw_data.notch_filter(np.arange(50, 251, 50))
        
        raw_data_sss.save(out_file, overwrite=True)
    
def coreg_subs(subs):
    script_dir  = Path(__file__).resolve() # Location of current scripts
    base_path   = script_dir.parent.parent.parent.parent
    raw_path    = Path(r'\\raw\Project0407')
    
    ses_name    = 'subject number'
    ses_n_lbl   = 'session number (1 = seq, 2 = control)' 
    hash_id     = 'subject hash'
    ses_dat     = 'session date'
    
    sub_key  = pd.read_csv(base_path / 'Data/Subject_Keys.csv', dtype={0: str, 1: str, 2: str, 3: float, 4: str})  
    
    ses_folders = ['ses-1','ses-2'] # give options for two sessions
    for sub in subs:
        for s, ses in enumerate(ses_folders):            
            ses_path = base_path / 'Data' / sub / ses / 'meg'
            if ses_path.exists(): 
                sub_number  = sub.split('-')[1]
                row_idx     = sub_key[sub_key[ses_name] == str(int(sub_number))]
                row_ses     = row_idx.iloc[s]
                ID          = f'{int(row_ses[ses_n_lbl])}{int(row_ses[ses_name]):03d}'
                
                meg_file    = raw_path / row_ses[hash_id] / row_ses[ses_dat] / f'MEG_{ID}_SRT1.fif'
                out_path    = base_path / 'Data' / sub / ses / 'meg' / f'{sub}_{ses}_run-1_meg_tsss.fif'    

                mne.gui.coregistration(subject=sub, subjects_dir='C:/fs_data/', inst=meg_file, block = True )