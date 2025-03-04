# %%
import mne 
import shutil
import numpy as np
import os
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt  

#%% Watershed WSL function
def run_watershed_alt(subject_id, pf):
    '''
    Run watershed algorithm with different parameters for problematic subjects. 
    Change pre-flood and use atlases 
    '''
    
    shell_script = f"""#!/bin/bash
# Set FreeSurfer home directory
export FREESURFER_HOME=/usr/local/freesurfer
export FS_LICENSE=$FREESURFER_HOME/.license
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/mnt/c/fs_data/

# Run improved Watershed with better parameters
mne watershed_bem -s {subject_id} --preflood {pf} --atlas --gcaatlas --overwrite

# Creat Head Surface
mkheadsurf -s {subject_id}

# Fix permissions
chmod -R 755 /mnt/c/fs_data/{subject_id}
find /mnt/c/fs_data/{subject_id} -type f -exec chmod 644 {{}} \;
"""

    # Choose ONE approach - either write to current directory or C:/fs_data/
    # Option 1: Write to C:/fs_data/ (recommended since your data is there)
    script_path = 'C:/fs_data/run_watershed.sh'
    with open(script_path, 'w', newline='\n') as f:
        f.write(shell_script)
    os.chmod(script_path, 0o755)
    wsl_command = 'wsl bash -c "/bin/bash /mnt/c/fs_data/run_watershed.sh"'
    
    # Execute the command
    result = subprocess.run(wsl_command, shell=True, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

    # Clean up
    os.remove(script_path)
# %% 
def re_run_BEM(subject_id, pre_flood = 10, scale_factor = 0):
    ''' Re-run watershed model for problematic MRIs (for example intersections) 
            by enabling atlas use use pre-flood paramters
            if that doesn't work, one could consider altering conductivities
            requires problematic MRIs to be placed in 'Hospital' folder (this is done to avoid accidental rerunning of previous correct BEMs)

        input paramteres: 
            subject_id      = necessary; in the form of sub-xx where xx is the number of the subject
            pre_flood       = set aggressiveness of pre-flood parameter for watershed algorithm (usually between 0 and 20)
            scale_factor    = optional; factor for scaling, if 0 no surface scaling is performed; 0.98 for example would mean a reduction in 2% of the innter skull towards the centroid

    '''
    # Define relevant parameters
    conductivity = (0.3, 0.006, 0.3) 

    # set the folder structures I will need 
    local_dir = r'C:/fs_data' # Local folder to with read-write access for Freesurfer

    # path containing MRI's in question
    data_path = Path('Z:/Hospital/')

    # Freesurfer Location
    fs_path = Path(local_dir) / subject_id

    # MRI path and name
    mri_file_nii    = (subject_id + '_T1w.nii.gz')
    mri_path        = data_path / mri_file_nii 

    # boundary element model path
    bem_path = data_path / subject_id /  'bem'
    
    # Check folders and files 
    if not os.path.exists(mri_path):
        print('MRI for ' +  mri_path.name + ' not found! Skipping...')

        
    # creat folder for BEM model if it doesn't exist already
    bem_path.mkdir(parents = True, exist_ok=True)
                
    # make sure Boundary Element model hasn't been created yet for this Subject. If it exists suggest to delete and redo
    bem_file = bem_path / (subject_id + '_bem.h5')
    if os.path.isfile(bem_file):
        print(f'BEM file for {subject_id} already exists!')
        user_input = input("Do you want to delete the existing BEM files and continue? (yes/no): ")
        if user_input.lower() == 'yes':
            # Delete all contents of the BEM folder
            for file in bem_path.glob('*'):
                try:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(file)
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")
            print("All contents of the BEM folder have been deleted.")
        else:
            print("Operation aborted by the user.")
            exit()

    #  Redo the watershed model
    run_watershed_alt(subject_id, pre_flood)

####################### Apply Scaling (Optional)
    if not scale_factor == 0:
        # Read the inner skull surface
        inner_skull_file = os.path.join(fs_path, 'bem', f'inner_skull.surf')
        points, faces = mne.read_surface(inner_skull_file)

        # Calculate centroid
        centroid = np.mean(points, axis=0)

        # Shrink the inner skull surface by scaling it toward the centroid
        points_scaled = centroid + (points - centroid) * scale_factor

        # Write the modified surface
        mne.write_surface(inner_skull_file, points_scaled, faces, overwrite=True)

####################################    
    # Set up source space
    src = mne.setup_source_space(
        subject= subject_id,
        subjects_dir = local_dir,
        spacing='oct6', add_dist=False)

    src_name = os.path.join(bem_path, subject_id + '-src.fif')
    src.save(src_name,overwrite=True)
        
    # Visualize surfaces - works EVEN IF previous BEM step failed and save them for easy viewing later
    plot_bem_kwargs = dict(
        subject=subject_id, subjects_dir=local_dir,
        brain_surfaces='white', orientation='coronal',
        slices=[50,75, 100,125,130, 150,160,175])


    fig = mne.viz.plot_bem(**plot_bem_kwargs)
    figname = os.path.join(bem_path, subject_id + '_BEM_slices.png')

    fig.savefig(figname)
    plt.close('all')    

    try: # verify model
    # Load the BEM surfaces and check that there are no intersections
        bem_model = mne.make_bem_model(subject_id, subjects_dir=local_dir, ico = 4, conductivity=conductivity)
        bem_solution = mne.make_bem_solution(bem_model)

        mne.write_bem_solution(bem_file, bem_solution)
        print(f"Successfully created BEM for subject {subject_id}") 
    except Exception as e:
        print('#################### BEM ERROR FOR SUBJECT ' + subject_id + ' ####################')
        print(e)

# %%
