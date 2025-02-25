import subprocess
from pathlib import Path
import os

def windows_to_wsl_path(windows_path):
    """
    Converts a Windows path to a WSL path.
    
    Parameters:
    windows_path (str): The Windows path to convert.
    
    Returns:
    str: The converted WSL path.
    """
    return windows_path.replace('\\', '/').replace('Z:', '/mnt/z')

def run_recon_all(subject_id, mri_path):
    """
    Runs FreeSurfer's recon-all command to process MRI data and stores output in a specified folder.
    
    Parameters:
    subject_id (str): Subject identifier.
    mri_path (Path): Path to the NIfTI file.
    """
    # Convert Windows paths to WSL paths
    mri_path_wsl = windows_to_wsl_path(str(mri_path))
    subject_dir_wsl = windows_to_wsl_path(str(mri_path.parent))

    # Set working directory to a local path
    local_working_dir = Path(r'C:/Users/mirceav/Desktop/Study Files/MEG/Python')  # Ensure this path exists on your machine
    os.chdir(local_working_dir)

    # Print the paths for debugging
    print(f"WSL MRI Path: {mri_path_wsl}")
    print(f"WSL Subject Directory: {subject_dir_wsl}")

    # Command to print current directory and list files for debugging
    debug_command = 'bash -c "mount && ls -la /mnt/z/Data/sub-13/anat"'

    try:
        # Run the debug command to check the mount points and list files
        debug_result = subprocess.run(debug_command, shell=True, check=True, capture_output=True, text=True)
        print("Debug Command Output:")
        print(debug_result.stdout)
        print(debug_result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Debug command failed with error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

    setup_env_command = "export FREESURFER_HOME=/usr/local/freesurfer && source $FREESURFER_HOME/SetUpFreeSurfer.sh"
    set_subjects_dir_command = f'export SUBJECTS_DIR={subject_dir_wsl}'
    recon_all_command = f'recon-all -i {mri_path_wsl} -s fs -all'
    full_command = f'bash -c "{setup_env_command} && {set_subjects_dir_command} && {recon_all_command}"'

    # Print the full command for debugging
    print(f"Full Command: {full_command}")

    try:
        # Run the command and capture the output for debugging
        result = subprocess.run(full_command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

        print(f"FreeSurfer recon-all completed for subject {subject_id}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")


# Example usage
subject_id = 'sub-13'
mri_path = Path(r'Z:\Data\sub-13\anat\sub-13_T1w.nii.gz')  # This should be the NIfTI file path on the mapped drive

run_recon_all(subject_id, mri_path)