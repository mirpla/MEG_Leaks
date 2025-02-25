import mne
import numpy as np
from mne.beamformer import make_lcmv, apply_lcmv
from mne.time_frequency import csd_morlet
from pathlib import Path

def run_lcmv_beamforming(raw_file, base_win=(1, 2), test_win=(2, 3), freq_band=(4, 8), 
                        grid_res=10, out_folder='source_recon'):
    """
    Perform LCMV beamforming on MEG data with parameters matching the SPM implementation.
    
    Parameters
    ----------
    raw_file : str
        Path to the preprocessed MEG data file
    base_win : tuple
        Start and end time for baseline window in seconds
    test_win : tuple
        Start and end time for test window in seconds
    freq_band : tuple
        Lower and upper frequency bounds in Hz
    grid_res : int
        Source space grid resolution in mm
    out_folder : str
        Output directory name
    """
    # Create output directory
    out_path = Path(raw_file).parent / out_folder
    out_path.mkdir(exist_ok=True)
    
    # Load preprocessed data
    raw = mne.io.read_raw(raw_file, preload=True)
    
    # Set up source space (volumetric grid)
    # Using 'vol' for volume source space, similar to SPM's mesh
    sphere = mne.make_sphere_model(r0='auto', head_radius='auto')
    vol_src = mne.setup_volume_source_space(
        sphere=sphere,
        pos=grid_res,  # Grid resolution in mm
        mindist=0,     # No minimum distance constraint
        exclude=10.,   # Exclude points closer than 10mm to the surface
    )
    
    # Compute forward solution using single shell model (like SPM)
    fwd = mne.make_forward_solution(
        raw.info,
        trans='auto',  # Automated co-registration
        src=vol_src,
        bem=sphere,
        meg=True,      # MEG only
        eeg=False
    )
    
    # Filter data to frequency band of interest
    raw_band = raw.copy().filter(freq_band[0], freq_band[1])
    
    # Create epochs for baseline and test windows
    events = mne.make_fixed_length_events(raw_band, duration=base_win[1]-base_win[0])
    epochs = mne.Epochs(raw_band, events, tmin=0, tmax=max(test_win[1], base_win[1]),
                       baseline=None, preload=True)
    
    # Compute cross-spectral density matrices
    csd = csd_morlet(epochs, frequencies=np.mean(freq_band), n_cycles=7, 
                    tmin=min(base_win[0], test_win[0]), 
                    tmax=max(base_win[1], test_win[1]))
    
    # Compute LCMV beamformer filters
    filters = make_lcmv(epochs.info, fwd, csd, reg=0.05,
                       pick_ori='max-power',
                       weight_norm='unit-noise-gain',
                       rank=None)
    
    # Apply beamformer to get source estimates
    base_stc = apply_lcmv_epochs(epochs.crop(*base_win), filters).mean()
    test_stc = apply_lcmv_epochs(epochs.crop(*test_win), filters).mean()
    
    # Compute power difference (test - baseline)
    diff_stc = test_stc - base_stc
    
    # Save results
    diff_stc.save(out_path / 'lcmv_power_diff')
    
    return diff_stc, filters

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    meg_file = "path/to/your/preprocessed_meg_data.fif"
    
    # Run beamforming
    power_diff, filters = run_lcmv_beamforming(
        meg_file,
        base_win=(1, 2),
        test_win=(2, 3),
        freq_band=(4, 8),
        grid_res=10
    )
