#!/usr/bin/env python3
"""
High-memory optimized MEG PSD analysis with aggressive parallelization
For HPC systems with abundant RAM (120GB+)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import mne
import h5py
from mne.time_frequency import psd_array_multitaper, psd_array_welch
from datetime import datetime
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
from meg_analysis.Scripts.Rest_Analyses.Source_Class import RestSourceData

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RestSourcePSDData(RestSourceData):
    """Extended class to handle MEG source PSD data storage and retrieval."""
    
    def initialize_psd_file(self, fname, freqs, source_file_path, sfreq, 
                           fmin=1, fmax=30, method='multitaper', 
                           store_vertices=False):
        """Initialize HDF5 file for PSD data storage."""
        with h5py.File(fname, 'w') as f:
            # Store global metadata
            f.attrs['subject_id'] = str(self.subject_id)
            f.attrs['session_id'] = str(self.session_id)
            f.attrs['creation_date'] = str(datetime.now().isoformat())
            f.attrs['mne_version'] = str(mne.__version__)
            f.attrs['data_type'] = 'source_psd'
            f.attrs['psd_method'] = str(method)
            f.attrs['fmin'] = float(fmin)
            f.attrs['fmax'] = float(fmax)
            f.attrs['sfreq'] = float(sfreq)
            f.attrs['source_file_path'] = str(source_file_path)
            
            # Create groups
            f.create_group('frequencies')
            f.create_group('psd_blocks')
            
            # Store frequency information
            f['frequencies'].create_dataset('freqs', data=freqs)
            f['frequencies'].attrs['n_freqs'] = len(freqs)
            
            # Optionally store source space info
            if store_vertices:
                with h5py.File(source_file_path, 'r') as src_f:
                    vertices_lh = src_f['source_space/vertices_lh'][:]
                    vertices_rh = src_f['source_space/vertices_rh'][:]
                
                f.create_group('source_space')
                f['source_space'].create_dataset('vertices_lh', data=vertices_lh)
                f['source_space'].create_dataset('vertices_rh', data=vertices_rh)
                f['source_space'].attrs['n_sources'] = len(vertices_lh) + len(vertices_rh)
                f['source_space'].attrs['stored_locally'] = True
            else:
                f.create_group('source_space')
                f['source_space'].attrs['stored_locally'] = False
                f['source_space'].attrs['source_file_reference'] = str(source_file_path)
    
    def add_psd_block(self, fname, block_id, power_avg):
        """Add averaged PSD data for a specific block."""
        with h5py.File(fname, 'a') as f:
            block_name = f'block_{block_id:02d}'
            
            if block_name in f['psd_blocks']:
                logger.warning(f"PSD block {block_name} already exists. Skipping...")
                return
            
            block_group = f['psd_blocks'].create_group(block_name)
            block_group.create_dataset(
                'power_avg',
                data=power_avg,
                dtype=np.float32,
                compression='gzip',
                compression_opts=9
            )
            
            block_group.attrs['n_sources'] = power_avg.shape[0]
            block_group.attrs['n_freqs'] = power_avg.shape[1]
            block_group.attrs['block_id'] = block_id

def process_single_block_psd(args):
    """
    Process a single block for PSD computation.
    Designed to be called in parallel.
    """
    source_file_path, block_id, sfreq, fmin, fmax, method, n_jobs_psd = args
    
    try:
        # Parse subject info
        filename = Path(source_file_path).stem
        parts = filename.split('_')
        subject_id = parts[0]
        session_id = parts[1]
        
        # Load block data
        source_handler = RestSourceData(subject_id, session_id)
        block_data, _, _, _ = source_handler.load_block_data(source_file_path, block_id)
        
        # Compute PSD
        if method == 'multitaper':
            psds, freqs = psd_array_multitaper(
                block_data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_jobs=n_jobs_psd)
        elif method == 'welch':
            psds, freqs = psd_array_welch(
                block_data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_jobs=n_jobs_psd)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Average across epochs
        power_avg = np.mean(psds, axis=0)
        
        return {
            'block_id': block_id,
            'power_avg': power_avg,
            'freqs': freqs,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'block_id': block_id,
            'power_avg': None,
            'freqs': None,
            'success': False,
            'error': str(e)
        }

def process_subject_parallel_blocks(source_file_path, save_path=None, fmin=1, fmax=30, 
                                  method='multitaper', n_jobs_psd=2, max_parallel_blocks=8,
                                  store_vertices=False, overwrite=False):
    """
    Process all blocks for a single subject with block-level parallelization.
    
    Parameters
    ----------
    max_parallel_blocks : int
        Maximum number of blocks to process simultaneously
    n_jobs_psd : int
        Number of cores per PSD computation (keep lower when processing multiple blocks)
    overwrite : bool
        Whether to overwrite existing complete PSD files
    """
    try:
        source_file_path = Path(source_file_path)
        filename = source_file_path.stem
        parts = filename.split('_')
        subject_id = parts[0]
        session_id = parts[1]
        
        logger.info(f"Processing {subject_id} {session_id} with parallel blocks")
        
        # Initialize handlers
        source_handler = RestSourceData(subject_id, session_id)
        psd_handler = RestSourcePSDData(subject_id, session_id)
        
        # Get available blocks
        available_blocks = source_handler.get_available_blocks(source_file_path)
        logger.info(f"Found {len(available_blocks)} blocks: {available_blocks}")
        
        if not available_blocks:
            logger.error(f"No blocks found in {source_file_path}")
            return None, False
        
        # Set up save path
        if save_path is None:
            save_path = source_file_path.parent / "derivatives" / "spectral_analysis"
        else:
            save_path = Path(save_path)
            
        bids_output_dir = save_path / subject_id / session_id / "meg"
        bids_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create PSD filename
        method_suffix = f"psd-{method}-{fmin}-{fmax}Hz"
        psd_filename = filename.replace('src_rest-all', f'src_rest-all_{method_suffix}') + '.h5'
        psd_file_path = bids_output_dir / psd_filename
        
        # Check if file already exists and is complete
        if psd_file_path.exists():
            try:
                existing_blocks = psd_handler.get_available_psd_blocks(psd_file_path)
                if set(existing_blocks) == set(available_blocks):
                    if not overwrite:
                        logger.info(f"PSD file already complete for {subject_id}. Use --overwrite to reprocess. Skipping.")
                        return psd_file_path, True
                    else:
                        logger.info(f"PSD file exists for {subject_id}. Overwriting due to --overwrite flag.")
                        psd_file_path.unlink()  # Delete existing file
                else:
                    logger.info(f"Incomplete PSD file found for {subject_id} ({len(existing_blocks)}/{len(available_blocks)} blocks). Reprocessing.")
                    if not overwrite:
                        psd_file_path.unlink()  # Delete incomplete file
            except Exception as e:
                logger.info(f"Corrupted PSD file found for {subject_id}: {str(e)}. Reprocessing.")
                if psd_file_path.exists():
                    psd_file_path.unlink()  # Delete corrupted file
        
        # Get metadata from first block
        block_data, sfreq, vertices_lh, vertices_rh = source_handler.load_block_data(
            source_file_path, available_blocks[0])
        
        logger.info(f"Starting parallel block processing for {subject_id}")
        start_time = time.time()
        
        # Prepare arguments for parallel processing
        block_args = [
            (source_file_path, block_id, sfreq, fmin, fmax, method, n_jobs_psd)
            for block_id in available_blocks
        ]
        
        # Process blocks in parallel
        results = {}
        freqs = None
        
        with ProcessPoolExecutor(max_workers=max_parallel_blocks) as executor:
            # Submit all block jobs
            future_to_block = {
                executor.submit(process_single_block_psd, args): args[1] 
                for args in block_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_block):
                block_id = future_to_block[future]
                try:
                    result = future.result()
                    if result['success']:
                        results[block_id] = result
                        if freqs is None:
                            freqs = result['freqs']  # Get freqs from first successful block
                        logger.info(f"✓ Completed block {block_id}")
                    else:
                        logger.error(f"✗ Failed block {block_id}: {result['error']}")
                except Exception as e:
                    logger.error(f"✗ Failed block {block_id}: {str(e)}")
        
        if not results:
            logger.error(f"No blocks processed successfully for {subject_id}")
            return None, False
        
        # Initialize PSD file with results from first successful block
        psd_handler.initialize_psd_file(
            psd_file_path, freqs, source_file_path, sfreq, fmin, fmax, method, store_vertices)
        
        # Add all successful blocks to file
        successful_blocks = 0
        for block_id in sorted(results.keys()):
            psd_handler.add_psd_block(psd_file_path, block_id, results[block_id]['power_avg'])
            successful_blocks += 1
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Completed {subject_id}: {successful_blocks}/{len(available_blocks)} blocks in {elapsed:.1f}s")
        
        return psd_file_path, True
        
    except Exception as e:
        logger.error(f"Error processing {source_file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, False

def process_multiple_subjects_parallel(source_files, save_path=None, fmin=1, fmax=30,
                                     method='multitaper', n_jobs_psd=2, 
                                     max_parallel_blocks=4, max_parallel_subjects=4,
                                     store_vertices=False, overwrite=False):
    """
    Process multiple subjects in parallel.
    
    Parameters
    ----------
    max_parallel_subjects : int
        Maximum number of subjects to process simultaneously
    max_parallel_blocks : int
        Maximum number of blocks per subject to process simultaneously
    n_jobs_psd : int
        Cores per PSD computation (keep low when using heavy parallelization)
    overwrite : bool
        Whether to overwrite existing complete PSD files
    """
    logger.info(f"Processing {len(source_files)} subjects with {max_parallel_subjects} parallel subjects")
    logger.info(f"Each subject uses {max_parallel_blocks} parallel blocks with {n_jobs_psd} cores per PSD")
    
    success_count = 0
    total_count = len(source_files)
    start_time = time.time()
    
    # Prepare arguments for each subject
    subject_args = [
        (source_file, save_path, fmin, fmax, method, n_jobs_psd, max_parallel_blocks, store_vertices, overwrite)
        for source_file in source_files
    ]
    
    # Process subjects in parallel
    with ProcessPoolExecutor(max_workers=max_parallel_subjects) as executor:
        # Submit all subject jobs
        future_to_file = {
            executor.submit(process_subject_parallel_blocks, *args): args[0]
            for args in subject_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            source_file = future_to_file[future]
            try:
                psd_file, success = future.result()
                if success:
                    success_count += 1
                    logger.info(f"✓ Subject completed: {source_file.name}")
                else:
                    logger.error(f"✗ Subject failed: {source_file.name}")
            except Exception as e:
                logger.error(f"✗ Subject failed: {source_file.name} - {str(e)}")
    
    elapsed = time.time() - start_time
    logger.info(f"Processing complete: {success_count}/{total_count} successful in {elapsed:.1f}s")
    
    return success_count == total_count

def main():
    """Main function with optimized parallelization options."""
    parser = argparse.ArgumentParser(description='High-performance MEG source PSD analysis')
    parser.add_argument('--source_file', type=str, help='Single source file to process')
    parser.add_argument('--data_root', type=str, help='Root directory containing source files')
    parser.add_argument('--subject_list', type=str, help='File containing list of source files')
    parser.add_argument('--save_path', type=str, help='Directory to save results; default is derivatives/spectral_analysis')
    parser.add_argument('--fmin', type=float, default=1, help='Minimum frequency')
    parser.add_argument('--fmax', type=float, default=30, help='Maximum frequency')
    parser.add_argument('--method', type=str, default='multitaper', choices=['multitaper', 'welch'])
    parser.add_argument('--store_vertices', action='store_true', 
                        help='Store vertices locally (default: reference original file)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing complete PSD files (default: skip existing)')
    
    # Parallelization options
    parser.add_argument('--n_jobs_psd', type=int, default=2, 
                        help='CPU cores per PSD computation (reduce when using parallel blocks)')
    parser.add_argument('--max_parallel_blocks', type=int, default=8,
                        help='Maximum blocks to process simultaneously per subject')
    parser.add_argument('--max_parallel_subjects', type=int, default=4,
                        help='Maximum subjects to process simultaneously')
    parser.add_argument('--parallel_mode', type=str, default='blocks', 
                        choices=['blocks', 'subjects', 'both'],
                        help='Parallelization strategy')
    
    args = parser.parse_args()
    
    # Auto-detect optimal settings based on available cores
    total_cores = cpu_count()
    logger.info(f"Detected {total_cores} CPU cores")
    
    if args.parallel_mode == 'both':
        logger.info("Using aggressive parallelization: subjects + blocks + PSD cores")
    elif args.parallel_mode == 'subjects':
        logger.info("Using subject-level parallelization")
        args.max_parallel_blocks = 1  # Sequential blocks per subject
    else:  # blocks
        logger.info("Using block-level parallelization")
        args.max_parallel_subjects = 1  # Sequential subjects
    
    # Determine which files to process
    if args.source_file:
        source_files = [Path(args.source_file)]
    elif args.data_root:
        pattern = "*ses-*_src_rest-all_*.h5"
        source_files = list(Path(args.data_root).rglob(pattern))
        logger.info(f"Found {len(source_files)} source files")
    elif args.subject_list:
        with open(args.subject_list, 'r') as f:
            source_files = [Path(line.strip()) for line in f if line.strip()]
    else:
        logger.error("Must specify --source_file, --data_root, or --subject_list")
        sys.exit(1)
    
    # Filter existing files and check for existing PSD files
    valid_source_files = []
    skipped_complete = 0
    
    for source_file in source_files:
        if not source_file.exists():
            logger.warning(f"Source file not found: {source_file}")
            continue
            
        # Check if PSD file already exists
        if not args.overwrite:
            # Generate expected PSD filename
            filename = source_file.stem
            parts = filename.split('_')
            subject_id = parts[0]
            session_id = parts[1]

            if args.save_path:
                bids_output_dir = Path(args.save_path) / subject_id / session_id / "meg"
            else:
                bids_output_dir = source_file.parent / "derivatives" / "spectral_analysis" / subject_id / session_id / "meg"

            method_suffix = f"psd-{args.method}-{args.fmin}-{args.fmax}Hz"
            psd_filename = filename.replace('src_rest-all', f'src_rest-all_{method_suffix}') + '.h5'
            psd_file_path = bids_output_dir / psd_filename
            
            if psd_file_path.exists():
                # Quick check if file is complete
                try:
                    parts = filename.split('_')
                    subject_id = parts[0]
                    session_id = parts[1]
                    
                    source_handler = RestSourceData(subject_id, session_id)
                    psd_handler = RestSourcePSDData(subject_id, session_id)
                    
                    available_blocks = source_handler.get_available_blocks(source_file)
                    existing_blocks = psd_handler.get_available_psd_blocks(psd_file_path)
                    
                    if set(existing_blocks) == set(available_blocks):
                        logger.info(f"Complete PSD file exists for {subject_id}. Skipping. Use --overwrite to reprocess.")
                        skipped_complete += 1
                        continue
                    else:
                        logger.info(f"Incomplete PSD file exists for {subject_id} ({len(existing_blocks)}/{len(available_blocks)} blocks). Will reprocess.")
                except Exception as e:
                    logger.info(f"Cannot validate existing PSD file for {source_file.name}: {str(e)}. Will reprocess.")
        
        valid_source_files.append(source_file)
    
    source_files = valid_source_files
    logger.info(f"Processing {len(source_files)} files")
    if skipped_complete > 0:
        logger.info(f"Skipped {skipped_complete} files with complete PSD data")
    
    if not source_files:
        logger.info("No files to process. All files either missing or already complete.")
        return
    
    # Process based on parallelization mode
    if len(source_files) == 1 or args.max_parallel_subjects == 1:
        # Single subject or sequential subjects with parallel blocks
        success_count = 0
        for source_file in source_files:
            psd_file, success = process_subject_parallel_blocks(
                source_file_path=source_file,
                save_path=args.save_path,
                fmin=args.fmin,
                fmax=args.fmax,
                method=args.method,
                n_jobs_psd=args.n_jobs_psd,
                max_parallel_blocks=args.max_parallel_blocks,
                store_vertices=args.store_vertices,
                overwrite=args.overwrite
            )
            if success:
                success_count += 1
        
        total_success = success_count == len(source_files)
    else:
        # Multiple subjects in parallel
        total_success = process_multiple_subjects_parallel(
            source_files=source_files,
            save_path=args.save_path,
            fmin=args.fmin,
            fmax=args.fmax,
            method=args.method,
            n_jobs_psd=args.n_jobs_psd,
            max_parallel_blocks=args.max_parallel_blocks,
            max_parallel_subjects=args.max_parallel_subjects,
            store_vertices=args.store_vertices,
            overwrite=args.overwrite
        )
    
    if not total_success:
        sys.exit(1)

if __name__ == "__main__":
    main()