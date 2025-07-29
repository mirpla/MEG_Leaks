#%% 
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from meg_analysis.Scripts.Rest_Analyses.Source_Data_Rest import make_source_rest
#%%
# set up path info
script_path = Path(__file__).resolve()
script_dir = script_path.parent
project_root = script_dir.parents[3] # Root folder on windows for some reason is processed differently and has to be 2
sub_path = project_root / 'Data' / 'Subject_Information.csv'

print(project_root / 'Data' / 'Subject_Information.csv')
# set up participant info
data = pd.read_csv(project_root / 'Data' / 'Subject_Information.csv', encoding='latin1') 
Included = ~data['Excluded'].astype(bool)
#subs = data['sub'][Included].tolist()
subs = ['sub-76'] # 'sub-XX'
ses = 'ses-1'  
print(subs)

# source loc parameters
src_d = 0.8  # depth
src_l = 0.3  # loose
src_f = False  # Fixed? (True/False)
src_method = 'eLORETA' # inverse method to use for the source localization (e.g., 'dSPM', 'sLORETA')
use_baseline_cov = False # Set to True to use baseline block instead of empty room

# Set up error logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = script_dir / f"source_localization_errors_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting source localization for {len(subs)} subjects")
logger.info(f"Error log will be saved to: {log_file}")

# Track successes and failures
successful_subjects = []
failed_subjects = []

# Run source localization for each subject
for i, sub in enumerate(subs, 1):
    try:
        logger.info(f"[{i}/{len(subs)}] Processing {sub}")
        make_source_rest(sub, ses, src_d, src_l, src_f, src_method, 
                        n_jobs=6, max_block_workers=3, use_baseline_cov=use_baseline_cov)
        successful_subjects.append(sub)
        logger.info(f"[{i}/{len(subs)}] {sub} completed successfully")
    except Exception as e:
        failed_subjects.append((sub, str(e)))
        logger.error(f"[{i}/{len(subs)}] Error processing {sub}: {str(e)}")
        continue  # Continue with next subject even if one fails

# Summary report
logger.info("="*60)
logger.info("FINAL SUMMARY")
logger.info("="*60)
logger.info(f"Successfully processed: {len(successful_subjects)}/{len(subs)} subjects")
logger.info(f"Failed: {len(failed_subjects)}/{len(subs)} subjects")

if successful_subjects:
    logger.info(f"Successful subjects: {', '.join(successful_subjects)}")

if failed_subjects:
    logger.error("FAILED SUBJECTS:")
    for sub, error in failed_subjects:
        logger.error(f"  {sub}: {error}")

logger.info(f"Detailed log saved to: {log_file}")
print(f"\n*** Error log saved to: {log_file} ***")
# %%