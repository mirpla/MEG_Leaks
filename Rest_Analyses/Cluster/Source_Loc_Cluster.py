import pandas as pd
from pathlib import Path
from meg_analysis.Scripts.Rest_Analyses.Source_Data_Rest import make_source_parallel

# set up path info
script_path = Path(__file__).resolve()
script_dir = script_path.parent
project_root = script_dir.parents[3] # Root folder on windows for some reason is processed differently and has to be 2
sub_path = project_root / 'Data' / 'Subject_Information.csv'

# set up participant info
data = pd.read_csv(project_root / 'Data' / 'Subject_Information.csv', encoding='latin1') 
Included = ~data['Excluded'].astype(bool)
#subs = data['sub'][Included].tolist()
subs = ['sub-65','sub-68'] # 'sub-XX'
ses = 'ses-1'  
print(subs)

# source loc parameters
src_d = 0.8  # depth
src_l = 0.4  # loose
src_f = False  # Fixed? (True/False)
src_method = 'sLORETA' # inverse method to use for the source localization (e.g., 'dSPM', 'sLORETA')
make_source_parallel(subs, ses, src_d, src_l, src_f, src_method, max_subjects=2)