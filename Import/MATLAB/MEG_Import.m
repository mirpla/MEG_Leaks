% Script that contains all the scripts for importing the data related to the Robertson MEG project in a BIDS compatible 
% format.  
formatSpec = '%s %s %s %f %s';
sub_key     = readtable([base_path, 'Data\Subject_Keys.csv'], 'Format', formatSpec); % This file keeps track of the subject and session mappings
sub_info    = readtable([base_path, 'Data\Subject_Information.csv']); % Load in the subject information file

sub_codes = {'00'}; % subjects codes that are meant to be imported (row vector)
ses = 2;            % sessions that are meant to be imported

% loop through the subejcts to be imported and import the sessions
for sesidx = ses
    for subidx = 1:size(sub_codes,2)
        rowidx = find(contains(sub_key{:, 1}, sub_codes{subidx})); % find the correct subject
        ses_col_idx = [];
        % find the correct session and make sure it's only one entry per session
        for i = colidx'
            if strcmp(sub_key{i, 1}, sub_codes{subidx}) && sub_key{i, 4} == sesidx
                ses_col_idx = [ses_col_idx; i]; % This should only be 1 number. If there is multiple something is wrong
            end
        end
        
        paths.data_MEG  = [raw_path,sub_key{ses_col_idx,2}{1},'\',sub_key{ses_col_idx,3}{1},'\'];
        paths.data_beh  = [raw_path,sub_key{ses_col_idx,2}{1},'\SRT\'];        
        paths.mri       = ['\\raw\10\Anatomical\',upper(sub_key{ses_col_idx,2}{1}),'_RAW_32_ADNI'];
        
        MEG_flag = 1; % set to one if you want MEG data to also be transferred
        Convert_BIDS(base_path,paths,sub_codes,sub_info,subidx,sesidx,MEG_flag)
        
        %BIDS_path = ;
        Lock = 1;% Determines whether analysis is stimulus or response locked 
        Import_SRT(base_path,sub_key.subjectNumber{1},sesidx,Lock) % Script for importing the SRT data for the respective session

    end
end
