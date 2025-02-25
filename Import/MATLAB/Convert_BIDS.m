function Convert_BIDS(base_path,paths,sub_codes,sub_info,subidx,ses,MEG_flag)

% Copy collected data from the raw server and convert it in BIDS compatible format
run_name    = {'_SRT','_WL_FREE','_WL_PACED','_WL_SPBUT'};
task_name   = {'SRT','WL','WL_PACE','WL_AMPEL'};
sex_name    = {'Female','Male'};
for runidx      = 1:4 % run 1 = SRT, run 2 = WL During pilot WL subdivided into more sections   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!ID CHANGE FOR REALPART !!!!!!!!!!!!!!!!!!!!!!!!%%%%%%%%%%%%%
    ID = 2000;% ses*1000+ str2num(sub_codes{subidx})
    sub_info_idx = find(sub_info.ID==ID);
    
    orig_data_path = [paths.data_MEG, 'MEG_',num2str(ID),run_name{runidx},'.fif'];
    
    BIDS_root = [base_path,'Data\'];
    
    cfg             = [];
    cfg.bidsroot    = BIDS_root;  % write to the present working directory
    cfg.sub         = sub_codes{subidx};
    cfg.ses         = num2str(ses);
    cfg.run         = runidx;
    cfg.task        = task_name{runidx};
    
    cfg.meg.writejson                   = 'replace';
    cfg.dataset_description.writejson   = 'replace';
    
    cfg.datatype    = 'meg';
    cfg.dataset     = orig_data_path; % this is the intermediate name
    
    cfg.method = 'copy'; % the original data is in a BIDS-compliant format and can simply be copied
    
    cfg.InstitutionName             = 'University of Glasgow';
    cfg.InstitutionalDepartmentName = 'Centre for Cognitive Neuroimaging (CCNI)';
    cfg.InstitutionAddress          = '58 Hillhead Street, G12 8QB Scotland, UK';
    cfg.Manufacturer                = 'MEGIN';
    
    % required for dataset_description.json
    cfg.dataset_description.Name                = 'Electrophysiological underpinnings of information transfer/leaks between memory systems';
    cfg.dataset_description.BIDSVersion         = 'v1.9.0';
    
    % optional for dataset_description.json
    cfg.dataset_description.License             = 'UNKNOWN';
    cfg.dataset_description.Authors             = {'M van der Plas, E Robertson'};
    cfg.dataset_description.EthicsApprovals     = 'MVLS	200230185';
    % Short make longer if actually shared
    cfg.TaskDescription = 'In this Experiment participants performed a serial reaction time task during which they had to press prompted buttons. This task was followed in a second run by a word list recall task.';
    
    % tsv relevant stuff
    cfg.participants.age        = sub_info{sub_info_idx,3};
    cfg.participants.sex        = sex_name{sub_info{sub_info_idx,4}+1};
    
    
    % For anatomical and functional MRI data you can specify cfg.dicomfile to read the
    % detailed MRI scanner and sequence details from the header of that DICOM file. This
    % will be used to fill in the details of the corresponding JSON file.
    %   cfg.dicomfile               = string, filename of a matching DICOM file for header details (default = [])
    %   mri_dum = dir(mri_path);
    %   mri = ft_read_mri([mri_path,'/',mri_dum(3).name]);
    %   cfg.deface                  = 'no')
    
    % General BIDS options that apply to all functional data types are
    %   cfg.TaskName                    = string
    %   cfg.TaskDescription             = string
    %   cfg.Instructions                = string
    %   cfg.CogAtlasID                  = string
    %   cfg.CogPOID                     = string
    %
    % There are more BIDS options for the mri/meg/eeg/ieegÂ data type specific sidecars.
    % Rather than listing them all here, please open this function in the MATLAB editor,
    % and scroll down a bit to see what those are. In general the information in the JSON
    % files is specified by a field that is specified in CamelCase
    %   cfg.mri.SomeOption              = string, please check the MATLAB code
    %   cfg.meg.SomeOption              = string, please check the MATLAB code
    
    % Convert the .fif recorded MEG data from the separate runs and sessions into the BIDS format
    if MEG_flag == 1
        data2bids(cfg);
    end
    %% Copy over behavioral Data
    if runidx == 1
        % List all files in the source directory
        dest_dir = [BIDS_root,'sub-',cfg.sub,'\ses-',cfg.ses,'\beh\']; % Specify the path to your destination directory
        fileList = dir(fullfile(paths.data_beh, '*.txt'));
        
        % Loop through each file in the source directory
        for i = 1:length(fileList)
            filename = fileList(i).name;
            
            % Check if the file name matches the pattern "MEG_p1001_SRTT_blocksX"
            if startsWith(filename, ['MEG_p',num2str(ID),'_SRTT_blocks']) && endsWith(filename, '.txt')
                % Extract the X value
                X = sscanf(filename, 'MEG_p1001_SRTT_blocks%d.txt');
                
                % Check if X is in the range [1, 3]
                if ~isempty(X) && X >= 1 && X <= 3
                    if ~isfolder(dest_dir)
                        mkdir(dest_dir);
                    end
                    % Copy the file to the destination directory
                    copyfile(fullfile(paths.data_beh, filename), fullfile(dest_dir, filename));
                end
            end
        end
    end
end