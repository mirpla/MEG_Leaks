function [SkillLearning, fcRTc, fLim, eTRL, ErrorRate]  = SRT_ImportFit(base_path,fitLimit,SLWindow,ID)
SeqPath     = [base_path, 'Sequence_files\'];
SeqOri{1}   = dlmread([SeqPath,'seq-1.txt']);
SeqOri{2}   = dlmread([SeqPath,'seq-2.txt']);

% Load Subject information to track inclusion criteria
SubInfo         = readtable([base_path,'Data/Subject_Information.csv']);

rel_info        = [SubInfo.ID,SubInfo.Explicitness];
rel_info(:,3) = floor(rel_info(:,1)/1000); % extract the order; 1 = Exp first 2 = Cont first
rel_info(:,4) = mod(rel_info(:,1), 1000); % extract the corresponding subject IDs

% only analyse subjects with one session
first_ses(:,1)  = (rel_info(:,3) == 1) & (rel_info(:,2) == 0); % Experimental session in First session
first_ses(:,2)  = rel_info(:,3) == 2; % Control condition in first session
first_ses(:,3)  =  SubInfo.Random; % mark subjects that used random vs non-random control

data_path = fullfile(base_path, '\Data\');
sub_folders = dir(fullfile(data_path, 'sub-*'));

data = cell(size(sub_folders,1),2);
% Loop through each subject folder
for i = 1:size(sub_folders,1)
    % find the names of the subject folders
    sub_folder = sub_folders(i).name;
    
    % find and loop through all sessions
    ses_folders = dir(fullfile(data_path, sub_folder, 'ses-*'));
    for j = 1:length(ses_folders)
        ses_folder = ses_folders(j).name;
        % Only continue if ses_folder is not empty
        if ~isempty(ses_folder)
            c = sscanf(ses_folder, 'ses-%d'); % Extract session number
            
            % Define the path to behavior folder
            beh_path = fullfile(data_path, sub_folder, ses_folder,'beh');
            
            % Only continue if behavior folder actually exists
            if isfolder(beh_path)
                % Define the file name based on the session and subject number
                s = sscanf(sub_folder, 'sub-%d');
                
                % Load or process the .csv file (e.g., read its content)
                [FilesData] = dir(fullfile(beh_path,'*.txt'));                
                for b = 1:size(FilesData,1)
                    AllB        = readmatrix(fullfile(beh_path,FilesData(b).name));
                    oriID       = SeqOri{c}(SeqOri{c}(:,1) == b,3);
                    if size(oriID,1)<size(AllB,1)
                        xtraT   = size(AllB,1) - size(oriID,1);
                        AllB    = AllB(xtraT+1:end,:);
                    elseif size(oriID,1)>size(AllB,1)
                        warning('Insufficient trials detected, the Experiment seems to have been prematurely terminated mid-block')
                        %error('Insufficient trials detected, the Experiment seems to have been prematurely terminated mid-block')
                    end
                    
                    trialID     = AllB(:,5);
                    RandDum     = AllB(trialID == ID.Random,:);
                    
                    if c == 1
                        RT{s}{c,b}{1}  = RandDum(1:end/2,:);
                        RT{s}{c,b}{2}  = AllB(trialID == ID.Sequence,:);
                        RT{s}{c,b}{3}  = RandDum(end/2+1:end,:);
                        
                    elseif c == 2
                        RT{s}{c,b}{1} = RandDum;
                    end 
                        for f = 1:size(RT{s}{c,b},2)
                            for trls = 1:size(RT{s}{c,b}{f},1)
                                cRT{s}{c,b}{f}(trls)       = RT{s}{c,b}{f}(trls,5+RT{s}{c,b}{f}(trls,4));
                                if size(find(RT{s}{c,b}{f}(trls,6:9)),2)>1
                                    eTRL{s}{c,b}{f}(trls,:)= RT{s}{c,b}{f}(trls,6:9);
                                else
                                    eTRL{s}{c,b}{f}(trls,:)= [0 0 0 0];
                                end
                            end
                            ErrorRate{f}{s}(c,b)   = sum(any(eTRL{s}{c,b}{f},2))/size(any(eTRL{s}{c,b}{f},2),1);
                            if f == 2
                                y = 2;
                            else
                                y = 1;
                            end
                            % filter out trials too fast or too slow
                            [fcRTc{s}{c,b}{f}, fLim{s}{c,b}{f}{1}, fLim{s}{c,b}{f}{2}] = lowess([(1:length(cRT{s}{c,b}{f}));cRT{s}{c,b}{f}]',fitLimit(y),1);
                        end
                        if c ==1
                            SkillLearning{s}(c,b) = mean(fcRTc{s}{c,b}{3}(1:SLWindow,3)) - mean(fcRTc{s}{c,b}{2}(end-(SLWindow-1):end,3));
                        end
                end
            end
        end
    end
end
end