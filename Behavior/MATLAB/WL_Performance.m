function [SR, Edit_Dist] = WL_Performance(base_path)
figsize = [100, 100, 1200, 600];

% Load Subject information to track inclusion criteria 
SubInfo         = readtable([base_path,'Data/Subject_Information.csv']);

rel_info        = [SubInfo.ID,SubInfo.Explicitness];
rel_info(:,3) = floor(rel_info(:,1)/1000); % extract the order; 1 = Exp first 2 = Cont first
rel_info(:,4) = mod(rel_info(:,1), 1000); % extract the corresponding subject IDs


first_ses(:,1)  = (rel_info(:,3) == 1) %& (rel_info(:,2) == 0); % Experimental session in First session
first_ses(:,2)  = rel_info(:,3) == 2; % Control condition in first session
second_ses(:,1) = (rel_info(:,3) == 2) %& (rel_info(:,2) == 0); % Control condition in Second session
second_ses(:,2) = rel_info(:,3) == 1; % Experimental condition ins Second session

WL = table2array(readtable([base_path,'\Data\MEG_WL_ITEMS.csv'],'Range','A:C'));


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
            ses_number = sscanf(ses_folder, 'ses-%d'); % Extract session number
            
            % Define the path to behavior folder
            beh_path = fullfile(data_path, sub_folder, ses_folder,'beh');
            
            % Only continue if behavior folder actually exists
            if isfolder(beh_path)
                % Define the file name based on the session and subject number
                sub_number = sscanf(sub_folder, 'sub-%d');
                file_name = sprintf('%d%03d.csv', ses_number, sub_number);
                
                % Define the full path to the .csv file
                file_path = fullfile( beh_path, file_name);
                
                % Check if the file exists
                if isfile(file_path)
                    % Load or process the .csv file (e.g., read its content)
                    data{sub_number, ses_number} = table2array(readtable(file_path,'Range','B:K'));                    
                else
                    fprintf('No file for subject%d - Session%d.\n',sub_number,ses_number );
                end
            end
        end
    end
end

%% Figures for First session only 
figure('Position', figsize)
tiledlayout(1,2) 

% determine colors 
col = {[0.8 0.8 1],[1 0.8 0.8]};
meancol = {[0 0 1],[1 0 0]};

% Figure 1
nexttile 
hold on
title(['Serial Recall Session 1'])
SR = zeros(size(data,1),10);
for c = 1:2
    for s = rel_info(first_ses(:,c),4)'
        for t = 1:10
            data{s,c}(isnan(data{s,c}(:,t)),t)=0;
            x = (data{s,c}(:,t))';
            
            D = diff(x)==1;
            
            i=reshape(find(diff([0;D';0])~=0),2,[]);
            [lgtmax,jmax]=max(diff(i));
            if ~isempty(lgtmax)
                SR(s,t) = lgtmax+1;
            end
        end
        plot([SR(s,:)],'Color',col{c},'LineWidth',2)
    end
    allSR{1,c} = SR;
    MeanLine(c) = plot([nanmean(SR(rel_info(first_ses(:,c),4)',:),1)], 'Color', meancol{c},'LineWidth',3 );
end

uistack(MeanLine(1), 'top');
uistack(MeanLine(2), 'top')
hold off
ylim([1 12])
xlim([1 10])
xlabel('Trial')
ylabel('Serial Recall')
legend([MeanLine(1),MeanLine(2)], 'Congruent','Random','Location','eastoutside')

% second session 
nexttile 
col = {[0.8 0.8 1],[1 0.8 0.8]};
meancol = {[0 0 1],[1 0 0]};

hold on
title(['Serial Recall Session 2'])
SR = zeros(size(data,1),10);
for c = 1:2
    for s = rel_info(second_ses(:,c),4)'
        if ~isempty(data{s,c})
            for t = 1:10                
                data{s,c}(isnan(data{s,c}(:,t)),t)=0;
                x = (data{s,c}(:,t))';
                
                D = diff(x)==1;
                
                i=reshape(find(diff([0;D';0])~=0),2,[]);
                [lgtmax,jmax]=max(diff(i));
                if ~isempty(lgtmax)
                    SR(s,t) = lgtmax+1;
                end
            end
            plot([SR(s,:)],'Color',col{c},'LineWidth',2)
        else
            SR(s,:) = NaN;
        end 
    end
    allSR{2,c} = SR; 
    MeanLine(c) = plot([nanmean(SR(rel_info(second_ses(:,c),4)',:),1)], 'Color', meancol{c},'LineWidth',3 );
end

uistack(MeanLine(1), 'top');
uistack(MeanLine(2), 'top')
hold off
ylim([1 12])
xlim([1 10])
xlabel('Trial')
ylabel('Serial Recall')


%% Levensthein Distance
figure('Position', figsize)
tiledlayout(1,2) 

nexttile
hold on    
title(['Edit Distance Session 1'])
    
for c = 1:2
    for s = rel_info(first_ses(:,c),4)'
        for t = 1:10
            data{s,c}(isnan(data{s,c}(:,t)),t)=0;
            x = (data{s,c}(:,t))';
            strdum = 'ABCDEFGHIJKL';
            strdum2 = strdum;
            strdum( x == 0 ) = [];
            Edit_Dist(s,t) = editDistance(strdum2, strdum(x(x~=0)));
        end
        plot( 12- Edit_Dist(s,:),'Color',col{c},'LineWidth',3)
    end    
    mean_edit = nanmean(12- Edit_Dist(rel_info(first_ses(:,c),4)',:),1);
    MeanLineLV(c) = plot(mean_edit, 'Color', meancol{c},'LineWidth',3 );
end
uistack(MeanLineLV(2), 'top');
uistack(MeanLineLV(1), 'top')
hold off
ylim([0 12])
xlim([1 10])
xlabel('Trial')
ylabel('Normalized Levensthein Distance')
legend([MeanLineLV(1),MeanLineLV(2)], 'Congruent','Random', 'Location','eastoutside')
% session 2
nexttile
hold on    
title(['Edit Distance Session 2'])
    
for c = 1:2
    for s = rel_info(second_ses(:,c),4)'
        if ~isempty(data{s,c})
            for t = 1:10
                data{s,c}(isnan(data{s,c}(:,t)),t)=0;
                x = (data{s,c}(:,t))';
                strdum = 'ABCDEFGHIJKL';
                strdum2 = strdum;
                strdum( x == 0 ) = [];
                Edit_Dist(s,t) = editDistance(strdum2, strdum(x(x~=0)));
            end
            plot( 12- Edit_Dist(s,:),'Color',col{c},'LineWidth',3)
        else
            Edit_Dist(s,:) = NaN;
        end
    end
    mean_edit = nanmean(12- Edit_Dist(rel_info(second_ses(:,c),4)',:),1);
    MeanLineLV(c) = plot(mean_edit, 'Color', meancol{c},'LineWidth',3 );
end
uistack(MeanLineLV(2), 'top');
uistack(MeanLineLV(1), 'top')
hold off
ylim([0 12])
xlim([1 10])
xlabel('Trial')
ylabel('Normalized Levensthein Distance')

%% Make plots of serial recall ses 1 vs ses 2 

cond_diff_id(:,1) = any(allSR{1,1},2) & any(allSR{2,2},2);
cond_diff_id(:,2) = any(allSR{2,1},2) & any(allSR{1,2},2);

cond_diff{1} = allSR{1,1}(cond_diff_id(:,1),:)-allSR{2,2}(cond_diff_id(:,1),:);
cond_diff{2} = allSR{2,1}(cond_diff_id(:,2),:)-allSR{1,2}(cond_diff_id(:,2),:);
figure
title('Session 1 - Session 2')
hold on
expf    = plot( cond_diff{1}','Color',col{1},'LineWidth',3);
condf   = plot( cond_diff{2}','Color',col{2},'LineWidth',3);
plot(zeros(1,10),'k')
hold off
legend([expf(1), condf(1)], 'Congruent - Random', 'Random - Congruent')
 
    