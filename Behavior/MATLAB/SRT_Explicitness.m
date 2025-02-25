function [SubjIdx, RecallScore] = SRT_Explicitness(base_path)
% Use Explicitness Questionnaire to Calculate SRT performance scores in a
% Recall score table where:
%   Column 1 = Subject ID
%   Column 2 = indicates the Score based on the answer to question 1, where
%       the strenght of certainty of the presence of a sequence is coded with 0
%       if random, 0.5 if unsure, 1 if sure there is a sequence
%   Column 3 = Explicitness based on Verbal Report
%   Column 4 = Mean difference of scores for the Recognition questions
%       where the real answer are subtracted with the distractor questions

seq         =   [
                2,3,1,4,3,2,4,1,3,4,2,1; 
                2,1,3,2,3,4,1,4,2,4,3,1
                ];

TablePath   = [base_path,'Data/ExplicitnessQuestionnaires.xlsx'];
OutputPath  = [base_path,'Log/RecallScore.csv'];

opt = detectImportOptions(TablePath);
opt = setvartype(opt,5,'int64');
FullExpQ    = readtable(TablePath,opt);

subIDs      = table2array(FullExpQ(:,1)); 
Recall      = table2array(FullExpQ(:,5));
Questions   = table2array(FullExpQ(:,7:end));
CondID      = (subIDs - mod(subIDs, 1000)) / 1000;
Awareness  = table2array(FullExpQ(:,4));

AwScFull = contains(Awareness,'The same sequence of movements occurred throughout the entire task')|...
    contains(Awareness,'The same sequence of movements would appear');

AwScHalf = contains(Awareness,'The movements of the dots from position to position was often predictable')*0.5;

AwarenessScore = AwScFull+AwScHalf;

Qseqsiz = size(FullExpQ.Properties.VariableNames(7:end),2);
Qseq = cell(1,Qseqsiz);
QIdx = cell(1,Qseqsiz);
for Qseqcntr = 1:Qseqsiz
    Qseq{1,Qseqcntr} = int64(cellfun(@str2num, split(extractBetween(FullExpQ.Properties.VariableNames{Qseqcntr+6},'__','__')))');
    Qseqdum = num2str( Qseq{1,Qseqcntr})-'0';
    for shiftcnt = 0:size(seq,2)-1
        if ~isempty(strfind(circshift(seq,shiftcnt),Qseqdum)) %#ok<*STREMP>
            QIdx{1,Qseqcntr} = 1;
        end       
    end
end

QIdxnum = ~cellfun(@isempty,QIdx);

% Target Congruent
Q{1,1} = Questions(:, QIdxnum(1,:));

% Distractor
Q{1,2} = Questions(:, ~QIdxnum(1,:));

RecognitionScore = mean(Q{1,1},2) - mean(Q{1,2},2);

DumScore = cell(size(Recall,1),1);
for Rcnt = 1:size(Recall,1) % cycle through subject
    if ~isempty(Recall(Rcnt,1))  % Check that there was actually a report
        for giveSeq = 1:size(Recall(Rcnt,1),1) % cycle through the recalled sequences if there are multiple
            RecallDum = num2str(Recall(Rcnt,giveSeq))-'0';
            if  size(RecallDum,2)>3 % check that recalled sequence is larger than 3
                for shiftcnt = 0:size(seq,2)-1
                    if ~isempty(strfind(circshift(seq,shiftcnt), RecallDum)) % #ok<STREMP>
                        DumScore{Rcnt} = 1;
                    end
                end
            end
        end
    end
end


RecallScoreDum      = [subIDs,AwarenessScore,~cellfun(@isempty,DumScore)];
RecallScore         = [RecallScoreDum,RecognitionScore];

% figure
% subplot(2,1,1)
% scatter(1:length(RecognitionScore),RecognitionScore)



%%

% check which Seq is used
SubInfo = readtable([base_path,'Log/Subject_Information.csv']);
SubProp = [SubInfo.ID,SubInfo.Excluded,SubInfo.Explicitness];

for x = 1:size(SubProp,1)
    SI(x,2)     = mod(SubProp(x,1), 1000); % Subject number
    SI(x,1)     = (SubProp(x,1) - SI(x,2)) / 1000; % Condition
end

SubjIdx{1,1} = SI(SubProp(:,2)~=1 & SubProp(:,3)~=1 & SI(:,1)==1,2);    % Implicit Subjects
SubjIdx{1,2} = SI(SubProp(:,2)~=1 & SubProp(:,3)==1 & SI(:,1)==1,2);    % Explicit Subjects


writematrix(RecallScore,OutputPath)

