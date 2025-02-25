function [Condition, subjID, ID, sB, wB, tB] = CheckID
% This Script requests input from the user for a Subject ID 
% Valid subject ID consists of 4 numbers, where the first number indicates
% the condition and the last 3 code the number of the subject 
% Example: subjID = 1001, is the first subject in condition 1
%   1 = Congruent
%   2 = Incongruent
% Also requests which block to start on for the SRT and wordlist. If left
% blank the default is 1
% lastly, checks whether the SRT should be skipped or not

flagID      = 0;
while flagID == 0    
    subjID      = input('Subject ID): ','s');
    if isempty(subjID)
        disp('No Input Detected; Please enter valid subjID');
    else 
        subjID = str2double(subjID);
        if ~isnumeric(subjID)
            disp('Input was not a number; Please enter valid subjID');
        elseif  numel(num2str(subjID))<4 || numel(num2str(subjID))>4
            disp('Input does not follow specificationis for subject IDs; Please consult documentation')
        elseif subjID > 3999 || subjID < 1000
            disp('Input was not a valid number; Please consult documentation for valid subject IDs')
        else
            flagID = 1;
        end
    end
end

% Ask for which blocks to start and which task. If no input is given the default is set to 1
Block{3}      = str2double(input('Start from SRT (1) or Wordlist (2) : ','s'));
if Block{3} == 1 || isnan(Block{3}) % only ask about SRT if it's included
    Block{1}      = str2double(input('SRT Block: ','s'));
end 
Block{2}      = str2double(input('WL Block: ','s'));

for x = 1:length(Block) % Set numbers to 1 if ignored
    if  isempty(Block{x})==1 || isnan(Block{x})==1 
        Block{x} = 1;
    end
end

sB = Block{1};
wB = Block{2};
tB = Block{3};

selnum = 1000;
ID = mod(subjID, selnum); 
Condition = (subjID - ID) / selnum;