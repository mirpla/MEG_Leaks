startup 

%% Sort out path/hierarchy and Stimulus paths
sca;
close all;
clear;
debug   = 1;
audio   = 0;
MEG     = 1;
scriptpath = cd(fileparts(which('MEG_Experiment.m')));       % open the folder in which the scripts for the experiment are located
addpath(scriptpath)                                         % Add the path where all the scripts are located just in case
idcs    = strfind(scriptpath , filesep);                    % determine the file separator on the system so the path can be determined
path    = scriptpath(1:idcs(end-1)-1);                      % get the path to the experiment 

% Determine Subject ID and Condition
% Valid subject ID consists of 4 numbers, where the first number indicates
% the condition and the last 3 code the number of the subject
[SubInfo.Cond, SubInfo.subjID, SubInfo.ID, Block.SRT, Block.WL, Order] = CheckID;

% Standardize key names on this system
KbName('UnifyKeyNames');

%% SRT Experiment

if Order == 1 % Check whether SRT should be skipped or not
    %MEG_EyeTracker
    
    MEG_SRT(SubInfo, path, debug, Block.SRT,MEG) % Perform SRT
elseif Order ~= 2 
    error('Wrong input: Please only use 1 to indicate the start from SRT and 2 to jump to the wordlist. No other characters are allowed')
end 

pahandle = [];
if audio == 1
    InitializePsychSound;
    
    freq = 44100;
    pahandle = PsychPortAudio('Open', [], 2, 0, freq, 2);
end


selfpaced = 2; % 0 nothing, 1 self, 2 ampel
MEG_WL(SubInfo, path, debug, 5,pahandle,audio,MEG, selfpaced) % Perform Wordlist