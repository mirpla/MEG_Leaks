%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Trial sequences %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Txt for random SRT Trials where:
%       First column indicates Block
%       Second column indicates correct finger press
%       Third row indicates random or main sequence (82 = random, 84 = sequence)

%% Main Experiment Trial Generation
clear
clc
% Define parameters for each block
block_sizes = [280, 400, 280];
num_conditions = 4;

% Initialize data matrix
data = [];

for block = 1:numel(block_sizes)
    num_trials = block_sizes(block);
    
    % Create an array representing the four different conditions
    conditions = repmat(1:num_conditions, 1, ceil(num_trials/num_conditions));
    conditions = conditions(randperm(length(conditions)));

    % Initialize trial sequence
    trial_sequence = zeros(1, num_trials);

    % Randomize trial sequence ensuring no trial is followed by the same trial
    trial_sequence(1) = randi(num_conditions);
    for i = 2:num_trials
        available_conditions = setdiff(1:num_conditions, trial_sequence(i-1));
        trial_sequence(i) = available_conditions(randi(length(available_conditions)));
    end

    % Create block data matrix
    block_data = [repmat(block, 1, num_trials); trial_sequence; repmat(82, 1, num_trials)]';

    % Append block data to main data matrix
    data = [data; block_data];
end

% Save data into a tab-delimited TXT file
dlmwrite('seq-2.txt', data, 'delimiter', '\t');