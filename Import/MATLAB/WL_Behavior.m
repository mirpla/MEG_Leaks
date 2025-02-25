function [OBhv] = OBehavior(Basepath,c,s)
headerfile  = [Basepath, 'EEG\OddballOB',num2str(c),sprintf('%03d',s),'.vhdr'];
dataset     = [Basepath, 'EEG\OddballOB',num2str(c),sprintf('%03d',s),'.vhdr'];

% read the header information and the events from the data
hdr   = ft_read_header(headerfile);
event = ft_read_event(dataset);

% search for "trigger" events
value  = {event(find(strcmp('Stimulus', {event.type}))).value}';
sample = [event(find(strcmp('Stimulus', {event.type}))).sample]';

% look for the combination of a trigger "7" followed by a trigger "64"
% for each trigger except the last one
trl = [];
trlidx = 1;
for j = 1:(length(value)-1)
    trg1 = value{j};
    trg2 = value(j+1);
    if strcmp(trg1,  'S  7') && strcmp(trg2, 'S  6')
        OBhv(trlidx) = ((sample(j+1) - sample(j))/hdr.Fs)-0.4;
        trlidx = trlidx +1;
    end
end