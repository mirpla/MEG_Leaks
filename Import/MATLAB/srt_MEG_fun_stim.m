function [trl, event] = srt_MEG_fun_stim(cfg)

% read the header information and the events from the data
hdr   = ft_read_header(cfg.dataset);
event = ft_read_event(cfg.dataset);

% search for "trigger" events
% values 5 and 1 follow every flip
value  = {event(find(strcmp('STI101', {event.type}))).value}';
sample = [event(find(strcmp('STI101', {event.type}))).sample]';

if any(strcmp(hdr.label, 'STI102')) 
    button_val      = {event(find(strcmp('STI102', {event.type}))).value}';
    button_sample   = {event(find(strcmp('STI102', {event.type}))).sample}';
    
    
else
    warning('STI102 not detected. Button box inputs have to be inferred, for this Dataset, please make sure to doublecheck the values')
    
end  
% determine the number of samples before and after the trigger
pretrig  = -round(cfg.trialdef.pre  * hdr.Fs);
posttrig =  round(cfg.trialdef.post * hdr.Fs);

% look for the combination of a trigger "7" followed by a trigger "64"
% for each trigger except the last one
trl = [];
for j = 1:(length(value)-1)
    trg1 = value{j};
    if strcmp(trg1,  'S  6')
        trlbegin = sample(j) + pretrig;
        trlend   = sample(j) + posttrig;
        offset   = pretrig;
        newtrl   = [trlbegin trlend offset];
        trl      = [trl; newtrl];
    end
end