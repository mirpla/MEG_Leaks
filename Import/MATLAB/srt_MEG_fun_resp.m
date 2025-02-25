function [trl, event] = srt_MEG_fun_resp(cfg)

% read the header information and the events from the data
hdr   = ft_read_header(cfg.dataset);
event = ft_read_event(cfg.dataset);

% search for "trigger" events
value  = {event(find(strcmp('Stimulus', {event.type}))).value}';
sample = [event(find(strcmp('Stimulus', {event.type}))).sample]';

% determine the number of samples before and after the trigger
pretrig  = -round(cfg.trialdef.pre  * hdr.Fs);
posttrig =  round(cfg.trialdef.post * hdr.Fs);

% look for the combination of a trigger "7" followed by a trigger "64"
% for each trigger except the last one
trl = [];
for j = 1:(length(value)-1)
    trg1 = value{j};
    if strcmp(trg1,  'S  7')
        trlbegin = sample(j) + pretrig;
        trlend   = sample(j) + posttrig;
        offset   = pretrig;
        newtrl   = [trlbegin trlend offset 1];
        trl      = [trl; newtrl];
    elseif strcmp(trg1,  'S  9')
        trlbegin = sample(j) + pretrig;
        trlend   = sample(j) + posttrig;
        offset   = pretrig;
        newtrl   = [trlbegin trlend offset 2];
        trl      = [trl; newtrl];
    end    
end