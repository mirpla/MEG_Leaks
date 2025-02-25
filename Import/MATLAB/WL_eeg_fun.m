function [trl, event] = WL_eeg_fun(cfg)

% read the header information and the events from the data
hdr   = ft_read_header(cfg.dataset);
event = ft_read_event(cfg.dataset);

% search for "trigger" events
value  = {event(find(strcmp('Stimulus', {event.type}))).value}';
sample = [event(find(strcmp('Stimulus', {event.type}))).sample]';

% determine the number of samples before and after the trigger
pretrig  = -round(cfg.trialdef.pre  * hdr.Fs);
posttrig =  round(cfg.trialdef.post * hdr.Fs);

trl = [];
first = 0;
for j = 1:(length(value)-1)
    trg1 = value{j};
    trg2 = value(j+1);
    if strcmp(trg1,  'S  6')
        trlbegin = sample(j) + pretrig;
        trlend   = sample(j) + posttrig;
        offset   = pretrig;
        newtrl   = [trlbegin trlend offset 3];
        trl      = [trl; newtrl];
    elseif strcmp(trg1,  'S 14') && strcmp(trg2,  'S 14')
        trlbegin = sample(j);
        trlend   = sample(j+1);
        offset   = 0;
        newtrl   = [trlbegin trlend offset 4];
        trl      = [trl; newtrl];
    end
end
end