function [trl, event] = WL_eeg_fun_Baseline(cfg)

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
trg1 = value{1};

if strcmp(trg1,  'S 14')
    trlbegin = sample(1) + pretrig;
    trlend   = sample(1) + posttrig;
    offset   = pretrig;
    newtrl   = [trlbegin trlend offset 2];
    trl      = [trl; newtrl];   
end