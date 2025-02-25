function [trl, event] = WL_MEG_fun(cfg)

% read the header information and the events from the data
hdr   = ft_read_header(cfg.dataset);
event = ft_read_event(cfg.dataset);

% search for "trigger" events
value  = {event(find(strcmp('STI101', {event.type}))).value}';
sample = [event(find(strcmp('STI101', {event.type}))).sample]';

% determine the number of samples before and after the trigger
pretrig  = -round(cfg.trialdef.pre  * hdr.Fs);
posttrig =  round(cfg.trialdef.post * hdr.Fs);

% look for 779 which corresponds to recall +image onset (768 + 11)
trl = [];
for j = 1:(length(value)-1)
    trg = value{j};
    if trg == 779
        trlbegin = sample(j) + pretrig;
        trlend   = sample(j) + posttrig;
        offset   = pretrig;
        newtrl   = [trlbegin trlend offset 1];
        trl      = [trl; newtrl];
    end  
end
end