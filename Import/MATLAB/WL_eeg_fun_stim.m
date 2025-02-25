function [trl, event] = POD_eeg_fun_stim(cfg)

% read the header information and the events from the data
hdr   = ft_read_header(cfg.dataset);
event = ft_read_event(cfg.dataset);

% search for "trigger" events
value  = {event(find(strcmp('Stimulus', {event.type}))).value}';
sample = [event(find(strcmp('Stimulus', {event.type}))).sample]';

% determine the number of samples before and after the trigger
pretrig  = -round(cfg.trialdef.pre  * hdr.Fs);
posttrig =  round(cfg.trialdef.post * hdr.Fs);

pretrig2  =  -round(cfg.trialdef.pre  * hdr.Fs);
posttrig2 =   round(cfg.trialdef.post-60  * hdr.Fs);

% look for the combination of a trigger "7" followed by a trigger "6"
% for each trigger except the last one 
% and look for "12" for recall trials
trl = [];
first = 0;
for j = 1:(length(value)-1)
    trg1 = value{j};
    trg2 = value(j+1);
    if first == 0 % First rest period is only 2 min and has no trigger so use the first two minutes before trial onset to get that one
         if strcmp(trg2,  'S  6')
            trlbegin = sample(j+1) - posttrig2 + pretrig2+ pretrig2;
            trlend   = sample(j+j);
            offset   = pretrig2;
            newtrl   = [trlbegin trlend offset 1];
            trl      = [trl; newtrl];
         end 
    else
        if strcmp(trg1,  'S 14')
            trlbegin = sample(j) + pretrig;
            trlend   = sample(j) + posttrig;
            offset   = pretrig;
            newtrl   = [trlbegin trlend offset 1];
            trl      = [trl; newtrl];
        end
    end
end