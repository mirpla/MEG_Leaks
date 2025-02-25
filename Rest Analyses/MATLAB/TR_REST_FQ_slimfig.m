function [FQData]= TR_REST_FQ_slimfig(Basepath, CondIdx, sub, ChanSel, manual)

Cond        = {'Cong','InCong'};
Explabel    = {'Implicit','Explicit'};
col         = {'b','r'};
freqRange(1) = 5;
freqRange(2) = 15;
xwidth  =1280;
ywidth   =720;

transpalpha = 0.7;
%%

pkFreq = zeros(max(sub),10,size(ChanSel,2));
pkFreqLoc = pkFreq;

for s = sub
    base_freq = [];
    ID = [num2str(CondIdx),sprintf('%03d',s)];
    load([Basepath,'EEG/Processed/TR_CLEAN_Rest_',ID,'.mat'],'data_clean')
    
    tnum = size(data_clean.trial,2);
    w = [250, 238, 2]/255;
    b = [8, 7, 05]/255;
    colors_f = [linspace(w(1),b(1),tnum-1)', linspace(w(2),b(2),tnum-1)', linspace(w(3),b(3),tnum-1)', ones(tnum-1,1)*transpalpha];
    colors_p = [1 0 0 1; colors_f];
    %tit      = ['Sub_',num2str(sub)];
    %%
    for t = 1:tnum
        
        
        cfg2 = [];
        cfg2.output  = 'pow';
        cfg2.channel = 'all';
        cfg2.method  = 'mtmfft';
        cfg2.taper   = 'hanning';
        cfg2.foi     = 0.5:0.5:45; % 1/cfg1.length  = 2;
        base_freq{t}   = ft_freqanalysis(cfg2, base_rpt);
        
        [pks,locs] = findpeaks(mean(base_freq{t}.powspctrm(chan,freqzoom)));
        if ~isempty(pks((locs > freqRange(1)) & (locs < freqRange(2))))
            pkFreq(s,t,figC)     = max(pks((locs > freqRange(1)) & (locs < freqRange(2))));
            pkFreqLoc(s,t,figC)  = locs(pks == max(pks((locs > freqRange(1)) & (locs < freqRange(2)))));
        end
    end
    
    FQData.Spctrm{s}   = base_freq;
end

FQData.freq     = pkFreq;
FQData.loc      = pkFreqLoc; 
end
