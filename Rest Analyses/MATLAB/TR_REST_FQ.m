function [FQData]= TR_REST_FQ(Basepath, CondIdx, sub, ChanSel, FigureFlag)

Cond        = {'Cong','InCong'};
Explabel    = {'Implicit','Explicit'};
col         = {'b','r'};
freqRange(1) = 5;
freqRange(2) = 15;
xwidth  =1280;
ywidth   =720;

transpalpha = 0.7;
%%
for s = sub
    base_freq = [];
    ID = [num2str(CondIdx),sprintf('%03d',s)];
    load([Basepath,'EEG/Processed/TR_pp_Rest_',ID,'.mat'],'base_r_pp')
    
    tnum = max(base_r_pp.trialinfo(:,2));
    w = [250, 238, 2]/255;
    b = [8, 7, 05]/255;
    colors_f = [linspace(w(1),b(1),tnum-1)', linspace(w(2),b(2),tnum-1)', linspace(w(3),b(3),tnum-1)', ones(tnum-1,1)*transpalpha];
    colors_p = [1 0 0 1; colors_f];
    %tit      = ['Sub_',num2str(sub)];
    %%
    for t = 1:tnum
        
        cfg = [];
        cfg.output  = 'pow';
        cfg.channel = 'all';
        cfg.method  = 'mtmfft';
        cfg.taper   = 'hanning';
        cfg.foi     = 0.5:0.5:45; % 1/cfg1.length  = 2;
        cfg.trials  = base_r_pp.trialinfo(:,2) == t;
        base_freq{t}   = ft_freqanalysis(cfg, base_r_pp);
        
        % compute the fractal and original spectra
        cfg               = [];
        cfg.foilim        = [1 45];
        cfg.pad           = 4;
        cfg.tapsmofrq     = 2;
        cfg.method        = 'mtmfft';
        cfg.output        = 'fooof_aperiodic';
        cfg.trials  = base_r_pp.trialinfo(:,2) == t;
        freq_fract{t}     = ft_freqanalysis(cfg,  base_r_pp );
        cfg.output        = 'pow';
        freq_orig{t}      = ft_freqanalysis(cfg,  base_r_pp );
        
        % subtract the fractal component from the power spectrum
        cfg               = [];
        cfg.parameter     = 'powspctrm';
        cfg.operation     = 'x2-x1';
        freq_osci{t,1}      = ft_math(cfg, freq_fract{t} , freq_orig{t});
        
        % original implementation by Donoghue et al. 2020 operates through the semilog-power
        % (linear frequency, log10-power) space and transformed back into linear-linear space.
        % thus defining an alternative expression for the oscillatory component as the quotient of
        % the power spectrum and the fractal component
        cfg               = [];
        cfg.parameter     = 'powspctrm';
        cfg.operation     = 'x2./x1';  % equivalent to 10^(log10(x2)-log10(x1))
        freq_osci{t,2} = ft_math(cfg, freq_fract{t}, freq_orig{t});
        
        aparam          = vertcat(freq_fract{t}.fooofparams.aperiodic_params);     
        Offset{s}(t,:)     = aparam(:,1);
        Exponent{s}(t,:)   = aparam(:,2); 
        
        for chan = 1:size(freq_osci{t,2}.label,1)
            [pks,locs] = findpeaks(freq_osci{t,2}.powspctrm(chan,:),'MinPeakProminence',0.05);
            
            if ~isempty(pks(( freq_osci{t,2}.freq(locs) > freqRange(1)) & ( freq_osci{t,2}.freq(locs) < freqRange(2))))
                pkFreq{s}(t,chan)     = max(pks(( freq_osci{t,2}.freq(locs) > freqRange(1)) & ( freq_osci{t,2}.freq(locs) < freqRange(2))));
                pkFreqLoc{s}(t,chan)  = freq_osci{t,2}.freq(locs(pks == max(pks(( freq_osci{t,2}.freq(locs) > freqRange(1)) & ( freq_osci{t,2}.freq(locs) < freqRange(2))))));
            end
        end
    end
    
    if FigureFlag == 1
        for figC = 1:6
            chan = find(ismember(base_r_pp.label', ChanSel{1,figC}.Lbl));
            figure('Position',[0,0,xwidth,ywidth]);
            hold on;
            for t = 1:tnum
                if t == 1
                    lt = 3;
                else
                    lt = 1;
                end
                plot(freq_osci{t,2}.freq(:)', mean( freq_osci{t,2}.powspctrm(chan,:)),'Color',colors_p(t,:),'LineWidth',lt)
               
            end
            xlim
            xlabel('Frequency (Hz)');
            ylabel('power (uV^2)');
            title([ChanSel{1,figC}.Title])
            saveas(gcf,[Basepath, 'EEG\Processed\Fig\Freq\SpectrumSub', num2str(s),'_', Cond{CondIdx},'_', ChanSel{1,figC}.Savefile,'.png'])
        end
    end
    FQData.Spctrm{s}   = base_freq;
end
FQData.off      = Offset;
FQData.exp      = Exponent;
FQData.freq     = pkFreq;
FQData.loc      = pkFreqLoc; 
end
