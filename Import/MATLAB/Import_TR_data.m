function Import_TR_data(Basepath,subj, CondIdx)
%% Import WL Data
EEGpath = [Basepath, 'EEG\'];

LckLbl{1} = 'Stim';
LckLbl{2} = 'Resp';

%% Import Rest Data    
for s = subj
        ID = [num2str(CondIdx),sprintf('%03d',s)];       
                       
        % for subject numbers etc
        %%
        cfg = [];
        cfg.headerfile  = [EEGpath, 'TR',num2str(CondIdx),sprintf('%03d',s),'WL.vhdr'];
        cfg.dataset     = [EEGpath, 'TR',num2str(CondIdx),sprintf('%03d',s),'WL.vhdr'];
        cfg.trialfun            = 'WL_eeg_fun_rest';
        cfg.trialdef.pre        = 5;
        cfg.trialdef.post       = 185;    % Window extra large for FreqAnalysis       
        cfg = ft_definetrial(cfg);
        data_eeg = ft_preprocessing(cfg);
        
        if subj > 7
            cfg = [];
            cfg.headerfile  = [EEGpath, 'TR',num2str(CondIdx),sprintf('%03d',s),'B.vhdr'];
            cfg.dataset     = [EEGpath, 'TR',num2str(CondIdx),sprintf('%03d',s),'B.vhdr'];
            cfg.trialfun            = 'WL_eeg_fun_Baseline';
            cfg.trialdef.pre        = 5;
            cfg.trialdef.post       = 180;    % Window extra large for FreqAnalysis
            cfg = ft_definetrial(cfg);
            data_eeg_Baseline = ft_preprocessing(cfg);
            
            cfg = [];
            data_eeg = ft_appenddata(cfg, data_eeg_Baseline, data_eeg );
            
        end 
        
        save([EEGpath, '/Raw/TR_Trl_Rest_',ID,'.mat'],'data_eeg','-v7.3')
        clear data_eeg 
        
        %% Import WL Data    
        cfg = [];
        cfg.headerfile  = [EEGpath, 'TR',num2str(CondIdx),sprintf('%03d',s),'WL.vhdr'];
        cfg.dataset     = [EEGpath, 'TR',num2str(CondIdx),sprintf('%03d',s),'WL.vhdr'];
        cfg.trialfun            = 'WL_eeg_fun';
        cfg.trialdef.pre        = 0.5;
        cfg.trialdef.post       = 2;    % Window extra large for FreqAnalysis       
        cfg = ft_definetrial(cfg);
        data_eeg = ft_preprocessing(cfg);
        
        save([EEGpath, '/Raw/TR_Trl_WL_',ID,'.mat'],'data_eeg','-v7.3')
        
        %% Improt SRT Data
        for Lock = 1:2
            cfg = [];
            cfg.headerfile  = [EEGpath, 'TR',num2str(CondIdx),sprintf('%03d',s),'SRT.vhdr'];
            cfg.dataset     = [EEGpath, 'TR',num2str(CondIdx),sprintf('%03d',s),'SRT.vhdr'];
            if Lock == 1
                cfg.trialdef.pre        = 0.5;
                cfg.trialdef.post       = 1.2;
                cfg.trialfun = 'srt_TR_eeg_fun_stim';
            elseif Lock == 2
                cfg.trialdef.pre        = 0.8;
                cfg.trialdef.post       = 0.6;
                cfg.trialfun = 'srt_TR_eeg_fun_resp';            
            end           
            cfg = ft_definetrial(cfg);
            
            data_eeg    = ft_preprocessing(cfg);
            
            save([EEGpath, '/Raw/TR_Trl_SRT_',ID,'_',LckLbl{Lock},'.mat'],'data_eeg','-v7.3')
        end
end
end

