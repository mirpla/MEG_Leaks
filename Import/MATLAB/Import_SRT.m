function Import_SRT(base_path, sub,ses,Lock)
% Import SRT EEG Data
%% Import SRT Data    
if Lock == 1   
    LckLbl = 'Stim';
elseif Lock == 0
    LckLbl = 'Resp';
end
%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!Change this for real subjects%%!!!!!!%
srt_data_path = [base_path, 'Data\sub-',sub,'\ses-',num2str(ses),'\meg\sub-',sub,'_ses-',num2str(ses),'_task-SRT_run-1_meg.fif'];

cfg = [];
cfg.dataset     = [srt_data_path];
if Lock == 1
    cfg.trialdef.pre        = 0.5;
    cfg.trialdef.post       = 1.2;
    cfg.trialfun = 'srt_MEG_fun_stim';
elseif Lock == 0
    cfg.trialdef.pre        = 0.8;
    cfg.trialdef.post       = 0.6;
    cfg.trialfun = 'srt_MEG_fun_resp';
else
    error('invalid Stimulus Locking selected, please select 1 for Stimulus onset lock and 0 for Responselock');
end

cfg = ft_definetrial(cfg);

data_eeg    = ft_preprocessing(cfg);
save([EEGpath,'/Raw/',LckLbl,'/SRT_Trl_',ID,'.mat'],'data_eeg','-v7.3')
