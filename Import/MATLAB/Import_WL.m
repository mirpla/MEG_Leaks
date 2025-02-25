%function Import_WL

% for subject numbers etc
%%

datasetname{1,1} = '\\raw\Project0407\rmk06\WL\WL2.wav';
datasetname{1,2} = '\\raw\Project0407\rmk06\240614\MEG_2014_WL2.fif';

datasetname{2,1} = '\\raw\Project0407\sce17\WL\WL2.wav';
datasetname{2,2} = '\\raw\Project0407\sce17\240614\MEG_1013_WL2.fif';

datasetname{3,1} = '\\raw\Project0407\tva26\WL\WL2.wav';
datasetname{3,2} = '\\raw\Project0407\tva26\240618\MEG_2015_WL2.fif';

datasetname{4,1} = '\\raw\Project0407\jmr19\\WL\WL2.wav';
datasetname{4,2} = '\\raw\Project0407\jmr19\240619\MEG_2016_WL2.fif';

datasetname{5,1} = '\\raw\Project0407\ebd10\WL\WL2.wav';
datasetname{5,2} = '\\raw\Project0407\ebd10\240620\MEG_2017_WL2.fif';

datasetname{6,1} = '\\raw\Project0407\gaa26\WL\WL2.wav';
datasetname{6,2} = '\\raw\Project0407\gaa26\240621\MEG_2018_WL2.fif';

datasetname{7,1} = '\\raw\Project0407\nce28\WL\WL2.wav';
datasetname{7,2} = '\\raw\Project0407\nce28\240621\MEG_1019_WL2.fif';

for x = 1:7
    [audio{x}.track, audio{x}.FS]= audioread(datasetname{x,1});
    cfg = [];
    cfg.dataset     = [datasetname{x,2}];
    cfg.trialfun            = 'WL_MEG_fun';
    cfg.trialdef.pre        = 0.5;
    cfg.trialdef.post       = 1.5;    % Window extra large for FreqAnalysis
    cfg = ft_definetrial(cfg);
    data_meg = ft_preprocessing(cfg);
    data_meg_full = ft_read_data(datasetname{x,2});
    
    audiochan = find(strcmp(data_meg.label, 'MISC007'));
    %%
    figure
    audiox = (1:length(audio{x}.track))/audio{x}.FS;
    megmax = max(audiox)*data_meg.fsample;
    megx = (1:megmax)/data_meg.fsample;
    tiledlayout(2,1)
    nexttile
    plot(audiox,audio{x}.track(:,1)')
    title('Audio recording')
    
    nexttile
    plot(megx,data_meg_full(audiochan,1:megmax))
    title(['Corresponding MEG | FS :', num2str(data_meg.fsample)])
    xlabel('time in s')
    
    meg_audio{x}.track = megx,data_meg_full(audiochan,:);
    meg_audio{x}.fs = data_meg.fsample
end
%%

for x = 5
    [audio{x}.track, audio{x}.FS]= audioread(datasetname{x,1});
    cfg = [];
    cfg.dataset     = [datasetname{x,2}];
    cfg.trialfun            = 'WL_MEG_fun';
    cfg.trialdef.pre        = 0.5;
    cfg.trialdef.post       = 1.5;    % Window extra large for FreqAnalysis
    cfg = ft_definetrial(cfg);
    data_meg = ft_preprocessing(cfg);
    data_meg_full = ft_read_data(datasetname{x,2});
    
    otherchan{1} = find(strcmp(data_meg.label, 'MISC005'));
    otherchan{2} = find(strcmp(data_meg.label, 'MISC006'));
    %%
    minmisc = 20
    maxmisc = 25
    figure
    audiox = (1:length(audio{x}.track))/audio{x}.FS;
    megmax = max(audiox)*data_meg.fsample;
    megx = (1:megmax)/data_meg.fsample;
    tiledlayout(2,1)
    nexttile
    plot(megx((minmisc*data_meg.fsample):(maxmisc *data_meg.fsample)),data_meg_full(otherchan{2},(minmisc*data_meg.fsample):(maxmisc *data_meg.fsample)))
    title('MISC006')
    
    nexttile
    plot(megx((minmisc*data_meg.fsample):(maxmisc *data_meg.fsample)),data_meg_full(audiochan,(minmisc*data_meg.fsample):(maxmisc *data_meg.fsample)))
    title(['Corresponding MEG | FS :', num2str(data_meg.fsample)])
    xlabel('time in s')
      
    
end
%%
for x = 1:12
    sound(data_meg.trial{1,x}(320,:),data_meg.fsample)
    pause(2)
end 

%%
sound(data_meg.trial{1,x}(320,:),data_meg.fsample)
sound(audio{x}.track,audio{x}.FS)

%end
