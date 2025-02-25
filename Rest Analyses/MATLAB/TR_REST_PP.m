function TR_REST_PP(Basepath,sub,CondIdx,manual)
EEGpath = [Basepath,'EEG\Processed\'];
for s = sub
    ID = [num2str(CondIdx),sprintf('%03d',s)];
    load([EEGpath, 'TR_CLEAN_Rest_',ID,'.mat'],'data_clean')
    
    if manual ~= 1
        Artf = readmatrix([EEGpath,'Rest\RestArt',ID,'.csv']);
    end
    
    tnum = size(data_clean.trial,2);
    %%
    base_rpt = cell(tnum,1);
    for t = 1:tnum
        disp(['Subject ',num2str(s), '; Trial ',num2str(t)]);
        cfg = [];
        cfg.trials  = t;
        data_single = ft_selectdata(cfg, data_clean);
        
        cfg = [];
        cfg.overlap = 0;
        cfg.length  = 2;
        base_dat    = ft_redefinetrial(cfg, data_single);
        
        if manual == 1
            cfg = [];
            cfg.layout = 'easycapM11.lay'; % specify the layout file that should be used for plotting
            cfg.viewmode = 'vertical';
            ft_databrowser(cfg,  base_dat)
            
            cfg             = [];
            cfg.method      = 'summary';
            base_rpt{t}    = ft_rejectvisual(cfg, base_dat);            
        else
            TBD = sort(Artf(t,~isnan(Artf(t,:))));
            cfg = [];
            cfg.trials      = find(~ismember((1:size(base_dat.trial,2)),TBD));
            base_rpt{t}     = ft_selectdata(cfg, base_dat);
        end
         
    end  
    basedum = base_rpt(~cellfun(@isempty,base_rpt))';

    base_r_pp = ft_appenddata([], basedum{:});
    save([Basepath,'EEG/Processed/TR_pp_Rest_',ID,'.mat'],'base_r_pp','-v7.3')
end 