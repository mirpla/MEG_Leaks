function POD_SRT_Analysis(Basepath, SubjIdx,SubInfo,SRTflag)
%% SRT
ImpExp{1} = 'Implicit'; 
ImpExp{2} = 'Explicit';
NNidx{1}  = 'Native';
NNidx{2}  = 'NonNative';

if SRTflag == 1
    %% Parameters
    % Results(Trial, 1)     = Subject ID;
    % Results(Trial, 2)     = Block Number
    % Results(Trial, 3)     = Trial Number
    % Results(Trial, 4)     = Correct Trial
    % Results(Trial, 5)     = Task Code (random or Sequence)
    % Results(Trial, 6:9)   = Response Times
    % Results(Trial, 10)    = Sequence File Used
    
    % hp/lp cut-off parameters
    % RTlimit(1)   = 0.01;   % minimum reaction time for a trial in seconds
    % RTlimit(2)   = 1.2;   % maximum reaction time for a trial in seconds
    
    fitLimit(1) = 0.24; % alpha value for LOWESS fitting for the random trials
    fitLimit(2) = 0.14; % alpha value for LOWESS fitting for the experimental trials
    
    ID.Random   = 82;  % Task Code of Random trials in Logfiles
    ID.Sequence = 84;  % Task Code of Main Sequence trials in Logfiles
    
    SLWindow    = 50;
    [SRT.SL, SRT.data, SRT.fitLimit, SRT.Error,SRT.Errorrate, SRT.ID, SRT.Cond]  = SRTImportFit(Basepath,fitLimit,SLWindow,ID);
    save([Basepath, '/Log/SRT/BehavSRT.mat'],'SRT','-v7.3');
else
    load([Basepath, '/Log/SRT/BehavSRT.mat'],'SRT');
end
%% 

xa  = 1:size(SRT.SL,2);
bgm{1,1} = [0.8,0.4,0.4];
bgm{1,2} = [0.4,0.4,0.8];
bgm{2,1} = [0.5,0.0,0.0];
bgm{2,2} = [0.0,0.0,0.5];

for NN = 1:2
    figure
    for ImpExp = 1:2 % separate figure for implicit and explicit
        subplot(1,2,ImpExp);
        hold on
        clear handl
        for ConInc = 1:2
            dum = SRT.SL(SRT.Cond == ConInc,:);
            %title(FigTits{1,figcnt})           
            for x = SubjIdx{NN,ImpExp}{1,ConInc}'
                handl{ConInc,x} = plot(xa,dum(x,:),'o-','linewidth',1.5,'markersize',3,'Color',bgm{1,ConInc});
            end
            xticks(xa)
            if  size(dum(SubjIdx{NN,ImpExp}{1,ConInc}',:),1) ~= 1 % only make errorbars if group has more than 1 subject
                errorbar(xa,mean(dum(SubjIdx{NN,ImpExp}{1,ConInc}',:),1),std(dum(SubjIdx{NN,ImpExp}{1,ConInc}',:),1),'x-','linewidth',4,'markersize',5, 'Color',bgm{2,ConInc})
            end
        end
        legend([handl{1,find(~cellfun(@isempty,handl(1,:)),1)},handl{2,find(~cellfun(@isempty,handl(2,:)),1)}], 'Congruent', 'Incongruent');
        hold off
        ylim([-0.04 0.16])
    end
    %linkaxes
    saveas(gcf,[Basepath, 'Log\SRT\Fig\LearningRate',NNidx{NN},'.png'])
end

%% Individual SRT Performances
Cond        = {'Cong','InCong'};
for x = 1:size(SRT.data,1)
    IDnum = SubInfo{x,1};
    %c = (IDnum - mod(IDnum, 1000)) / 1000
    SRTfit =[]; 
    for bl = 1:size(SRT.data,2)
        for fl = 2       
            SRTfit  = [SRTfit;SRT.data{x,bl}{fl}(:,3)];           
        end 
    end
    figure
    title(['Subject ', num2str(IDnum)])
    subplot(1,2,1)
    normplot(SRTfit)
    subplot(1,2,2)
    qqplot(SRTfit)
    saveas(gcf,[Basepath, 'Log\SRT\Fig\DistPlot_', num2str(IDnum),'.png'])
    SRTfit = [];
    SRTdev = [];
    %     SRTlim = cell(1,2);
    %     LimxVec = 0;
    for bl = 1:size(SRT.data,2)
        for fl = 1:size(SRT.data{x,bl},2)
            SRTfit  = [SRTfit;SRT.data{x,bl}{fl}(:,3)];
            SRTdev  = [SRTdev;ones(size(SRT.data{x,bl}{fl}(:,2),1),1)*std(SRT.data{x,bl}{fl}(:,2))];
            %             for limdir = 1:2
            %                 SRTlim{limdir} = [SRTlim{limdir}; [SRT.fitLimit{x,bl}{fl}{limdir}(:,1)+LimxVec, SRT.fitLimit{x,bl}{fl}{limdir}(:,2)]];
            %             end
            %            LimxVec = size(SRTfit,1);
        end
    end 
    
    figure
    hold on
    h = scatterfigI(SRT.data(x,:));   
    h.FitFig = plot(SRTfit,'k','LineWidth',5);
    h.FitDevB = plot(SRTfit+SRTdev*3,':k','LineWidth',1.5);
    plot(SRTfit-SRTdev*3,':k','LineWidth',1.5);
    h.FitDevN = plot(SRTfit+SRTdev*2,'--k','LineWidth',3);
    plot(SRTfit-SRTdev*2,'--k','LineWidth',3);
    title(['Subject ', num2str(IDnum)])
    legend([h.scat{1,1},h.scat{1,2},h.scat{1,3} h.FitFig,h.FitDevN, h.FitDevB],'Pre-Random','Sequence','Post-Random','LOWES fit', '2 x StdDev', '3 x StdDev')
    hold off
    ylim([0 2])
    saveas(gcf,[Basepath, 'Log\SRT\Fig\RT_', num2str(IDnum),'.png'])
end
close all

%% Average Implicit vs Explicit Scatterfig
% ImExptit{1} = {'Implicit'};
% ImExptit{2} = {'Explicit'};
% figure
% for ImpExp = 1:2   
%     subplot(1,2,ImpExp)
%     dumcond = [];
%     hold on
%     for ConInc = 1:2
%         dum     = SRT.SL(SRT.Cond == ConInc,:);
%         dumcond = [dumcond;SubjIdx{1,ImpExp}{1,ConInc}];
%     end
%     scatterfig(SRT.correct,dumcond')
%     title(ImExptit{ImpExp})
%     hold off
%     linkaxes
% end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function scatterfig(Data,subj)
clrs = [27,158,119; 240,80,2; 117,112,179]/255;
xax = 0;

for b = 1:size(Data,2)
    for f = 1:3
        for s = subj
            figdum{b}{f}(s,:) = Data{s,b}{f};
        end
        xax =  1+max(xax):length([figdum{b}{f}])+max(xax);
        scatter(xax,mean(figdum{b}{f},'omitnan' ),[],clrs(f,:),'filled')              
    end
end
ylabel('RT in s')
xlabel('Trial')
end 

function h = scatterfigI(Data)
clrs = [27,158,119; 240,80,2; 117,112,179]/255;
xax = 0;

for b = 1:size(Data,2)
    for f = 1:3
        figdum{b}{f}(1,:) = Data{1,b}{f}(:,2);
        xax =  1+max(xax):length([figdum{b}{f}])+max(xax);
        h.scat{b,f} = scatter(xax,figdum{b}{f},[],clrs(f,:),'filled');            
    end
end
ylabel('RT in s')
xlabel('Trial')
end 
end 