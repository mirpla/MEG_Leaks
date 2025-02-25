%% Params
ChanSel{1,1}.Lbl        = {'AFz','AF3','AF4','F1','F2','Fz'};
ChanSel{1,1}.Title      = 'Frontal Response';
ChanSel{1,1}.Savefile   = 'FrontChans';

ChanSel{1,2}.Lbl        = {'Cz','C2','C1'};
ChanSel{1,2}.Title      = 'Central Response';
ChanSel{1,2}.Savefile   = 'CentChans';

ChanSel{1,3}.Lbl        = {'Pz','P1','P2'};
ChanSel{1,3}.Title      = 'Parietal Response';
ChanSel{1,3}.Savefile   = 'ParChans';

ChanSel{1,4}.Lbl        = {'Oz','O2','O1','POz','PO3','PO4'};
ChanSel{1,4}.Title      = 'Occipital Response';
ChanSel{1,4}.Savefile   = 'OccChans';

ChanSel{1,5}.Lbl        = {'T7','TP7','C5','CP5'};
ChanSel{1,5}.Title      = 'Temporal 1 Response';
ChanSel{1,5}.Savefile   = 'TLChans';

ChanSel{1,6}.Lbl        = {'T8','TP8','C6','CP6'};
ChanSel{1,6}.Title      = 'Temporal R Response';
ChanSel{1,6}.Savefile   = 'TRChans';


col{1} = 'bo';
col{2} = 'ro';
col{3} = 'go';
col{4} = 'yo';
 
o = [4,3,2,1];

%% Loop for different Areas
data = RestFQ{c}.freq;
dataspec = RestFQ{c}.Spctrm;
for figC = 1:6
    %% Figure Data Prep          
    for exp = 1:2       
        for s = Inclusion{exp}'
            chan = find(ismember(dataspec{1,s}{1,1}.label', ChanSel{1,figC}.Lbl));
            if s >= 8
                trialsel = 3:12;
                preWL = 2;
                preTask = 1;
            else
                trialsel = 2:11;
                preWL = 1;
                clear preTask
            end
            
            y(s,1) = mean(mean(data{1,s}(find(perf.SR(s,:) == 12)+preWL,chan)/max(max(data{1,s}(:,chan))))); % Correct
            y(s,2) = mean(mean(data{1,s}(find(perf.SR(s,:) ~= 12)+preWL,chan)/max(max(data{1,s}(:,chan))))); % Incorrect
            y(s,3) = data{1,s}(preWL,chan)/max(data{1,s}(:,chan)); % Pre WL baseline
            if exist('preTask','var')
                y(s,4) = mean(data{1,s}(preTask,chan)./max(data{1,s}(:,chan))); % Pre Task Baseline
                clear preTask
            end

            eb(s,1) = std(mean(data{1,s}(find(perf.SR(s,:) == 12)+preWL,chan)/max(max(data{1,s}(:,chan)))));
            eb(s,2) = std(mean(data{1,s}(find(perf.SR(s,:) ~= 12)+preWL,chan)/max(max(data{1,s}(:,chan)))));
        end
    end
    %% Actual Figure
    
    figure
    tiledlayout(1,5)
    nexttile([1,4])
    x = 1:10;
        
    hold on
    errorbar(x',y(:,1),eb(:,1),'bo')
    errorbar(x',y(:,2),eb(:,2),'ro')
    plot(x',y(:,3),'go')
    plot(x',y(:,4),'yo')
    hold off
    
    xlim([0 11])
    ylim([0.1 1])
    xticks([1 2 3 4 5 6 7 8 9 10])
    xlabel('Subjects')
    ylabel('Normalized Power')
    title(['Individual Power Values', ChanSel{1,figC}.Title])
    %legend('Correct','Incorrect','PreWL Baseline', 'PreSRT Baseline')
    
    nexttile
    hold on
    for p = 1:4
        errorbar(o(p),mean(nonzeros(y(:,p))), std(nonzeros(y(:,p)))/sqrt(length(data)),col{p})
    end
    xlim([0 5])
    ylim([0.1 1])
    xticklabels([])
    title('Mean Alpha Power Values')
    legend('Learned','not Learned','PreWL Baseline', 'PreSRT Baseline','Location','southeast')
    
end

%% Loop for different Areas
data = RestFQ{c}.exp;
dataspec = RestFQ{c}.Spctrm;

for figC = 1:6
    %% Figure Data Prep          
    for exp = 1:2       
        for s = Inclusion{exp}'
            chan = find(ismember(dataspec{1,s}{1,1}.label', ChanSel{1,figC}.Lbl));
            zdata{1,s} = (data{1,s}-mean(data{1,s},1))./std(data{1,s});
            if s >= 8
                trialsel = 3:12;
                preWL = 2;
                preTask = 1;
            else
                trialsel = 2:11;
                preWL = 1;
                clear preTask
            end
            
            y(s,1) = mean(mean(zdata{1,s}(find(perf.SR(s,:) == 12)+preWL,chan))); % Correct
            y(s,2) = mean(mean(zdata{1,s}(find(perf.SR(s,:) ~= 12)+preWL,chan))); % Incorrect
            y(s,3) = mean(zdata{1,s}(preWL,chan)); % Pre WL baseline
            if exist('preTask','var')
                y(s,4) = mean(zdata{1,s}(preTask,chan)); % Pre Task Baseline
                clear preTask
            end

            eb(s,1) = std(mean(zdata{1,s}(find(perf.SR(s,:) == 12)+preWL,chan)));
            eb(s,2) = std(mean(zdata{1,s}(find(perf.SR(s,:) ~= 12)+preWL,chan)));
        end
    end
    %% Actual Figure
    
    figure
    tiledlayout(1,5)
    ax1 = nexttile([1,4]);
    x = 1:10;
        
    hold on
    errorbar(x',y(:,1),eb(:,1),'bo')
    errorbar(x',y(:,2),eb(:,2),'ro')
    plot(x',y(:,3),'go')
    plot(x',y(:,4),'yo')
    hold off
    
    xlim([0 11])
    %ylim([0.1 1])
    xticks([1 2 3 4 5 6 7 8 9 10])
    xlabel('Subjects')
    ylabel('Normalized Offset')
    title(['Individual Offset Values', ChanSel{1,figC}.Title])
    %legend('Correct','Incorrect','PreWL Baseline', 'PreSRT Baseline')
    
    ax2 = nexttile;
    hold on
    for p = 1:4
        errorbar(o(p),mean(nonzeros(y(:,p))), std(nonzeros(y(:,p)))/sqrt(length(data)),col{p})
    end
    xlim([0 5])
    %ylim([0.1 1])
    xticklabels([])
    title('Mean Offest Values')
    legend('Learned','not Learned','PreWL Baseline', 'PreSRT Baseline','Location','southeast')
    linkaxes([ax1, ax2],'y')
end


%% New Sum 
data = RestFQ{c}.off;
dataspec = RestFQ{c}.Spctrm;

for figC = 1:6
    %% Figure Data Prep          
    for exp = 1:2       
        for s = Inclusion{exp}'
            chan = find(ismember(dataspec{1,s}{1,1}.label', ChanSel{1,figC}.Lbl));
            zdata{1,s} = (data{1,s}-mean(data{1,s},1))./std(data{1,s});
            if s >= 8
                trialsel = 3:12;
                preWL = 2;
                preTask = 1;
            else
                trialsel = 2:11;
                preWL = 1;
                clear preTask
            end
            
            y(s,1) = mean(mean(zdata{1,s}(find(perf.SR(s,:) == 12)+preWL,chan))); % Correct
            y(s,2) = mean(mean(zdata{1,s}(find(perf.SR(s,:) ~= 12)+preWL,chan))); % Incorrect
            y(s,3) = mean(zdata{1,s}(preWL,chan)); % Pre WL baseline
            if exist('preTask','var')
                y(s,4) = mean(zdata{1,s}(preTask,chan)); % Pre Task Baseline
                clear preTask
            end

            eb(s,1) = std(mean(zdata{1,s}(find(perf.SR(s,:) == 12)+preWL,chan)));
            eb(s,2) = std(mean(zdata{1,s}(find(perf.SR(s,:) ~= 12)+preWL,chan)));
        end
    end
    %% Actual Figure
    
    figure
    tiledlayout(1,5)
    ax1 = nexttile([1,4]);
    x = 1:10;
        
    hold on
    errorbar(x',y(:,1),eb(:,1),'bo')
    errorbar(x',y(:,2),eb(:,2),'ro')
    plot(x',y(:,3),'go')
    plot(x',y(:,4),'yo')
    hold off
    
    xlim([0 11])
    %ylim([0.1 1])
    xticks([1 2 3 4 5 6 7 8 9 10])
    xlabel('Subjects')
    ylabel('Normalized Exponent')
    title(['Individual Exponent Values', ChanSel{1,figC}.Title])
    %legend('Correct','Incorrect','PreWL Baseline', 'PreSRT Baseline')
    
    ax2 = nexttile;
    hold on
    for p = 1:4
        errorbar(o(p),mean(nonzeros(y(:,p))), std(nonzeros(y(:,p)))/sqrt(length(data)),col{p})
    end
    xlim([0 5])
    %ylim([0.1 1])
    xticklabels([])
    title('Mean Exponent Values')
    legend('Learned','not Learned','PreWL Baseline', 'PreSRT Baseline','Location','southeast')
    linkaxes([ax1, ax2],'y')
end