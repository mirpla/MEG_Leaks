Start_up_MEG

%% SRT Plots 

[s,r] = SRT_Explicitness(base_path);

SRT_Analysis(base_path)

%% WL behavioral plots

%SRT_Explicitness
[perf.SR,perf.ED] = WL_Performance(base_path);

%% Importing of the EEG Data

subj{1}        = [10];    % Include subjects to be analysed
subj{2}        = [];
CondIdx     = [1];        % Condition index
% First round of artifact Rejection
DataType    = 1; % DataType 1 = WL Rest, 2 = WL Encoding/Retrieval 3 = SRT Response Locked, 4 = SRT Stimulus locked

for c = CondIdx  
    Import_TR_data(Basepath,subj{c},c);
end
%%
% Early Processing 
for c = CondIdx
    TR_EEGPreProcArtf(Basepath, DataType, subj{c}, c)
    TR_EEG_ICA(Basepath, DataType, subj{c}, c)
end

%%
rejcomp{1}{1,1}     = [1 2 3 7];
rejcomp{1}{1,2}     = [1 3 12 21]; % alpha in 2 and 6
rejcomp{1}{1,3}     = [1 2 3 4 5 8 13];
rejcomp{1}{1,5}     = [1 2 3 8];
rejcomp{1}{1,6}     = [1 3 5 6 7 9];
rejcomp{1}{1,7}     = [1 3 4 5 7 12 14];
rejcomp{1}{1,8}     = [1 4 5 7];
rejcomp{1}{1,9}     = [1 4 7 9 10 16 19];
rejcomp{1}{1,10}    = [1 2 3 4 5 6 ];

TR_postICA(Basepath, [subj{c}], c,DataType, rejcomp{DataType})

%% REST Anaylses

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

c = 1;
TR_REST_PP(Basepath,sort([Inclusion{1}',Inclusion{2}']),c,0)

[RestFQ{c}] = TR_REST_FQ(Basepath, c,sort([Inclusion{1}',Inclusion{2}']),ChanSel,0);

%%
