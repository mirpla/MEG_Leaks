function [SkillLearning, SkillLearningH, fcRT, hcRT, ErrorRate]  = SRTProcessing(LogPath,RTlimit,h,ID)
% Preprocessing Script that loads in all subjects and rejects implausible
% reaction times

totblocks   = 6; % total number of blocks used in the experiment

[FilesData] = dir([LogPath,'*.txt']);
subnum = zeros(1,size(FilesData,1));
for ind = 1:size(FilesData,1)
    dum = FilesData(ind).name;
    subnumC = extractBetween(dum,'p','_SRTT');
    subnum(ind) = str2double(subnumC{1});
end

subs = unique(subnum);
subs(subs == 0) = [];

for s = subs
    lString{2} = sprintf('TMS-MEPs_p%d_SRTT_blocks12',s);
    lString{3} = sprintf('TMS-MEPs_p%d_SRTT_blocks345',s);
    lString{1} = sprintf('TMS-MEPs_p%d_SRTT_block6',s);
    
    for b = 1:totblocks
        files       = dir(LogPath);
        dum         = {files.name};
        df           = dum(contains(dum, sprintf('TMS-MEPs_p%d_SRTT_block',s)))';
        d           = erase(df,'_SameRoom');
        d           = erase(d,'_B2B');
        for didx    = 1:length(d)
            if contains(d{didx}(:,end-6:end-4),num2str(b)) % split the data into separate blocks and separate Random pre (RT{...}{1}), Exp (RT{...}{1}, and Random Post (RT{...}{1})
                AllB        = readmatrix([LogPath,df{contains(d,lString{didx})}]);
                if s == 3 && didx == 3
                    R{s,b}      = AllB(AllB(:,2) == b-2,:);
                elseif s == 3 && didx == 1
                    R{s,b}      = AllB(AllB(:,2) == b-3,:);
                else 
                    R{s,b}      = AllB(AllB(:,2) == b,:);
                end
                trialID     = R{s,b}(:,5);
                RandDum     = R{s,b}(trialID == ID.Random,:);
                RT{s,b}{1}  = RandDum(1:end/2,:);
                RT{s,b}{3}  = RandDum(end/2+1:end,:);
                RT{s,b}{2}  = R{s,b}(trialID == ID.Sequence,:);
            end
        end
        for f = 1:size(RT{s,b},2)
            for trls = 1:size(RT{s,b}{f},1)
                cRT{s,b}{f}(trls)       = RT{s,b}{f}(trls,5+RT{s,b}{f}(trls,4));
                if size(find(RT{s,b}{f}(trls,6:9)),2)>1
                    eTRL{s,b}{f}(trls,:)= RT{s,b}{f}(trls,6:9);
                else
                    eTRL{s,b}{f}(trls,:)= [0 0 0 0];
                end
            end
            ErrorRate{f}(s,b) = sum(any(eTRL{s,b}{f},2))/size(any(eTRL{s,b}{f},2),1);
           
            
            % filter out trials too fast or too slow
            rejT{s,b}{f}            = (cRT{s,b}{f}<RTlimit(1)) | (cRT{s,b}{f}>RTlimit(2));
            fcRT{s,b}{f}            = cRT{s,b}{f};
            fcRT{s,b}{f}(rejT{s,b}{f}) = NaN;
            
            % filter out outliers with the hampel filter
            [~, hIdx]               = hampel(fcRT{s,b}{f},h.Neigh,h.SD);
            hcRT{s,b}{f}            = fcRT{s,b}{f};
            hcRT{s,b}{f}(hIdx)      = NaN;
            
            hcRTc{s,b}{f}           = rmmissing(hcRT{s,b}{f} );
            fcRTc{s,b}{f}           = rmmissing(fcRT{s,b}{f} );
                        
            allRej{s,b}{f}          = isnan(hcRT{s,b}{f});
            
            fprintf(' %d %%( %d / %d )trials removed for subject %d , block %d\n',round(sum(allRej{s,b}{f})/ length(hcRT{s,b}{f})*100),sum(allRej{s,b}{f}), length(hcRT{s,b}{f}),s,b)
        end
        SkillLearning(s,b) = mean(fcRTc{s,b}{3}) - mean(fcRTc{s,b}{2}(end-(length(fcRTc{s,b}{3})-1):end));
        SkillLearningH(s,b) = mean(hcRTc{s,b}{3}) - mean(hcRTc{s,b}{2}(end-(length(hcRTc{s,b}{3})-1):end));        
    end
end
% SkillLearning([1,5,16,17],:) = [];
% SkillLearningH([1,5,16,17],:) = [];
% hcRT([1,5,16,17],:) = [];
% fcRT([1,5,16,17],:) = [];
end 