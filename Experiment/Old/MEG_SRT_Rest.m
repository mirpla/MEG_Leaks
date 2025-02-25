%function TR_SRT(SubInfo, path, debug, box, startBlock) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SRTT version modified with intermittent Rest %
% periods during which EEG will be recorded by %
%             Mircea van der Plas              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Original version: Andrea Kobor's ASRT script
%contact: kobor.andrea@ttk.mta.hu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prevent spilling of keystrokes into console:
if debug == 0
    ListenChar(-1);
end 
 
% Determine paths for Images and Output
imgpath                 = [path, '/Images/'];
SeqPath                 = [path, '/Sequence_files/'];
savepath.path           = [path, '/Log/SRT/' ];
savepath.restname       = [savepath.path, 'TR_', 'p', num2str(SubInfo.subjID), '_Rest.txt'];

% Stimulus to screen ratio
StimSize    = 1;

if box == 1
    ResponseBoxName = 'NAtA Technologies LxPAD v2014';
    [kbIdx, devName] = GetKeyboardIndices();
    deviceIndex = kbIdx(strcmp(devName, ResponseBoxName));   % not the default keyboard, change according to device you wish to be used
else
    deviceIndex = []; % uses defauly Keyboard if Bottonbox flag is not specified
end
    
StimPath{1,1}   = [imgpath,'master.bmp'];
StimPath{1,2}   = [imgpath,'R1-blue.bmp'];
StimPath{1,3}   = [imgpath,'R2-blue.bmp'];
StimPath{1,4}   = [imgpath,'R3-blue.bmp'];
StimPath{1,5}   = [imgpath,'R4-blue.bmp'];

%Reading sequence file
taskcode    = SubInfo.Cond;
seq         = [SeqPath,'seq', '-', num2str(taskcode), '.txt']; %sequence file to be loaded
alltrials   = dlmread(seq); %this contains all info about the trials
blocknum    = max(alltrials(:,1));

%% Trigger
% Parallel port initialization
base_addr = uint64 (49168);
resettrig = fliplr(logical(dec2bin(0,8)-48)); % send 0 to reset pp
port=uint8(0);

%Use function bin2dec ('10000000') to know the binary code for each pin
Pin4_1 = fliplr(logical(dec2bin  (6,8)-48));    %  s  6 : Indicates the start of a Trial
Pin4_2 = fliplr(logical(dec2bin  (7,8)-48));    %  s  7 : Indicates timestamp of Correct Response
Pin4_3 = fliplr(logical(dec2bin  (12,8)-48));   %  s 12 : Heralds the start of a block   
Pin4_4 = fliplr(logical(dec2bin  (13,8)-48));   %  s 13 : Indicates start and end of the Experiment
Pin4_5 = fliplr(logical(dec2bin  (14,8)-48));   %  s 14 : Ushers the Rest Period

%% set up default settings for Window/Screen
PsychDefaultSetup(2);

Screen('Preference', 'SkipSyncTests', debug ); % Disable calibrations for
% testing purposes, make sure to keep disabled during actual experiment
% (set to 0 in actual experiment)
screens = Screen('Screens');
screenNumber = max(screens); % Select correct Screen here
HideCursor();

white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
grey = white / 2;

windowcolor = white;

if debug == 1
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, windowcolor, [0 0 640 480]);
else
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, windowcolor);
end
% Get screen Dimensions and Centre of Screen
[p.screenwidth, p.screenheight] = Screen('WindowSize', window);

% Get info on frame/refresh rate
ifi = Screen('GetFlipInterval', window);
FRscreen = FrameRate(window);
nominalFR = Screen('NominalFrameRate', window);

% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Retrieve Maximum possible Priority number and give it to othe window
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

% Here we use to a waitframes number greater then 1 to flip at a rate not
% equal to the monitors refreash rate. For this example, once per second,
% to the nearest frame
flipSecs = 1;
waitframes = round(flipSecs / ifi);

RestDur     = 180; % Duraction of Rest Period in s 
rsi         = 0.5; % Interstimulus interval in s
%%
font.size                = 28;
font.name                = 'Arial';
font.color               = black;

Screen('TextSize', window, font.size);
Screen('TextFont', window, font.name);

if box == 1     % Change Key values according to your setup
    Key{1,1} = KbName('6^');
    Key{1,2} = KbName('7&');
    Key{1,3} = KbName('8*');
    Key{1,4} = KbName('9(');
    Key{1,5} = KbName('1!');
else
    Key{1,1} = KbName('h');
    Key{1,2} = KbName('j');
    Key{1,3} = KbName('k');
    Key{1,4} = KbName('l');
    Key{1,5} = KbName('ESCAPE');
end
    

% Prepare all the images used in the Experiment for fastest possible
% presentation
Stimuli         = cellmat(size(StimPath,1)*size(StimPath,2),1);
imageTexture    = Stimuli;
% load in main figures
for totstim = 1:(size(StimPath,1)*size(StimPath,2))
    [Stimuli{totstim}] = imread(StimPath{1,totstim});
    
    % Resize Images
    [s1, s2, s3] = size(Stimuli{totstim});
    aspectRatio = s2 / s1;
    heightScalers = StimSize;
    
    % Make the image into a texturepOD_SRT
    imageTexture{totstim} = Screen('MakeTexture', window, Stimuli{totstim});
end
imageHeights = p.screenheight .* heightScalers;
imageWidths = imageHeights .* aspectRatio;
theRect     = [0 0 imageWidths imageHeights];
StimuliRect = CenterRectOnPointd(theRect, p.screenwidth * 0.5, p.screenheight * 0.5);
%% %%%%%%%%%%% %%%%% %%%%% %%%%%%%%%%% %%
%%%%%%%%%%%%% Experiment %%%%%%%%%%%%%%%%
%%%%%%%%%%%%% %%%%%% %%%%%% %%%%%%%%%%%%%

DrawFormattedText (window, 'Welcome! This experiment is design to test how quickly you can respond to visual cues. \n\n You will be shown 4 circles on a grey screen.\n\nDuring each trial a dot will appear in one of the circles. You need to respond by pressing the corresponding keys:, "m"', 'center', 'center', font.color, '','','', 2);
Screen('Flip', window);

WaitSecs(0.5); %to avoid too short starting time and unwanted button press

ButtonPress(Key{1,5})

StartTime = GetSecs;
pp(uint8(2:9),resettrig,false,port,base_addr); %close
pp(uint8(2:9),Pin4_4,false,port,base_addr) %Write trigger 8
WaitSecs (0.001);
pp(uint8(2:9),resettrig,false,port,base_addr);

WaitSecs(1);
 
%% Rest Period
DrawFormattedText (window, 'Before we get to the fun part we would first like to take a baseline measure. \n\n Please remain in a still but relaxed position and focus on the cross', 'center', 'center', font.color, '','','', 2);
Screen('Flip', window);

Screen('DrawTexture', window,imageTexture{1}, [],StimuliRect);
ButtonPress(Key{1,5})

RestPeriod(Pin4_5,RestDur, Key{1,5},savepath.restname)
%% 

for block = startBlock:blocknum

    
    trials = alltrials(alltrials(:,1) == block,:);
    savepath.filename = [savepath.path, 'TR_', 'p', num2str(SubInfo.subjID), '_SRTT_blocks',num2str(block) '.txt'];
    TrialNum = size(trials,1);
    
    DrawFormattedText (window, ['start of Block: ', num2str(block) ], 'center', 'center', font.color, '','','', 2);
    Screen('Flip', window);

    ButtonPress(Key{1,5})
    
    Results = zeros(TrialNum,10); %Matrix for storing the results
    pp(uint8(2:9),resettrig,false,port,base_addr); %close
    pp(uint8(2:9),Pin4_3,false,port,base_addr) %Write trigger 8
    WaitSecs (0.001);
    pp(uint8(2:9),resettrig,false,port,base_addr);
    
    for Trial = 1:TrialNum %%%%%Main trial loop%%%%%
        Position = trials(Trial,2); %position of stimulus
        Type = trials(Trial,3); %random(82), sequence(84)
        
        target_button = Key{1,Position};
        
        Screen('DrawTexture', window,imageTexture{Position+1}, [],StimuliRect);
        Screen('Flip', window);
        TimeS = GetSecs; %the onset time of the stimulus (image) presentation
        
        pp(uint8(2:9),resettrig,false,port,base_addr); %close
        pp(uint8(2:9),Pin4_1,false,port,base_addr) %Write trigger 8
        WaitSecs (0.001);
        pp(uint8(2:9),resettrig,false,port,base_addr);
        
        KbQueueCreate(deviceIndex);
        KbQueueStart(deviceIndex);
        
        responseTimes = zeros(1,4);
        
        while 1
            [ pressed, firstPress]  = KbQueueCheck(deviceIndex);
            timeSecs                = firstPress(find(firstPress)); %#ok<FNDSB>
            
            
            if pressed
                for kpIdx = 1:length(responseTimes)
                    if firstPress(Key{1,kpIdx}) && ~ responseTimes(kpIdx)
                        responseTimes(kpIdx) = timeSecs - TimeS;
                    end
                end
                
                if firstPress(target_button)
                    pp(uint8(2:9),resettrig,false,port,base_addr); %close
                    pp(uint8(2:9),Pin4_2,false,port,base_addr) %Write trigger 8
                    WaitSecs (0.001);
                    pp(uint8(2:9),resettrig,false,port,base_addr);
                    
                    break;
                end
                
                if firstPress(KbName('ESCAPE')) %Escape the program
                    ListenChar(0);
                    sca
                    error('The program was shut down')
                end
            end
            %  WaitSecs(0.001);
            
        end
        
        KbQueueRelease(deviceIndex);
        
        
        %Master image after the stimulus
        Screen('DrawTexture', window, imageTexture{1}, [],StimuliRect);
        Screen('Flip', window);
        WaitSecs(rsi);
        
        %These data are needed as results: Output
        Results(Trial, 1)   = SubInfo.ID;
        Results(Trial, 2)   = trials(Trial,1);
        Results(Trial, 3)   = Trial;
        Results(Trial, 4)   = Position;
        Results(Trial, 5)   = Type; %random(82) or sequence(84)
        Results(Trial, 6:9) = responseTimes;
        Results(Trial, 10)  = taskcode;
        
        %Write the recent trial of raw data
        dlmwrite(savepath.filename, Results(Trial,:), '-append','delimiter', '\t');
        
        if Trial < TrialNum
            if (trials(Trial + 1,1) ~= trials(Trial,1))
                DrawFormattedText (window, ['Block ',num2str(trials(Trial -1,1)),' finished'], 'center', 'center', font.color, '','','', 2);
                Screen('Flip', window);
                WaitSecs(4);
                Screen('DrawTexture', window, imageTexture{1}, [],StimuliRect);
                Screen('Flip', window);
                WaitSecs(1);
                                
            end
        end
        
    end %end of Trial loop!
end
EndTime=GetSecs;
pp(uint8(2:9),resettrig,false,port,base_addr); %close
pp(uint8(2:9),Pin4_4,false,port,base_addr) %Write trigger 8
WaitSecs (0.001);
pp(uint8(2:9),resettrig,false,port,base_addr);

DrawFormattedText (window, 'End of block 3. Thank you!', 'center', 'center', font.color, '','','', 2);
Screen('Flip', window);

ButtonPress(Key{1,5})
disp('End of experiment')
sca

listenchar(0);
ShowCursor(); %shows the cursor
