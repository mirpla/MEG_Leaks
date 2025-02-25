 function MEG_SRT(SubInfo, path, debug, startBlock, MEG) 
try
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SRTT version modified with intermittent Rest %
% periods during which EEG will be recorded by %
%             Mircea van der Plas              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Original version: Andrea Kobor's ASRT script
%contact: kobor.andrea@ttk.mta.hu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AssertOpenGL;
Datapixx('Open');

% Force GetSecs and WaitSecs into memory to avoid latency later on
GetSecs;
WaitSecs(0.1);

% Prevent spilling of keystrokes into console:
if debug == 0
    ListenChar(-1);
    HideCursor();
end 
% Flag determining whether SRT also has resting state periods 
restFlag = 1; % if 1 rest; if 0 no rest  
RestDur = 180;

% Determine paths for Images and Output
imgpath                 = [path, '/Images/'];
SeqPath                 = [path, '/Sequence_files/'];
savepath.path           = [path, '/Log/SRT/' ];
savepath.restname       = [savepath.path, 'MEG_SRT_', 'p', num2str(SubInfo.subjID), '_Rest.csv'];

% Stimulus to screen ratio
StimSize    = 1;

%[kbIdx, devName] = GetKeyboardIndices();
% Make sure that MEG button box is selected
deviceIndex = []; % find appropriate device index fo the MEG box

StimPath{1,1}   = [imgpath,'master-grey.bmp'];
StimPath{1,2}   = [imgpath,'R1-blue-grey.bmp'];
StimPath{1,3}   = [imgpath,'R2-blue-grey.bmp'];
StimPath{1,4}   = [imgpath,'R3-blue-grey.bmp'];
StimPath{1,5}   = [imgpath,'R4-blue-grey.bmp'];

%Reading sequence file
taskcode    = SubInfo.Cond;
seq         = [SeqPath,'seq', '-', num2str(taskcode), '.txt']; %sequence file to be loaded
alltrials   = dlmread(seq); %this contains all info about the trials
blocknum    = max(alltrials(:,1));

%% set up default settings for Window/Screen
PsychDefaultSetup(2);

Screen('Preference', 'SkipSyncTests', debug ); % Disable calibrations for
% testing purposes, make sure to keep disabled during actual experiment
% (set to 0 in actual experiment)
screens = Screen('Screens');
screenNumber = max(screens); % Select correct Screen here

windowcolor = [128,128,128]/256;

if debug == 1
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, windowcolor, [0 0 640 480]);
else
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, windowcolor);
end
% Get screen Dimensions and Centre of Screen
[p.screenwidth, p.screenheight] = Screen('WindowSize', window);

% Get info on frame/refresh rate
%ifi = Screen('GetFlipInterval', window);
%FRscreen = FrameRate(window);
%nominalFR = Screen('NominalFrameRate', window);

% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
Screen('ColorRange',window,255); % Set the color range for pixel mode 

% Retrieve Maximum possible Priority number and give it to othe window
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

% Here we use to a waitframes number greater then 1 to flip at a rate not
% equal to the monitors refreash rate. For this example, once per second,
% to the nearest frame
%flipSecs = 1;
%waitframes = round(flipSecs / ifi);
black = BlackIndex(screenNumber);

rsi         = 0.5; % Interstimulus interval in s
%% Prepare Stimuli
font.size                = 28;
font.name                = 'Arial';
font.color               = black;

Screen('TextSize', window, font.size);
Screen('TextFont', window, font.name);

% Keys also represented in STI102
Key{1,1} = KbName('7&'); % Right Index Finger 2^9
Key{1,2} = KbName('8*'); % Right Middle Finger 2^8
Key{1,3} = KbName('9('); % Right Ring Finger 2^7
Key{1,4} = KbName('0)'); % Right Pinky 2^6
Key{1,5} = KbName('5%'); % Left Thumb 2^11

% Prepare all the images used in the Experiment for fastest possible
% presentation
Stimuli         = cellmat(size(StimPath,1)*size(StimPath,2),1);
imageTexture    = Stimuli;
% load in main figures
for totstim = 1:(size(StimPath,1)*size(StimPath,2))
    [Stimuli{totstim}] = imread(StimPath{1,totstim});
    
    % Resize Images
    [s1, s2, ~] = size(Stimuli{totstim});
    aspectRatio = s2 / s1;
    heightScalers = StimSize;
    
    % Make the image into a texturepOD_SRT
    imageTexture{totstim} = Screen('MakeTexture', window, Stimuli{totstim});
end
imageHeights = p.screenheight .* heightScalers;
imageWidths = imageHeights .* aspectRatio;
theRect     = [0 0 imageWidths imageHeights];
StimuliRect = CenterRectOnPointd(theRect, p.screenwidth * 0.5, p.screenheight * 0.5);

%% Prepare triggers

% Enabling Pixel Mode for precise triggering, see: https://vpixx.com/vocal/pixelmode/
Datapixx('EnablePixelMode');
Datapixx('SetPropixxDlpSequenceProgram', 0); 
Datapixx('RegWr');

% Parallel port triggers 
blocktrig = 3; % Indicating task is being performed coded as multipes of (3+blocknum)*256 on STI101

resttrig  = 1; % indicates official rest period coded as as multipes of 256 on STI101

% Pixel mode triggers (more accurate than parallel port but with consistent 8.33ms delay) 
trigPos     = [0 0 1 1]; % position of the pixel used for trigger
trigRect    = CenterRectOnPointd(trigPos, 1, 1);

% Define trigger values
tr_Col{1,1} = [128,128,128]/256; % 0 Default/ no trigger
tr_Col{1,2} = [128,139,160]/256; % 11 Start of a Trial/Visual stimulus
tr_Col{1,3} = [128,151, 65]/256; % 23 Timestamp of correct response
tr_Col{1,4} = [128,101, 66]/256; % 37 Timestamp of response
tr_Col{1,5} = [128,248,252]/256; % 200 Start & end of Experiment
tr_Col{1,6} = [128, 15,160]/256; % 5 Other button presses

% % make the corresponding textures
for tr_cnt = 1:size(tr_Col,2)
    % for some reason Chris' code does this so I keep it for now
    trigger_perm = uint8(permute(repmat(tr_Col{1,tr_cnt}',1,1,1),[2 3 1]));
    tr_tx{1,tr_cnt} = Screen('MakeTexture', window, trigger_perm);  
end 

%% Assemble the fixation cross
fixCr.CrossDimPix     = 15;
fixCr.lineWidthPix    = 6;
fixCr.Color           = black;
fixCr.GoCol           = [15, 255, 80]/256; % Neon green color for fixation cross during triggered recall

fixCr.xCoords = [-fixCr.CrossDimPix fixCr.CrossDimPix 0 0];
fixCr.yCoords = [0 0 -fixCr.CrossDimPix fixCr.CrossDimPix];
fixCr.allCoords = [fixCr.xCoords; fixCr.yCoords];

[fixCr.xCent, fixCr.yCent] = RectCenter(windowRect);


%% %%%%%%%%%%% %%%%% %%%%% %%%%%%%%%%% %%
%%%%%%%%%%%%% Experiment %%%%%%%%%%%%%%%%
%%%%%%%%%%%%% %%%%%% %%%%%% %%%%%%%%%%%%%

% Screen('DrawTexture',window, tr_tx{1,5}, [], trigPos); % signal start of SRT in data
% Screen('Flip', window);
WaitSecs(0.05);

DrawFormattedText(window, 'Welcome! This experiment is design to test how quickly you can respond to visual cues. \n\n You will be shown 4 circles on a grey screen.\n\n During each trial a dot will appear in one of the circles. \n\n You will need to respond by pressing the corresponding keys:, "m"', 'center', 'center', font.color, '','','', 2);
Screen('Flip', window);

WaitSecs(0.5); %to avoid too short starting time and unwanted button press

ButtonPress(Key{1,5})
 
%% Rest Period
DrawFormattedText (window, 'Before we get to the fun part we would first like to take a baseline measure. \n\n Please remain in a still but relaxed position and focus on the cross', 'center', 'center', font.color, '','','', 2);
Screen('DrawTexture',window,  tr_tx{1,5}, [], trigPos);
Screen('Flip', window);

ButtonPress(Key{1,5})

% first resting period
if MEG == 1
    ppdev_mex('Open', 1);
    ppdev_mex('Write', 1, restrig); % should give value 512
end

Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2)
Screen('Flip', window);
RestPeriod(RestDur, Key{1,5},deviceIndex,savepath.restname)       

if MEG == 1
    ppdev_mex('Close', 1);
end 

WaitSecs(1)
%%

for block = startBlock:blocknum
    if MEG == 1
        ppdev_mex('Open', 1);
        ppdev_mex('Write',1,blocktrig+block)
        ppdev_mex('Close', 1);
    end 
    
    trials = alltrials(alltrials(:,1) == block,:);
    savepath.filename = [savepath.path, 'MEG_', 'p', num2str(SubInfo.subjID), '_SRTT_blocks',num2str(block) '.txt'];
    TrialNum = size(trials,1);
    
    DrawFormattedText (window, ['start of Block: ', num2str(block) ], 'center', 'center', font.color, '','','', 2);
    Screen('Flip', window);
   
    ButtonPress(Key{1,5})
    
    Screen('DrawTexture', window,imageTexture{1}, [],StimuliRect);
    Screen('Flip', window);
    WaitSecs(0.5)
    
    for Trial = 1:TrialNum %%%%%Main trial loop%%%%%
        Position = trials(Trial,2); %position of stimulus
        Type = trials(Trial,3); %random(82), sequence(84)
        
        target_button = Key{1,Position};
        
        Screen('DrawTexture', window,imageTexture{Position+1}, [],StimuliRect);
        Screen('DrawTexture',window,  tr_tx{1,2}, [], trigPos); % signal start of trial
        vb1 = Screen('Flip', window);
        
        TimeS = GetSecs; %the onset time of the stimulus (image) presentation
        
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
                        
                        Screen('DrawTexture', window,imageTexture{Position+1}, [],StimuliRect);
                        Screen('DrawTexture',window,  tr_tx{1,4}, [], trigPos); % signal general response
                        Screen('Flip', window);                        
                    end
                end
                
                if firstPress(target_button)                
                    Screen('DrawTexture', window, imageTexture{1}, [],StimuliRect) % back to master image
                   Screen('DrawTexture',window,  tr_tx{1,3}, [], trigPos); % signal correct response
                    Screen('Flip', window);
                   
                    break;
                end
                
                if firstPress(KbName(Key{1,5})) %Escape the program
                    ListenChar(0);
                    sca
                    if MEG == 1
                        ppdev_mex('CloseAll', 1) 
                    end 
                    error('The program was shut down manually')                   
                end
            end
            %  WaitSecs(0.001);
            
        end
        
        KbQueueRelease(deviceIndex);
        
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
                if MEG == 1
                    ppdev_mex('Open', 1);
                    ppdev_mex('Write',1,blocktrig+block)
                    ppdev_mex('Close', 1);
                end 
                bidx = bidx +2;
                
                WaitSecs(4);            
            end
        end
        
    end %end of Trial loop!
   
    % rest period at the end
    if restFlag == 1  && block == blocknum
        
        DrawFormattedText (window, 'Once you press any button, please keep still and focus on the centre of the screen for the next 3 minutes', 'center', 'center', font.color, '','','', 2);
        Screen('Flip', window);
        ButtonPress(Key{1,5})
        
        Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2)
        Screen('Flip', window);
        if MEG == 1
            ppdev_mex('Open', 1);
            ppdev_mex('Write', 1, restrig); % should give value 512            
        end
        
        RestPeriod(RestDur, Key{1,5},deviceIndex,savepath.restname)
        
        if MEG == 1
            ppdev_mex('Write',1,0)
            ppdev_mex('Close', 1);
        end
        
    end
    Screen('DrawTexture', window, imageTexture{1}, [],StimuliRect);
    Screen('Flip', window);
    WaitSecs(1);
end


DrawFormattedText (window, 'End of block 3. Thank you!', 'center', 'center', font.color, '','','', 2);
Screen('Flip', window);

ButtonPress(Key{1,5})

Screen('DrawTexture',window,  tr_tx{1,5}, [], trigPos); % signal successful completion of SRT in data
Screen('Flip', window);
WaitSecs(0.05);

disp('End of experiment')
catch
    Datapixx('Close');
    sca
    Priority(0);
    
    ShowCursor(); %shows the cursor
    if MEG == 1
        ppdev_mex('CloseAll', 1)
end 
    psychrethrow(psychlasterror)
end 
Datapixx('Close'); 
sca
Priority(0);

ShowCursor(); %shows the cursor
if MEG == 1
    ppdev_mex('CloseAll', 1) 
end 