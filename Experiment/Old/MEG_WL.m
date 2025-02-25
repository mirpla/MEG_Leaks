function MEG_WL(SubInfo, path, debug, startBlock, pahandle, audio) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       WordList %       version modified by Alberto Failla    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Presentation for wordlist learning                               %
%       with TMS stimuli interleaved modified by Mircea van der Plas     %
%               based on code by Martina Bracco                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

selfpaced = 0;
MEG = 0; % Set to 0 to debug outside MEG/ no triggers/no Buttonbox
% Prevent spilling of keystrokes into console:
if debug == 0
    ListenChar(-1);
    HideCursor();
end


%% Set up paths and stimuli

% Determine paths for Images and Output
SeqPath                 = [path, '/Sequence_files/'];
savepath.path           = [path, '/Log/WL/' ];
savepath.restname       = [savepath.path, 'MEG_WL_', 'p', num2str(SubInfo.subjID), '_WL_Log.txt'];

%Reading sequence file
seq         = [SeqPath,'wordlist', '-', num2str(SubInfo.Cond), '.txt']; %sequence file to be loaded

fid = fopen(seq);           % open file
WL  = textscan(fid,'%s');   % scan for strings
fclose(fid);                % close file


%% Setup screen 

PsychDefaultSetup(2);

Screen('Preference', 'SkipSyncTests', debug ); % Disable calibrations for
% testing purposes, make sure to keep disabled during actual experiment
% (set to 0 in actual experiment)
screens = Screen('Screens');
screenNumber = max(screens); % Select correct Screen here

black = BlackIndex(screenNumber);

windowcolor = [128,128,128]/256;

if debug == 1
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, windowcolor, [0 0 1080 720]);
else
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, windowcolor);
end

% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Retrieve Maximum possible Priority number and give it to othe window
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

RestDur     = 180; % Duraction of Rest Period in s 
blocknum    = 10;

% Determine all the font parameters
font.size_in             = 28; % fontsize for the instructions
font.size_wl             = 35; % fontsize for the wordlist
font.name                = 'Arial';
font.color               = black;

Screen('TextSize', window, font.size_in);
Screen('TextFont', window, font.name);

% Get timing information 
ifi = Screen('GetFlipInterval', window);
isiTimeSecs = 1;
isiTimeFrames = round(isiTimeSecs / ifi);

% Numer of frames to wait before re-drawing
waitframes = 1;

% Assemble the fixation cross
fixCr.CrossDimPix     = 15;
fixCr.lineWidthPix    = 6;
fixCr.Color           = black;
fixCr.GoCol           = [15, 255, 80]/256; % Neon green color for fixation cross during triggered recall

fixCr.xCoords = [-fixCr.CrossDimPix fixCr.CrossDimPix 0 0];
fixCr.yCoords = [0 0 -fixCr.CrossDimPix fixCr.CrossDimPix];
fixCr.allCoords = [fixCr.xCoords; fixCr.yCoords];

[fixCr.xCent, fixCr.yCent] = RectCenter(windowRect);

%% Buttons 


deviceIndex = []; % uses defauly Keyboard if Bottonbox flag is not specified

if MEG == 1
    Key{1,1} = KbName('5%'); % Cancel button
    Key{1,2} = KbName('7&'); % Triggered response section
else
    Key{1,1} = KbName('ESCAPE'); % Cancel button
    Key{1,2} = KbName('space'); % Triggered response section
end
[~,~,~] = KbCheck;

%% Prepare triggers
if MEG == 1
    % Parallel port triggers
    enctrig = 1; % Indicating Encoding task is being performed coded as multipes of 256 STI101
    frectrig = 2; % Indicating free recall section
    trectrig = 3; % Indicating triggered recall section
    resttrig  = 4; % indicates official rest period coded as as multipes of 512 + x*512on STI101
end
% Pixel mode triggers (more accurate than parallel port but with consistent 8.33ms delay)
trigPos = [0 0 1 1]; % position of the pixel used for trigger

% Define trigger values
tr_Col{1,1} = [128,128,128]/256; % 0 Default/ no trigger
tr_Col{1,2} = [128,139,160]/256; % 11 Start of a Trial/Visual onset in encoding
tr_Col{1,3} = [128,151, 65]/256; % 23 onset and Offset of free Recall
tr_Col{1,4} = [128,101, 66]/256; % 37 onset of triggered Recall button press
tr_Col{1,5} = [128,123,146]/256; % 43 onset of triggered Recall8 color change
tr_Col{1,6} = [128,248,252]/256; % 200 Start & end of Experiment

% make the corresponding textures
for tr_cnt = 1:size(tr_Col,2)
    % for some reason Chris' code does this so I keep it for now
    trigger_perm = uint8(permute(repmat(tr_Col{1,tr_cnt}',1,1,1),[2 3 1]));
    tr_tx{1,tr_cnt} = Screen('MakeTexture', window, trigger_perm);
end
%% %%%%%%%%%%% %%%%% %%%%% %%%%%%%%%%% %%
%%%%%%%%%%%%% Experiment %%%%%%%%%%%%%%%%
%%%%%%%%%%%%% %%%%%% %%%%%% %%%%%%%%%%%%%
if MEG == 1
    ppdev_mex('Open', 1) % open parallel port
end
Screen('DrawTexture',window, tr_tx{1,6}, [], trigPos); % signal start of WL task in data
Screen('Flip', window);
WaitSecs(0.05);

% PREPARE SCREEN BEFORE TASK
DrawFormattedText (window, 'Welcome! In this task you will see a list of words. \n Please, read them aloud in the presented order!', 'center', 'center', font.color, '','','', 2);
Screen('Flip', window);

WaitSecs(0.5); %to avoid too short starting time and unwanted button press

ButtonPress(Key{1,1})

WaitSecs(0.5); %to avoid too short starting time and unwanted button press

%StartTime = GetSecs;
%Stimulus presentation

% Preallocate an internal audio recording  buffer with a capacity of 10 seconds:
if audio == 1
    PsychPortAudio('GetAudioData', pahandle,10);
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Main block loop%%%%%
for Block = startBlock:blocknum
    
    DrawFormattedText (window, ['start of Block: ', num2str(Block) ], 'center', 'center', font.color, '','','', 2);
    
    if MEG == 1
        ppdev_mex('Write', 1, enctrig)
    end
    
    Screen('Flip', window);
    
    ButtonPress(Key{1,1})
    WaitSecs(0.5); %
    
    recordedaudiointro = [];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if audio == 1
        PsychPortAudio('Start', pahandle, 0, 0, 1);
    end 
    
    WaitSecs(2);
    
    for Trial = 1:12 %%%%%Main trial loop%%%%%
        pres = WL{1,1}{Trial,1};
        wordpres = sprintf(pres);
        Screen('TextSize', window, font.size_wl);
        DrawFormattedText(window, wordpres, 'center', 'center', font.color);
        Screen('DrawTexture',window, tr_tx{1,2}, [], trigPos); % signal start of trial       
        Screen('Flip', window);
        
        WaitStart = GetSecs;
        WaitTimer = 0;
        while ((WaitTimer >= WaitStart+2)  ~=1)   % Checks continuously whether time has passed or ESCAPE has been pressed to skip/cancel it
            if audio == 1
                s = PsychPortAudio('GetStatus', pahandle);
                
                % Retrieve pending audio data from the drivers internal ringbuffer:
                audiodataintro = PsychPortAudio('GetAudioData', pahandle);
                
                % And attach it to our full sound vector:
                recordedaudiointro = [recordedaudiointro audiodataintro]; %#ok<AGROW>
            end
            WaitTimer = GetSecs;
        end
    end
    
    if MEG == 1
        ppdev_mex('Write',1, 0)
    end
    
    if audio == 1
        PsychPortAudio('Stop', pahandle);
        % Perform a last fetch operation to get all remaining data from the capture engine:
        audiodataintro = PsychPortAudio('GetAudioData', pahandle);
        recordedaudiointro = [recordedaudiointro audiodataintro];
        
        psychwavwrite(transpose(recordedaudiointro), 44100, 16, [path,'/Log/WL/P',num2str(SubInfo.subjID),'_WLintro',num2str(SubInfo.Cond),'_FR',num2str(Block),'.wav'])
    end
    %%  Recall
    Screen('TextSize', window, [font.size_in]);
    DrawFormattedText (window, 'Please, recall the words freely in the correct order!', 'center', 'center', font.color, '','','', 2);
    
    Screen('DrawTexture',window, tr_tx{1,3}, [], trigPos); % signal start of trial
    
    if MEG == 1
        ppdev_mex('Write',1, frectrig)
    end
    Screen('Flip', window);
    
    % Open the default audio device [], with mode 2 (== Only audio capture),
    % and a required latencyclass of zero 0 == no low-latency mode, as well as
    % a frequency of 44100 Hz and 2 sound channels for stereo capture.
    % This returns a handle to the audio device:
    
    
    % Start audio capture immediately and wait for the capture to start.
    % We set the number of 'repetitions' to zero,
    % i.e. record until recording is manually stopped.
    if audio == 1
        PsychPortAudio('Start', pahandle, 0, 0, 1);
    end
    recordedaudio = [];
    waitForKey = 0;
    
    while ~waitForKey
        [secs, keyCode, deltaSecs] = KbWait(-1, 0, 2);
        if ( keyCode(Key{1,1})==1 )
            % if ( keyCode( KbName('esc'))==1 )
            sca
            error('man quit')
        end
        
        if ( keyCode( Key{1,2})==1 )
            waitForKey = 1;
        end
        if audio == 1
            % Query current capture status and print it to the Matlab window:
            s = PsychPortAudio('GetStatus', pahandle);
            
            % Retrieve pending audio data from the drivers internal ringbuffer:
            audiodata = PsychPortAudio('GetAudioData', pahandle);
            
            % And attach it to our full sound vector:
            recordedaudio = [recordedaudio audiodata]; %#ok<AGROW>
        end
        
    end
    
    if MEG == 1
        ppdev_mex('Write',1, 0)
    end
    
    if audio == 1
        % Stop capture:
        PsychPortAudio('Stop', pahandle);
        
        % Perform a last fetch operation to get all remaining data from the capture engine:
        audiodata = PsychPortAudio('GetAudioData', pahandle);
        recordedaudio = [recordedaudio audiodata];
        
        psychwavwrite(transpose(recordedaudio), 44100, 16, [path,'/Log/WL/P',num2str(SubInfo.subjID),'_WL',num2str(SubInfo.Cond),'_FR',num2str(Block),'.wav'])
    end
   
    Screen('DrawTexture',window, tr_tx{1,1}, [], trigPos);
    Screen('Flip', window);
    WaitSecs(0.05); %
    
    %% Triggered Recall
    if selfpaced == 1
        w = 1; %counter for which word we are at during triggered retrieval
        DrawFormattedText (window, 'Well done! Next we have the prompted recall! Please say the next word you remember as soon as the fixation cross colour changes to green', 'center', 'center', font.color, '','','', 2);
        Screen('Flip', window);
        ButtonPress(Key{1,1})
        
        remembered = 1;
        while remembered == 1
            while ~waitForKey
                [~, keyCode, ~] = KbWait(-1, 0, 2);
                Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2)
                Screen('DrawTexture',window, tr_tx{1,4}, [], trigPos);
                Screen('Flip', window);
                
                if ( keyCode(Key{1,1})==1 )
                    remembered = 0;
                    break
                end
                if ( keyCode( Key{1,2})==1 )
                    waitForKey = 1;
                end
            end
            if remembered ==0
                break
            end
            if audio == 1
                % Start audio capture immediately and wait for the capture to start.
                % We set the number of 'repetitions' to zero,
                % i.e. record until recording is manually stopped.
                PsychPortAudio('Start', pahandle, 0, 0, 1);
            end
            recordedaudio = [];
            waitForKey = 0;
            
            % Draw fixation cross
            Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2)
            Screen('Flip', window);
            
            WaitSecs(1)
            
            % change fixation cross color to green
            Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.GoCol,[fixCr.xCent, fixCr.yCent], 2)
            Screen('DrawTexture',window, tr_tx{1,5}, [], trigPos);
            Screen('Flip', window);
            
            if audio == 1
                % Query current capture status and print it to the Matlab window:
                s = PsychPortAudio('GetStatus', pahandle);
                
                % Retrieve pending audio data from the drivers internal ringbuffer:
                audiodata = PsychPortAudio('GetAudioData', pahandle);
                
                % And attach it to our full sound vector:
                recordedaudio = [recordedaudio audiodata]; %#ok<AGROW>
                
                % Stop capture:
                PsychPortAudio('Stop', pahandle);
                
                % Perform a last fetch operation to get all remaining data from the capture engine:
                audiodata = PsychPortAudio('GetAudioData', pahandle);
                recordedaudio = [recordedaudio audiodata];
                
                psychwavwrite(transpose(recordedaudio), 44100, 16, [path,'/Log/WL/P',num2str(SubInfo.subjID),'_WL',num2str(SubInfo.Cond),'_B',num2str(Block),'W',w,'.wav'])
            end
            w = w + 1;
            WaitSecs(1)
            
            Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2)
            Screen('DrawTexture',window, tr_tx{1,1}, [], trigPos);
            Screen('Flip', window);
            WaitSecs(0.1)
        end
    else 
        %% Triggered non-selfpaced
        w = 1; %counter for which word we are at during triggered retrieval
        DrawFormattedText (window, 'Well done! Next we have the prompted recall! Please say the next word you remember as soon as the fixation cross colour changes to green', 'center', 'center', font.color, '','','', 2);
        Screen('Flip', window);
        ButtonPress(Key{1,1})
        
        remembered = 1;
        while remembered == 1
            Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2)
            Screen('DrawTexture',window, tr_tx{1,4}, [], trigPos);
            Screen('Flip', window);
            
            if audio == 1
                % Start audio capture immediately and wait for the capture to start.
                % We set the number of 'repetitions' to zero,
                % i.e. record until recording is manually stopped.
                PsychPortAudio('Start', pahandle, 0, 0, 1);
            end
            recordedaudio = [];

            % Draw fixation cross
            Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2)
            Screen('Flip', window);
            
            WaitSecs(1)
            
            % change fixation cross color to green
            Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.GoCol,[fixCr.xCent, fixCr.yCent], 2)
            Screen('DrawTexture',window, tr_tx{1,5}, [], trigPos);
            Screen('Flip', window);
            
            if audio == 1
                % Query current capture status and print it to the Matlab window:
                s = PsychPortAudio('GetStatus', pahandle);
                
                % Retrieve pending audio data from the drivers internal ringbuffer:
                audiodata = PsychPortAudio('GetAudioData', pahandle);
                
                % And attach it to our full sound vector:
                recordedaudio = [recordedaudio audiodata]; %#ok<AGROW>
                
                % Stop capture:
                PsychPortAudio('Stop', pahandle);
                
                % Perform a last fetch operation to get all remaining data from the capture engine:
                audiodata = PsychPortAudio('GetAudioData', pahandle);
                recordedaudio = [recordedaudio audiodata];
                
                psychwavwrite(transpose(recordedaudio), 44100, 16, [path,'/Log/WL/P',num2str(SubInfo.subjID),'_WL',num2str(SubInfo.Cond),'_B',num2str(Block),'W',w,'.wav'])
            end
            w = w + 1;
            WaitSecs(1)
            
            Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2)
            Screen('DrawTexture',window, tr_tx{1,1}, [], trigPos);
            Screen('Flip', window);
            WaitSecs(1.5)
        end
        
    end
    %% Posttask Rest Period
    
    DrawFormattedText (window, 'Thank you! Once you proceed and press a button, please stay as still as possible and focus on the fixation cross', 'center', 'center', font.color, '','','', 2);
    Screen('Flip', window);
    
    ButtonPress(Key{1,1})
    
    Screen('DrawLines', window, fixCr.allCoords, fixCr.lineWidthPix, fixCr.Color,[fixCr.xCent, fixCr.yCent], 2);
    if MEG == 1       
        ppdev_mex('Write',1,resttrig)
    end
    Screen('Flip', window);
      
    RestPeriod(RestDur, Key{1,1},deviceIndex,savepath.restname)
    if MEG == 1
        ppdev_mex('Write',1,0)
    end 
    
end %end of block loop!


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%EndTime=GetSecs;
    
%%
Screen('TextSize', window, font.size_in);
DrawFormattedText (window, 'Thank you! For your participation!', 'center', 'center', font.color, '','','', 2);

Screen('DrawTexture',window, tr_tx{1,6}, [], trigPos); % signal start of WL task in data
Screen('Flip', window);
ButtonPress(Key{1,1})

%%
ShowCursor(); %shows the cursor
Screen('CloseAll'); %Closes Screen