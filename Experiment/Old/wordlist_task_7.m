%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       WordList version for Double_Coil study       
%       version modified by Alberto Failla    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Presentation for wordlist learning 
%with TMS stimuli interleaved by Martina Bracco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Set to full screen mode before starting the real experiments!!!!
%esc is active.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






































%%
%addpath ('/Volumes/Maxtor/TMS-EEG experiment material/Run wordlist task/Wordlists');
addpath('/data/Experiments/Robert(sons)/DoubleCoil/Wordlist_Task/Wordlists')
clear all;clc;

HideCursor(); %hides the cursor

% Perform basic initialization of the sound driver:
InitializePsychSound;
%%
%Setting up the screen
screenNumber=max(Screen('Screens'));
Screen('Preference', 'SkipSyncTests', 0);
[keyIsDown,timeSecs,keyCode] = KbCheck;  %This is just to load the KbCheck function

windowcolor=[255,255,255];
textcolor=0;
textcolor=0;
fontsize_in=28; 
fontsize_w=35; 
fontsize_fc=70; 
fontname='Arial';

%%
%Setting variables; requiring user input
button0 = KbName('space');
button1 = KbName('ESCAPE'); %mac version

%%
% Parallel port initialization
base_addr = uint64 (49168) ;
resettrig = fliplr(logical(dec2bin(0,8)-48)) ; % send 0 to reset pp
port=uint8(0) ;

%Use function bin2dec ('10000000') to know the binary code for each pin
Pin37_WL_MEPs_7 = fliplr(logical(dec2bin  (37,8)-48)); % Use pin #37 to initialise the collection of MEPs at recall_recall

Pin36_WL_TS_7 = fliplr(logical(dec2bin  (36,8)-48)); % % % Pin number #36 >> Main Coil at recall_recall

Pin33_WL_CS_7 = fliplr(logical(dec2bin  (33,8)-48)); % % % Pin number #33 >> Conditioning Coil at recall_recall


%%
%define TMS ISI 
         a= 5.0; %minimum interval
         b= 7.0; %maximum interval
         n= 1; %number of trials

%%
% Read a text file into a matrix with one row per input line
% and with a fixed number of columns, set by the longest line.
% Each string is padded with NUL (ASCII 0) characters
%
% open the file for reading
ID = input('Number of participant: ');
code = input('Wordlist: ');
seq = ['wordlist', '-', num2str(code), '.txt'];
ip = fopen(seq,'rt');          % 'rt' means read text
if (ip < 0)
    error('could not open file');   % just abort if error
end
% find length of longest line
max=0;                              % record length of longest string
cnt=0;                              % record number of strings
s = fgetl(ip);                      % get a line
while (ischar(s))                   % while not end of file
   cnt = cnt+1;
   if (length(s) > max)           % keep record of longest
        max = length(s);
   end;
    s = fgetl(ip);                  % get next line
end
% rewind the file to the beginning
frewind(ip);
% create an empty matrix of appropriate size
tab=char(zeros(cnt,max));           % fill with ASCII zeros
% load the strings for real
cnt=0;
s = fgetl(ip);
while (ischar(s))
   cnt = cnt+1;
   tab(cnt,1:length(s)) = s;      % slot into table
    s = fgetl(ip);
end
%close the file and return
fclose(ip);
%tab will contain the wordlist

%%
%%Monitor parameters, picture size setting
%pilot pc: 1366*768; 34.5*19. cm, 1 meter
p.screenWidthCM = 52; %34.5
p.vDistCM = 100;
p.screenwidth = 1400; %1366 % width of the screen in pixels
p.screenheight = 900; %768 % height of the screen in pixels
p.refreshrate = 60;
% Stimulus size
p.imgwidthDeg = 15; % width of the images - 640/427 ratio
p.imgheightDeg = 10; 

%%
%Draw welcome screen with Instructions
[mainwindow, p.sRect]=Screen(screenNumber,'OpenWindow',windowcolor)%,[10 10 1024 768]); %screen size setting for debugging , 
Screen('TextSize', mainwindow,[fontsize_in]);
Screen('TextFont', mainwindow, [fontname]);

%compute and store the center of the screen: p.sRect contains the upper left coordinates (x,y) and the lower right coordinates (x,y)
p = deg2pix(p);
p.xCenter = (p.sRect(3) - p.sRect(1))/2;
p.yCenter = (p.sRect(4) - p.sRect(2))/2;
p.imgRect = [(p.xCenter - p.imgwidthPix/2),(p.yCenter - p.imgheightPix/2),(p.xCenter + p.imgwidthPix/2), (p.yCenter + p.imgheightPix/2)];

DrawFormattedText (mainwindow, 'Your task is now to recall the word list that you practiced earlier. \n When you are ready, press the space key and say the words in the correct order to the mic!', 'center', 'center', textcolor, '','','', 2);
Screen('Flip', mainwindow);
waitForSpace = 0;

while ~waitForSpace
[secs, keyCode, deltaSecs] = KbWait(-1, 0, inf);    
        if ( keyCode( KbName('ESCAPE'))==1 )
            sca
            error('man quit')
        end
        if ( keyCode( KbName('space'))==1 )
            waitForSpace = 1;
        end
end

WaitSecs(0.5); %to avoid too short starting time and unwanted button press
StartTime = GetSecs;
Block = 1;
%%
%Stimulus presentation

cd('/data/Experiments/Robert(sons)/DoubleCoil/Wordlist_Task/RecordedAudio');

    freq = 44100;
    pahandle = PsychPortAudio('Open', [], 2, 0, freq, 2);

    % Preallocate an internal audio recording  buffer with a capacity of 10 seconds:
    PsychPortAudio('GetAudioData', pahandle,10);

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       
        for Block = 7 %%%%%Main block loop%%%%% %
   
    recordedaudio = [];
       
    DrawFormattedText (mainwindow, 'Please, recall the words in the correct order!', 'center', 'center', textcolor, '','','', 2);
    Screen('Flip', mainwindow);
    
    % Open the default audio device [], with mode 2 (== Only audio capture),
    % and a required latencyclass of zero 0 == no low-latency mode, as well as
    % a frequency of 44100 Hz and 2 sound channels for stereo capture.
    % This returns a handle to the audio device:


    % Start audio capture immediately and wait for the capture to start.
    % We set the number of 'repetitions' to zero,
    % i.e. record until recording is manually stopped.
    PsychPortAudio('Start', pahandle, 0, 0, 1);


    waitForSpace = 0;

    while ~waitForSpace
    [secs, keyCode, deltaSecs] = KbWait(-1, 0, 2);    
            if ( keyCode( KbName('ESCAPE'))==1 )
                sca
                error('man quit')
            end

            if ( keyCode( KbName('space'))==1 )
                waitForSpace = 1;
                
            end
       

        % Query current capture status and print it to the Matlab window:
        s = PsychPortAudio('GetStatus', pahandle);

        % Retrieve pending audio data from the drivers internal ringbuffer:
        audiodata = PsychPortAudio('GetAudioData', pahandle);

        % And attach it to our full sound vector:
        recordedaudio = [recordedaudio audiodata]; %#ok<AGROW>

    end
        
    % Stop capture:
    PsychPortAudio('Stop', pahandle);

    % Perform a last fetch operation to get all remaining data from the capture engine:
    audiodata = PsychPortAudio('GetAudioData', pahandle);
    recordedaudio = [recordedaudio audiodata];

    psychwavwrite(transpose(recordedaudio), 44100, 16, ['P',num2str(ID),'_WL',num2str(code),'_FR',num2str(Block),'.wav'])
    

  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  
   %                                 PREPARE SCREEN BEFORE MEPs 
   
%     Screen('TextSize',mainwindow,17);
    DrawFormattedText (mainwindow, 'Stimulation is about to start! \n Stay still and fixate the fixation cross', 'center', 'center', textcolor, '','','', 2);
Screen('Flip', mainwindow);

WaitSecs (2)

waitForSpace = 0;
while ~waitForSpace
[secs, keyCode, deltaSecs] = KbWait(-1, 0, inf);    
        if ( keyCode( KbName('ESCAPE'))==1 )
% if ( keyCode( KbName('esc'))==1 )
            sca
            error('man quit')
        end
        if ( keyCode( KbName('space'))==1 )
            waitForSpace = 1;
        end
        
end

 % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
   
                                        % MEPs STIMULATION
                                        
 Screen('TextSize',mainwindow,[fontsize_fc]);
DrawFormattedText (mainwindow, '+', 'center', 'center', textcolor, '','','', 2);
Screen('Flip', mainwindow);
    
     WaitSecs(2);
    
     for Trial = 1:20 %%%%%TMS stimuli%%%%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin37_WL_MEPs_7,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
           x1 = a + (b-a).*rand(n,1) 
        
         WaitSecs(x1);

    Block = Block+1;
    end %end of block loop!  
    
    
    
    
   % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
    
    %                                                    PREPARE SCREEN BEFORE DOUBLE COIL STIMULATION
    
    
      DrawFormattedText (mainwindow, 'Now we are going to place the other coil on your head. \n The block is about to start! \n Stay STILL and fixate the fixation cross', 'center', 'center', textcolor, '','','', 2);
Screen('Flip', mainwindow);

WaitSecs (2)  
    
 waitForSpace = 0;
while ~waitForSpace
[secs, keyCode, deltaSecs] = KbWait(-1, 0, inf);    
        if ( keyCode( KbName('ESCAPE'))==1 )
% if ( keyCode( KbName('esc'))==1 )
            sca
            error('man quit')
        end
        if ( keyCode( KbName('space'))==1 )
            waitForSpace = 1;
        end
        
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
    
    %                                                    DOUBLE COIL STIMULATION
Screen('TextSize',mainwindow,[fontsize_fc]);
DrawFormattedText (mainwindow, '+', 'center', 'center', textcolor, '','','', 2);
Screen('Flip', mainwindow);
    
     WaitSecs(2);
    
     for Trial = 1:12 %%%  Conditioning Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin33_WL_CS_7,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            WaitSecs(0.009); %%% Inter-Stimulus Interval %%%        10ms = 0.01sec >> However, after we close the first pin, we ask to wait 1ms, so here we just ask to wait (10 - 1ms = 9ms = 0.009s)
            
            
            
            %%%  Test Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin36_WL_TS_7,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            
           x1 = a + (b-a).*rand(n,1) %%% Inter-Trial Interval %%%
        
         WaitSecs(x1);

         
         
    Block = Block+1;
    
    
    
        end %end of block loop!
        
        
        end 
            
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  

EndTime=GetSecs;


%%
 Screen('TextSize',mainwindow,[fontsize_in]);
DrawFormattedText (mainwindow, 'End of the task. Thank you!', 'center', 'center', textcolor, '','','', 2);
Screen('Flip', mainwindow);
WaitSecs(3);


ShowCursor(); %shows the cursor
Screen('CloseAll'); %Closes Screen


%%

