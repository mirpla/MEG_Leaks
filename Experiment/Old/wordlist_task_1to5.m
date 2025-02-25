%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       WordList version for Double_Coil study       
%       version modified by Alberto Failla    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Presentation for wordlist learning 
%       with TMS stimuli interleaved by Martina Bracco
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
fontsize_in=28; 
fontsize_w=35; 
fontsize_fc=70; 
fontname='Arial';

%Setting variables; requiring user input
button0 = KbName('space');
button1 = KbName('ESCAPE'); %mac version
% button1 = KbName('esc'); %windows version
%%
% Parallel port initialization
base_addr = uint64 (49168);
resettrig = fliplr(logical(dec2bin(0,8)-48)); % send 0 to reset pp
port=uint8(0);

%Use function bin2dec ('10000000') to know the binary code for each pin
% THIS IS THE OLD VERSION THAT MARTINA USED. EACH ROUND OF THE WORD_LIST IS
% ASSIGNED TO A DIFFERENT PIN NUMBER. I THINK THIS IS TO HAVE DIFFERENT TRIGGERS ON THE EEG. EACH BLOCK OF THE WORD_LIST IS BEING TAGGED DIFFERENTLY ON TH EEEG.
% AS WE ARE USING TWO COILS WE WANT TO KEEP THINGS CLEAN AND CLEAR.

Pin13_WL_MEPs_1 = fliplr(logical(dec2bin  (13,8)-48)); % Use pin #13 to initialise the collection of MEPs at round 1
Pin15_WL_MEPs_2 = fliplr(logical(dec2bin  (15,8)-48)); % Use pin #15 to initialise the collection of MEPs at round 2
Pin21_WL_MEPs_3 = fliplr(logical(dec2bin  (21,8)-48)); % Use pin #21 to initialise the collection of MEPs at round 3
Pin23_WL_MEPs_4 = fliplr(logical(dec2bin  (23,8)-48)); % Use pin #23 to initialise the collection of MEPs at round 4
Pin29_WL_MEPs_5 = fliplr(logical(dec2bin  (29,8)-48)); % Use pin #29 to initialise the collection of MEPs at round 5

Pin12_WL_TS_1 = fliplr(logical(dec2bin  (12,8)-48)); % % % Pin number #12 >> Main Coil at round 1
Pin14_WL_TS_2 = fliplr(logical(dec2bin  (14,8)-48)); % % % Pin number #14 >> Main Coil at round 2
Pin20_WL_TS_3 = fliplr(logical(dec2bin  (20,8)-48)); % % % Pin number #20 >> Main Coil at round 3
Pin22_WL_TS_4 = fliplr(logical(dec2bin  (22,8)-48)); % % % Pin number #22 >> Main Coil at round 4
Pin28_WL_TS_5 = fliplr(logical(dec2bin  (28,8)-48)); % % % Pin number #28 >> Main Coil at round 5

Pin9_WL_CS_1 = fliplr(logical(dec2bin  (9,8)-48)); % % % Pin number #9 >> Conditioning Coil at round 1
Pin11_WL_CS_2 = fliplr(logical(dec2bin  (11,8)-48)); % % % Pin number #11 >> Conditioning Coil at round 2
Pin17_WL_CS_3 = fliplr(logical(dec2bin  (17,8)-48)); % % % Pin number #17 >> Conditioning Coil at round 3
Pin19_WL_CS_4 = fliplr(logical(dec2bin  (19,8)-48)); % % % Pin number #19 >> Conditioning Coil at round 4
Pin25_WL_CS_5 = fliplr(logical(dec2bin  (25,8)-48)); % % % Pin number #25 >> Conditioning Coil at round 5

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
p.imgwidthDeg = 15; % width of the images - 640/427 ratio3
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

%%

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  
   %                                 PREPARE SCREEN BEFORE TASK 
DrawFormattedText (mainwindow, 'Welcome! In this task you will see a list of words. \n Please, read them aloud in the presented order!', 'center', 'center', textcolor, '','','', 2);
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
%Stimulus presentation

cd('/data/Experiments/Robert(sons)/DoubleCoil/Wordlist_Task/RecordedAudio');

    freq = 44100;
    pahandle = PsychPortAudio('Open', [], 2, 0, freq, 2);

    % Preallocate an internal audio recording  buffer with a capacity of 10 seconds:
    PsychPortAudio('GetAudioData', pahandle,10);

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
% seq == 'wordlist-1.txt'; % MAIN WORD LIST
   
    %%%%%Main block loop%%%%%
   for Block = 1:5 

    recordedaudio = [];
    
%%%%%%%%%%%%%%Change Row here if needed!!!!!!!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Row = 1; %indexing rows in the sequence file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
    WaitSecs(2);
    for Trial = 1:12 %%%%%Main trial loop%%%%%
        pres = tab(Row,:);
        wordpres = sprintf(pres);
        Screen('TextSize', mainwindow, fontsize_w);   
        DrawFormattedText (mainwindow, wordpres, 'center', 'center', textcolor);
        Screen('Flip', mainwindow);
        WaitSecs(2);
        %Increment
        Row = Row + 1; %jump to the next row of the sequence file
        
    end
    
    Screen('TextSize', mainwindow, [fontsize_in]); 
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
% if ( keyCode( KbName('esc'))==1 )
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
  
   %                                 MEPs STIMULATION 
 Screen('TextSize',mainwindow,[fontsize_fc]);
DrawFormattedText (mainwindow, '+', 'center', 'center', textcolor, '','','', 2);
Screen('Flip', mainwindow);


  
     WaitSecs(5);
     if (Block ==1)
     for Trial = 1:20 %%%%%TMS stimuli%%%%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin13_WL_MEPs_1,false,port,base_addr) %Write trigger
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
           ISI = a + (b-a).*rand(n,1) 
        
         WaitSecs(ISI);
     end   
            elseif (Block == 2)
                for Trial = 1:20
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin15_WL_MEPs_2,false,port,base_addr) %Write trigger
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            ISI = a + (b-a).*rand(n,1) 
        
         WaitSecs(ISI);

                end 
     
                elseif (Block == 3)
                for Trial = 1:20
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin21_WL_MEPs_3,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            ISI = a + (b-a).*rand(n,1) 
        
         WaitSecs(ISI);
                end 
        
        elseif (Block == 4)
                for Trial = 1:20
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin23_WL_MEPs_4,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            ISI = a + (b-a).*rand(n,1) 
        
         WaitSecs(ISI);
                end 
     
                elseif (Block == 5)
                for Trial = 1:20
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin29_WL_MEPs_5,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            ISI = a + (b-a).*rand(n,1) 
        
         WaitSecs(ISI);
     end 
     end
     
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
    if (Block ==1)
     for Trial = 1:12 %%%  Conditioning Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin9_WL_CS_1 ,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            WaitSecs(0.009); %%% Inter-Stimulus Interval %%%        10ms = 0.01sec >> However, after we close the first pin, we ask to wait 1ms, so here we just ask to wait (10 - 1ms = 9ms = 0.009s)
            
            
            
            %%%  Test Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin12_WL_TS_1,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            
           x1 = a + (b-a).*rand(n,1) %%% Inter-Trial Interval %%%
        
         WaitSecs(x1);
     
    
        end %end of block loop!

        elseif (Block ==2)
     for Trial = 1:12 %%%  Conditioning Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin11_WL_CS_2 ,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            WaitSecs(0.006); %%% Inter-Stimulus Interval %%%        6ms = 0.006sec
            
            
            
            %%%  Test Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin14_WL_TS_2,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            
           x1 = a + (b-a).*rand(n,1) %%% Inter-Trial Interval %%%
        
         WaitSecs(x1);
     
    
        end %end of block loop!
        
        
                elseif (Block ==3)
     for Trial = 1:12 %%%  Conditioning Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin17_WL_CS_3 ,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            WaitSecs(0.006); %%% Inter-Stimulus Interval %%%        6ms = 0.006sec
            
            
            
            %%%  Test Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin20_WL_TS_3,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            
           x1 = a + (b-a).*rand(n,1) %%% Inter-Trial Interval %%%
        
         WaitSecs(x1);
     
    
        end %end of block loop!
        
        
                elseif (Block ==4)
     for Trial = 1:12 %%%  Conditioning Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin19_WL_CS_4 ,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            WaitSecs(0.006); %%% Inter-Stimulus Interval %%%        6ms = 0.006sec
            
            
            
            %%%  Test Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin22_WL_TS_4,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            
           x1 = a + (b-a).*rand(n,1) %%% Inter-Trial Interval %%%
        
         WaitSecs(x1);
     
    
        end %end of block loop!
        

     
             elseif (Block ==5)
     for Trial = 1:12 %%%  Conditioning Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin25_WL_CS_5 ,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            WaitSecs(0.006); %%% Inter-Stimulus Interval %%%        6ms = 0.006sec
            
            
            
            %%%  Test Stimulus Trigger  %%%
            pp(uint8(2:9),resettrig,false,port,base_addr); %close
            pp(uint8(2:9),Pin28_WL_TS_5,false,port,base_addr) %Write trigger 8
            WaitSecs (0.001);
            pp(uint8(2:9),resettrig,false,port,base_addr);
            
            
            
           x1 = a + (b-a).*rand(n,1) %%% Inter-Trial Interval %%%
        
         WaitSecs(x1);
     
    
        end %end of block loop!
        
    end 
        
             if Block < 5
     Screen('TextSize', mainwindow, [fontsize_in]); 
    DrawFormattedText (mainwindow, 'Take a breather! The next round is about to start.', 'center', 'center', textcolor, '','','', 2);
    Screen('Flip', mainwindow);       
     end
        
    Block = Block+1;
    
    
   %WaitSecs(5)
   % Instead of a fixd break, press the spaceba
   % this was we can avoid the new round to start without them be focusd on the screen
   %they just need to pressa the spacebar to continue
   
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
            WaitSecs(0.5); %to avoid too short starting time and unwanted button press
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end %end of block loop!
 
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


EndTime=GetSecs;


%%
Screen('TextSize', mainwindow, [fontsize_in]); 
DrawFormattedText (mainwindow, 'End of the task. Thank you!', 'center', 'center', textcolor, '','','', 2);
Screen('Flip', mainwindow);
WaitSecs(3);

%%
ShowCursor(); %shows the cursor
Screen('CloseAll'); %Closes Screen