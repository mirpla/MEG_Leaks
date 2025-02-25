
 Eyelink('Initialize')


%%
% -----------------------------------------------------------------
%% START EYETRACKER
% -----------------------------------------------------------------
EyelinkInit;
screenNumber=max(Screen('Screens'));
[weyelink, wRect]=Screen('OpenWindow', screenNumber, 0,[],32,2);
Screen(weyelink,'BlendFunction',GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
el=EyelinkInitDefaults(weyelink);
[v vs]=Eyelink('GetTrackerVersion');
fprintf('Running experiment on a ''%s'' tracker.\n', vs );
Eyelink('Command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA');
Eyelink('command', 'link_event_data = GAZE,GAZERES,HREF,AREA,VELOCITY');
Eyelink('command', 'link_event_filter = LEFT,RIGHT,FIXATION,BLINK,SACCADE,BUTTON');
HideCursor(ScreenWindow);
EyelinkDoTrackerSetup(el);
success=EyelinkDoDriftCorrection(el);
if success~=1
cleanup;
return;
end
% Open 'windowrect' sized window on screen, with background color:
[ScreenWindow, winrect] = PsychImaging('OpenWindow', whichScreen, BackgroundLum, DemoRect);
%PsychColorCorrection('SetEncodingGamma', ScreenWindow, exptdesign.screengamma );
% get eye that's tracked
eye_used1 = 1; % left, 0+1 because we access a matlab array
eye_used2 = 2; % right, 1+1 because we access a matlab array
eye_used = Eyelink('EyeAvailable');
Eyelink('Openfile','track.edf')
% start recording eye position
Eyelink('StartRecording');
fprintf('\n **** Starting eyetracker. ****\n')
% record a few samples before we actually start displaying
WaitSecs(0.1);
% mark zero-plot time in data file
Eyelink('Message', 'SYNCTIME');
for initialization, and 
if EyeTrack == 1
Eyelink('Stoprecording');
try
ListenChar(1);
transferEyeDataToMEG;
ListenChar(2);
catch
fprintf('\nError. Aborting transfer. Try manual transfer.\n')
end
Eyelink('closefile');
Eyelink('Shutdown');
fprintf('\n**** ALL GOOD. Stopping eyetracker. ****\n')
end

% for transferring the data to the folder where individual MEG run is saved. 
% Bear in mind that the file will always have the same name 'track.edf' 
%   so you need to make sure to save it in the run folder or rename it. 

% My code didn't send messages around important events to eyelink, so this is something you should probably add. 
% I think the function to do that is [status =] Eyelink(‘Message’, ‘formatstring’, […]) 
%   but you'll find more info here http://psychtoolbox.org/docs/Eyelink. 
%There should also be sample code in Eyelink demos folder
