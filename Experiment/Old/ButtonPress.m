function ButtonPress(Escape)
[~, IdleButton,~] = KbStrokeWait;
if find(IdleButton) == Escape
    sca;
    %ppdev_mex('CloseAll', 1) 
    ListenChar(0)
    Priority(0);
    error('Experiment cancelled by Experimenter')
end
end 