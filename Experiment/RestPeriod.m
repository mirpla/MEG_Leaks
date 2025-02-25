function RestPeriod(RestDur, Key,deviceIndex, savepath)

RestStart = GetSecs;
RestTimer = 0;
keyCode   = 0;
while ((RestTimer >= RestStart+RestDur)  ~=1)  && keyCode ~= Key  % Checks continuously whether time has passed or ESCAPE has been pressed to skip/cancel it
    [~, keyVec, ~] = KbWait(deviceIndex, 2, RestStart+RestDur);
    keyCode = find(keyVec);
    
    RestTimer = GetSecs;
end

restsecs = RestTimer - RestStart;

if restsecs < RestDur
    warning('Rest Period cut short') 
end 

Restults = [sprintf('%16.f',RestStart), sprintf('%16.f',RestTimer), sprintf('%16.f',RestDur), sprintf('%16.f',restsecs < RestDur)];

dlmwrite(savepath, str2num(Restults) , '-append','delimiter', '\t','precision',16);