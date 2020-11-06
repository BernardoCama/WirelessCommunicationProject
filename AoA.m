function [angle] = AoA(s,d)
% s = Base Station
angle = atan2(s(1,2) - d(1,2), s(1,1) - d(1,1) )*180/pi;
end

