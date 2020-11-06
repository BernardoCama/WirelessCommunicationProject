function [angle] = ZoA(s,d)
% s = Base Station
angle = atan2(dist3D(s,d), s(1,3) - d(1,3) )*180/pi;
end

