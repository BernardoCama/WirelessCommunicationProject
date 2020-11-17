function [y,w] = Conventional_BF(Geometry, Pars, doas, x)

steervec = phased.SteeringVector('SensorArray',Geometry.BSarray);

s = steervec(Pars.fc, doas);

w = s.*1/prod(Geometry.BSarray.Size);

y = (w.')*x';

y = y';
end

% doas = 2 x 1 direction of signal
% x = 1800 x 16 signal received from each antenna
% w = 16 x 1 
% y = 1800 x 1 signal in output