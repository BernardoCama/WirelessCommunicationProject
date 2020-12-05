function [y,w] = Conventional_BF(Geometry, Pars, doas, x)

steervec = phased.SteeringVector('SensorArray',Geometry.BSarray);

s = steervec(Pars.fc, doas);

w = s.*1/prod(Geometry.BSarray.Size);

y = (w.')*x';

y = y';
end
