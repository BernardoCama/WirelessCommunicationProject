function [y,w] = Conventional_BF(Geometry, Pars, doas, x)

steervec = phased.SteeringVector('SensorArray',Geometry.BSarray);

s = steervec(Pars.fc, doas);

w = s.*1/prod(Geometry.BSarray.Size);

w_H = w';

y = w_H*x.';

y = y.';
end
