function [y,w] = MVDR_BF(Geometry, Pars, doas, x)

steervec = phased.SteeringVector('SensorArray',Geometry.BSarray);

s = steervec(Pars.fc, doas);

% Alternative
% Signal_power = xcorr(x(:),x(:),'unbiased');
% Signal_power = fftshift(abs(Signal_power));
% Noise_power = Signal_power/db2pow(Pars.SNR);
% R = sensorcov(Geometry.BSAntennaPos, doas, Noise_power(1));
R = x'*x;

w = (inv(R)*s)/(s'*inv(R)*s);

w_H = w';

y = (w_H)*x.';

y = y.';

% s.t. w_H*s = 1
end

