function [y,w] = MVDR_BF(Geometry, Pars, doas, x)

steervec = phased.SteeringVector('SensorArray',Geometry.BSarray);

s = steervec(Pars.fc, doas);

% Alternative
% Signal_power = xcorr(x(:),x(:),'unbiased');
% Signal_power = fftshift(abs(Signal_power));
% Noise_power = Signal_power/db2pow(Pars.SNR);
% R = sensorcov(Geometry.BSAntennaPos, doas, Noise_power(1));
R = x'*x;

w_H = (inv(R)*s)/(s'*inv(R)*s);

w_H = w_H.';

w= w_H';

y = x*w;

% y = y.';

% s.t. w_H*s = 1
end

