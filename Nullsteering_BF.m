function [y,w] = Nullsteering_BF(Geometry, Pars, doas, x)

steervec = phased.SteeringVector('SensorArray',Geometry.BSarray);

for i=1:length(doas(1,:)) % (1 + N_interf)

    s(:,i) = steervec(Pars.fc, doas(:,i));

end

g1 = zeros(1,length(doas(1,:)));
g1(1) = 1;

w_H = g1*(s.')*((s*s.')^-1);

w = w_H.';

y = (w_H)*x';

y = y';
end

% doas = 2 x (1 + N_interf) direction of signal + interf
% s = 2 x (1 + N_interf)
% x = 1800 x 16 signal received from each antenna
% w = 16 x 1 
% y = 1800 x 1 signal in output