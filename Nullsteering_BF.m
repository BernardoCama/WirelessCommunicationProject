function [y,w] = Nullsteering_BF(Geometry, Pars, doas, x)

steervec = phased.SteeringVector('SensorArray',Geometry.BSarray);

for i=1:length(doas(1,:)) % (1 + N_interf)

    s(:,i) = steervec(Pars.fc, doas(:,i));
    %doas(:,i)

end

g1 = zeros(1,length(doas(1,:)));
g1(1) = 1;

w_H = g1*pinv(s);

w = w_H';

y = w_H*x.';

y = y.';

% s.t. w_H*s = [1 0*N_interf]
end

