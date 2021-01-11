function [y,w] = LMS_BF(Geometry, Pars, doas, x, d)


steervec = phased.SteeringVector('SensorArray',Geometry.BSarray);

s = steervec(Pars.fc, doas);

x_short = x(1:size(d,1),:);

Rx = x_short*x_short';

r_xd = x_short.*(conj(d));

N = 100;

mu1 = 1/trace(Rx);


% r_xd 100 x 16
% w 100 x 16
% Rx 100 x 100
w_ = zeros(size(d,1)*N, prod(Geometry.BSarray.Size));
w_(1,:) = s.*1/prod(Geometry.BSarray.Size);
iter = 0;

for k = 1: N
    for i = 1:size(d,1)-1
        iter = iter + 1;
        w_(iter+1,:) = w_(iter,:) + mu1.*(Rx(i,i).*w_(iter,:) - r_xd(i,:));

    end
    mu1 = mu1 * (1/k)^0.3;
    
end

w = w_(end-2000,:).';

w_H = w';

% w_H = w_H';
% w_H = w_H.';

y = w_H*x.';

y = y.';
end

