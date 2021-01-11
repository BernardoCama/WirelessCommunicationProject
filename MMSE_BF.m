function [y,w] = MMSE_BF(Geometry, Pars, x, d)

x_short = x(1:size(d,1),:);

R = x_short'*x_short;

p = d'*x_short;

w_H = R\(p.');

w_H = w_H.';

w = w_H';

y = x*w;

% y = y.';

end

