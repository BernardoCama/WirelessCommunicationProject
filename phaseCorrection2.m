function yout = phaseCorrection2(y, d)



% a = angle(y((real(y) > 0) & (imag(y) > 0)));
% a(a < 0.1) = a(a < 0.1) + pi/2;
% theta = mean(a) - pi/4;
theta = +angle(y(1:length(d)))-angle(d);

theta = mean(theta);
yout = y * exp(-1i*theta);

end