function yout = phaseCorrection2(y, d)



ay = angle(y((real(y) > 0) & (imag(y) > 0)));
ad = angle(y((real(d) > 0) & (imag(d) > 0)));

theta = ay(1:length(ad)) - ad;

theta = mean(theta);
yout = y * exp(-1i*theta);

end

% theta = mean(a) - pi/4;
% theta = +angle(y(1:length(d)))-angle(d);
% 
% theta = mean(theta);
% yout = y * exp(-1i*theta);