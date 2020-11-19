function yout = phaseCorrection(y, d)

NFFT = 2^nextpow2(length(d));
Y = fft(y(1:length(d)), NFFT);
D = fft(d, NFFT);
H = Y./D;
h = ifft(H.^-1);
yout = conv(h,y);
%yout = ifft(fft(y)./H);



% a = angle(y((real(y) > 0) & (imag(y) > 0)));
% a(a < 0.1) = a(a < 0.1) + pi/2;
% theta = mean(a) - pi/4;
% y = y * exp(-1i*theta);
end