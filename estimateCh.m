function h = estimateCh(y, d, n_taps)

y_short = y(1:length(d));
Y_short = fft(y_short);
D = fft(d);

H = Y_short(:,1)./D(:,1);
X_hat_blocks = X_hat_blocks./repmat(H,1,size(X_hat_blocks,2));

end

