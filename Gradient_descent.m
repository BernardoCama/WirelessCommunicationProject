function g = Gradient_descent(x, d, training_seq ,   max_iter, g_len)



% autocorrelation of the input 
auto_corr_vec = xcorr(x,x,'unbiased');
mid_point = (length(auto_corr_vec)+1)/2;
c = auto_corr_vec(mid_point:mid_point+g_len-1); % first column of toeplitz 
r = fliplr(auto_corr_vec(mid_point-g_len+1:mid_point));
Rvv_Matrix = toeplitz(c,r);


% Rvv_Matrix = x'*x;
% Rvv_Matrix = Rvv_Matrix(1:g_len, 1:g_len);


% cross correlation 
cross_corr_vec = xcorr(d,x(1:training_seq),'unbiased');
MID_POINT = (length(cross_corr_vec)+1)/2;
cross_corr_vec = cross_corr_vec(MID_POINT:MID_POINT+g_len-1).';

% cross_corr_vec = xcorr(d,x(1:training_seq),'unbiased');
% cross_corr_vec = cross_corr_vec(1:g_len);

%---------------------------------------------
max_step_size = 1/(max(eig(Rvv_Matrix)));% maximum step size
%max_step_size = 2/(sum(eig(Rvv_Matrix)));% maximum step size
step_size = 0.125*max_step_size;
g = zeros(g_len,1);
for i= 1:max_iter
    g = g+step_size*(cross_corr_vec - Rvv_Matrix*g);
end
g=g.'; % now a row vector


end

