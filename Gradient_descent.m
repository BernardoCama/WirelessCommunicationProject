function g = gradient_descent(x, d, tolerance, min_perturb, max_iter)

d = d.';
x = x.';

X = fft(x);
D = fft(d);

% Initialization;
g_ = zeros(size(d,1)*max_iter, length(x));
g_(1, :) = ifft(X ./ (D + 0.001));


Rx = x*x';

r_xd = x.*(conj(d));

mu1 = 1/trace(Rx);


% r_xd 100 x 16
% g 100 x 16
% Rx 100 x 100

iter = 0;
    
for k = 1 : max_iter
    
    for i = 1 : size(d,1)-1
        
        iter = iter + 1;
        g_(iter+1,:) = g_(iter,:) + mu1.*(Rx(i,i).*g_(iter,:) - r_xd(i,:));

    end

    mu1 = mu1 * (1/k)^0.3;
    
end

g = g_(end/2, :).';

end

% doas = 2 x (1 + N_interf) direction of signal + interf
% s = 2 x (1 + N_interf)
% x = 1800 x 16 signal received from each antenna
% x = 100 x 16
% d = 100 x 16
% g = 16 x 1 
% y = 1800 x 1 signal in output