function w_out = LOS(waveform, Tx, Rx, Pars);

dist = dist3D(Tx,Rx);
w_out = waveform.*(4*pi*dist/Pars.lambda).^2.*exp(-1i*2*pi*dist/Pars.lambda);

end

