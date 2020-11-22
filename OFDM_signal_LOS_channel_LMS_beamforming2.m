%% Preparing 

clear all;
close all;
clc;

%% Defining geometry and pars

% CArrier frequency and wavelength
Pars.fc = 1e9;
Pars.c = physconst('LightSpeed');
Pars.lambda = Pars.c / Pars.fc;

% BS position (macrocell with high 25m):
Geometry.BSPos = [0, 0, 25];

% First veichle (V1):
Geometry.V1PosStart = [70, -100, 1.5]; % start
Geometry.V1PosEnd = [70, 100, 1.5];    % end

% Second veichle (V2):
Geometry.V2PosStart = [200, -50, 1.5]; % start 
Geometry.V2PosEnd = [10, -50, 1.5];    % end

% Interferents:
Geometry.I1Pos = [10, -210, 1.5]; 
Geometry.I2Pos = [-150, 100, 1.5];

% Distance covered by veichles:
Geometry.T1 = dist3D(Geometry.V1PosStart, Geometry.V1PosEnd);  % V1
Geometry.T2 = dist3D(Geometry.V2PosStart, Geometry.V2PosEnd);  % V2

% Initial distances between veichles and BS:
Geometry.DistV1Start = dist3D(Geometry.V1PosStart, Geometry.BSPos); % BS and V1
Geometry.DistV2Start = dist3D(Geometry.V2PosStart, Geometry.BSPos); % BS and V2

% Initial DoA = [AoA ZoA] (ZoA = 90 - elevation angle) for the two vehicles:
Geometry.AOAV1Start = AoA(Geometry.BSPos, Geometry.V1PosStart);
Geometry.ZOAV1Start = ZoA(Geometry.BSPos, Geometry.V1PosStart);
Geometry.DOAV1Start = [Geometry.AOAV1Start Geometry.ZOAV1Start]; % DoA of V1

Geometry.AOAV2Start = AoA(Geometry.BSPos, Geometry.V2PosStart);
Geometry.ZOAV2Start = ZoA(Geometry.BSPos, Geometry.V2PosStart);
Geometry.DOAV2Start = [Geometry.AOAV2Start Geometry.ZOAV2Start]; % DOA of V2

% Defining a rectangular Nant x Nant antenna array with antenna spacing = lambda/2:
Nant = 4;
Geometry.BSarray = phased.URA('Size', [4 4], ...
    'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'x');

% Getting position antenna array:
Geometry.BSAntennaPos = getElementPosition(Geometry. BSarray);

% Creating conformal antenna array:
Geometry.confarray = phased.ConformalArray('ElementPosition', Geometry.BSAntennaPos);


%% Generation of ODFM modulators and demodulators, M-QAM modulators and waveforms

% Number of ODFM symbols:
nSymbols1 = 80;

% Pilots symbols positioning at first antenna (default values)
pilot_indices1 = [12, 26, 40, 54]';

% Nfft for OFDM modulation
% nfft  = 256;
nfft  = 128;

% First OFDM modulator:
ofdmMod1 = comm.OFDMModulator('FFTLength', nfft, ...
    'NumGuardBandCarriers', [6; 5], ... % Default values
    'InsertDCNull', false, ...
    'CyclicPrefixLength', [0], ...
    'Windowing', false, ...
    'NumSymbols', nSymbols1, ...
    'NumTransmitAntennas', 1, ...
    'PilotInputPort', true, ...
    'PilotCarrierIndices', pilot_indices1);

% QAM modulation order:
M1 = 4;

% Generation of random bits (15 ?):
bitInput1 = randi([0 1], (nfft - 15) * nSymbols1 * 2, 1);

% Mudulation of bit_in_1 with QAM modulator:
dataInput1 = qammod(bitInput1, M1, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);

% Preparing dataInput1 for OFDM modulation:
ofdmInfo1 = info(ofdmMod1);
ofdmSize1 = ofdmInfo1.DataInputSize;
dataInput1 = reshape(dataInput1, ofdmSize1); 

% OFDM modulation:
pilotInput1 = ones(4, nSymbols1, 1);
waveform1 = ofdmMod1(dataInput1, pilotInput1);

% Used by the channel to determine the delay in number of samples:
Fs1 = 180000;

% Pilot indices for second modulator:
pilot_indices2 = pilot_indices1 + 5;

% Definition of a second OFDM modulator (different pilot carrier indices and different number of symbols):
nSymbols2 = nSymbols1;
ofdmMod2 = comm.OFDMModulator('FFTLength', nfft, ...
    'NumGuardBandCarriers', [6; 5], ...
    'InsertDCNull', false, ...
    'CyclicPrefixLength', [0], ...
    'Windowing', false, ...
    'NumSymbols', nSymbols2, ...
    'NumTransmitAntennas', 1, ...
    'PilotInputPort', true, ...
    'PilotCarrierIndices', pilot_indices2);

% Definition of a second M-QAM modulator:
M2 = M1;

% Generation of a second random string of bits:
bitInput2 = randi([0 1], (nfft - 15) * nSymbols2 * 2, 1);

% QAM modulation of bitInput2:
dataInput2 = qammod(bitInput2, M2, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);

% Preparing QAM modulated singal for OFDM modulation:
ofdmInfo2 = info(ofdmMod2);
ofdmSize2 = ofdmInfo2.DataInputSize;
dataInput2 = reshape(dataInput2, ofdmSize2);

% OFDM modulation:
pilotInput2 = ones(4, nSymbols2, 1);
waveform2 = ofdmMod2(dataInput2, pilotInput2);

% Used by the channel to determine the delay in number of samples:
Fs2 = 180000; 

% OFDM demodulators definition:
ofdmDemod1 = comm.OFDMDemodulator(ofdmMod1);
ofdmDemod2 = comm.OFDMDemodulator(ofdmMod2);

% Visualizing OFDM mapping:
% showResourceMapping(ofdmMod1);
% title('OFDM modulators (1 = 2)');

%% LoS Channel 

% Generation of LoS channel:
h_env = phased.FreeSpace('SampleRate', Fs1, ...
    'OperatingFrequency', Pars.fc);

% Velocities of veichles:
vel1 = [0;0;0];
vel2 = [0;0;0];

% Response of waveform1 passing through channel:
w1 = step(h_env, waveform1, ...
    Geometry.V1PosStart', ...
    Geometry.BSPos', ...
    vel1, ...
    vel2);

% Response of waveform2 passing through channel:
w2 = step(h_env, waveform2, ...
    Geometry.V2PosStart', ...
    Geometry.BSPos', ...
    vel1, ...
    vel2);

% Calucation of received wavefrom1 (attention to dimension of waveforms):
chOut = collectPlaneWave(Geometry.BSarray, [w1 w2], ...
        [Geometry.DOAV1Start', Geometry.DOAV2Start'], Pars.fc);

% Adding AWGN noise to waveform:
Pars.SNR = 20; % in dB
% std_dev = sqrt(1 / Pars.SNR);
% noise = std_dev * randn(size(chOut)); 
% chOut = chOut + noise;
chOut_noise = awgn(chOut, Pars.SNR, 'measured');
noise = chOut_noise - chOut;
chOut = chOut_noise;


%% Estimation of DoA (MUSIC algorithm)

% Generation of MUSIC  DoA estimator:
estimator = phased.MUSICEstimator2D( ...
    'SensorArray', Geometry.BSarray, ...
    'OperatingFrequency', Pars.fc, ...
    'ForwardBackwardAveraging', true, ...
    'NumSignalsSource', 'Property', ...
    'DOAOutputPort', true, ...
    'NumSignals', 2, ...
    'AzimuthScanAngles', -90:.1:90, ...
    'ElevationScanAngles', 0:0.5:90);

% Estimation od DoA of singal in output from the channel:
[~, DoAs] = estimator(chOut);

% THIS IS FOR THIS SPECIFIC CASE -> NEED TO FIX FOR GENERAL CASE:
DoAs(1,:) = -(DoAs(1,:) - 180);
temp1 = DoAs(:,1);
DoAs(:,1) = DoAs(:,2);
DoAs(:,2) = temp1;

% Plotting:
% figure();
% plotSpectrum(estimator);

%% LMS beamformer
% We use the first half of the received signal for the LMS algorithm, 
% finding then the weights to be assigned to each antenna and applying the
% to the received singal for better reception.

% Training sequence length:
nTrain = round(length(chOut(:,1)) / 2);

% Applying LMS beamformer:
[arrOut, w] = LMS_BF(Geometry, Pars, DoAs(:, 1), chOut, waveform1(1:nTrain, :));

%% OFDM demodulation

% OFDM demodulated signal with beamformer at first antenna:
[chOut_BF, pilotOut_BF] = step(ofdmDemod1, arrOut);


%% QAM demodulation for 1 pilot

pilotOut_BF_outQAM = qamdemod(pilotOut_BF(1, :), M1, 'OutputType', 'bit');
pilotOut_BF_outQAM = pilotOut_BF_outQAM(1, :);
 
%% Channel estimation (pilot-based)

% Number of points for the fft for channel estimation (pilotIn: length(pilot_indices) x nSymbols):
nfft_ch = 2^nextpow2(length(pilotInput1(1, :)));

% FFT of original pilots:
X_pilots = fft(pilotInput1(1, :), nfft_ch);

% FFT of received pilots (wiht BF):
Y_pilots_BF = fft(pilotOut_BF_outQAM(1, :), nfft_ch);

% Channel frequency response:
H_estimated = Y_pilots_BF ./ (X_pilots + 0.001);

% Channel impulse response:
h_estimated = ifft(H_estimated);

%% Channel equalization (MMSE algorithm)
% Symbols of training
n_training = 100;
dataInput1 = dataInput1(:);
chOut_BF = chOut_BF(:);

% Autocorrelation matrix of noise:
% R_noise = noise * noise';
% R_noise = mean(abs(chOut_BF).^2)/Pars.SNR;

% Autocorrelation of sent signal:
% waveform = waveform1 + waveform2;
% R_wf = waveform * waveform';
% R_dataInput1 = mean(abs(dataInput1(1:n_training)).^2);

% Equalized signal: 
% chOut_eq = (h_estimated' * inv(R_noise) * h_estimated + inv(R_wf)) * h_estimated' * inv(R_noise) * waveform;

% Equalization filter
G = H_estimated'.*(H_estimated*H_estimated.' + 1/Pars.SNR)^-1;

% Signal to be equalized:
chOut_to_equal = chOut_BF(n_training + 1 : end);
len = length(chOut_to_equal);

% Window for equalzation:
winLen = 60;
win = hann(winLen);
hopsize = winLen / 2;

nFrame = floor((len - winLen) / hopsize);

nG = length(G);
conv_len = len + nG - 1;

chOut_equal = zeros(conv_len, 1);

nfft = len + nG - 1;

g = ifft(G);
G = fft(g, nfft);

for i = 1 : nFrame

    start = (i-1) * hopsize + 1;
    stop = (i-1) * hopsize + winLen;
    x = chOut_to_equal(start:stop) .* win;
    
    X = fft(x, nfft);
    
    Y = X .* G;
    y = ifft(Y);
    
    start_ola = (i-1) * hopsize + 1;
    stop_ola = (i-1) * hopsize + (winLen + nG - 1);
    
    y = y(1 : (winLen + nG - 1));
    chOut_equal(start_ola:stop_ola) = chOut_equal(start_ola:stop_ola) + y;
 
end

chOut_equal = chOut_equal(1:len);


%% QAM demodulation

% No beamformer:

% Beamformer:



% With beamformer
figure;

x = real(chOut_equal);
y = imag(chOut_equal);
scatter(x,y);

% dataOut_beam = qamdemod(chOut_equal,M,'OutputType','bit');
% dataOut_beam = dataOut_beam(:);
% [numErrorsG_beam,berG_beam] = biterr(in1(1:end),dataOut_beam(1:end))

%% Plots for result comparison




