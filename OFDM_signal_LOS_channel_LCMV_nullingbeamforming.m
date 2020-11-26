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
nSymbols1 = 150;

% Pilots symbols positioning at first antenna
pilot_indices1 = [2]';

% Band Carriers
NumGuardBandCarriers = [1;1];

% Nfft for OFDM modulation
nfft  = 12;

% CyclicPrefixLength
CyclicPrefixLength  = [0];

% First OFDM modulator:
ofdmMod1 = comm.OFDMModulator('FFTLength', nfft, ...
    'NumGuardBandCarriers', NumGuardBandCarriers, ... % Default values
    'InsertDCNull', false, ...
    'CyclicPrefixLength', CyclicPrefixLength, ...
    'Windowing', false, ...
    'NumSymbols', nSymbols1, ...
    'NumTransmitAntennas', 1, ...
    'PilotInputPort', true, ...
    'PilotCarrierIndices', pilot_indices1);

% QAM modulation order:
M1 = 4;

% Generation of random bits:
bitInput1 = randi([0 1], (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))) * nSymbols1 * 2, 1);

% Mudulation of bit_in_1 with QAM modulator:
dataInput1 = qammod(bitInput1, M1, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);

% Preparing dataInput1 for OFDM modulation:
ofdmInfo1 = info(ofdmMod1);
ofdmSize1 = ofdmInfo1.DataInputSize;
dataInput1 = reshape(dataInput1, ofdmSize1); 

% OFDM modulation:
pilotInput1 = ones(1, nSymbols1, 1);
waveform1 = ofdmMod1(dataInput1, pilotInput1);

% Used by the channel to determine the delay in number of samples:
Fs1 = 180000;

% Pilot indices for second modulator:
pilot_indices2 = pilot_indices1 + 8;

% Definition of a second OFDM modulator (different pilot carrier indices and different number of symbols):
nSymbols2 = nSymbols1;
ofdmMod2 = comm.OFDMModulator('FFTLength', nfft, ...
    'NumGuardBandCarriers', NumGuardBandCarriers, ...
    'InsertDCNull', false, ...
    'CyclicPrefixLength', CyclicPrefixLength, ...
    'Windowing', false, ...
    'NumSymbols', nSymbols2, ...
    'NumTransmitAntennas', 1, ...
    'PilotInputPort', true, ...
    'PilotCarrierIndices', pilot_indices2);

% Definition of a second M-QAM modulator:
M2 = M1;

% Generation of a second random string of bits:
bitInput2 = randi([0 1], (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))) * nSymbols2 * 2, 1);

% QAM modulation of bitInput2:
dataInput2 = qammod(bitInput2, M2, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);

% Preparing QAM modulated singal for OFDM modulation:
ofdmInfo2 = info(ofdmMod2);
ofdmSize2 = ofdmInfo2.DataInputSize;
dataInput2 = reshape(dataInput2, ofdmSize2);

% OFDM modulation:
pilotInput2 = ones(1, nSymbols2, 1);
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
% 
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


%% LCMV beamformer
% steeringvector = phased.SteeringVector(...
%     'SensorArray',Geometry.BSarray,...
%     'PropagationSpeed',Pars.c);
% beamformer = phased.LCMVBeamformer('DesiredResponse',1,...
%     'TrainingInputPort',true,'WeightsOutputPort',true);
% beamformer.Constraint = steeringvector(Pars.fc,doas(:,1));
% beamformer.DesiredResponse = 1;
% [arrOut,w] = beamformer(chOut,chOutInt);
[arrOut,w] = Nullsteering_BF(Geometry, Pars, DoAs, chOut);


% % Plot Output of Beamformer
% figure;
% plot([0:1/Fs1:length(abs(arrOut))/Fs1-1/Fs1],abs(arrOut)); axis tight;
% title('Output of Beamformer');
% xlabel('Time (s)');ylabel('Magnitude (V)');


% Plot array pattern at azimuth = 0°
% figure;
% pattern(Geometry.BSarray,Pars.fc,[-180:180],0,...
%     'PropagationSpeed',Pars.c,...
%     'Type','powerdb',...
%     'CoordinateSystem','polar','Weights',w)
% 
% figure;
% pattern(Geometry.BSarray,Pars.fc,[-180:180],0,...
%     'PropagationSpeed',Pars.c,...
%     'Type','powerdb',...
%     'CoordinateSystem','rectangular','Weights',w)


%% OFDM demodulation

% OFDM demodulated signal with beamformer at first antenna:
%[chOut_BF, pilotOut_BF] = step(ofdmDemod1, arrOut);
chOut_BF= ofdmDemod1(arrOut);


 
%% Channel estimation

% OFDM symbol used to train
n_training = 10;

% Number of points for the fft for channel estimation (pilotIn: length(pilot_indices) x nSymbols):
%nfft_ch = 2^nextpow2(length(chOut_BF(:,1)));
nfft_ch = length(chOut_BF(:,1));

% FFT of received sequence:
Y = fft(chOut_BF(:,1:n_training));

% FFT of know sequence:
X = fft(dataInput1(:,1:n_training));

% Channel frequency response:
H_estimated = Y ./ (X);

% Channel impulse response:
h_estimated = ifft(H_estimated);



%% Channel equalization (MMSE algorithm)

% Equalization filter
G = zeros(size(H_estimated,1),size(H_estimated,2));

for f=1:n_training
    G(:,f) = conj(H_estimated(:,f)).*(H_estimated(:,f)'*H_estimated(:,f) + 1/Pars.SNR)^-1;

end

% Considering H as ideal channel (in all frequences) that introduces only a phase shift
G = [mean(mean(G,2),1)];

% Considering H as ideal channel (only in each subcarrier) that introduces only a phase shift
% G = [mean(G,2)];



% Signal to be equalized:
% g = ifft(G);
% chOut_equal = zeros(size(chOut_BF,1),size(chOut_BF,2));
% for f=1:nfft_ch
%     G = fft(g(f,:), size(chOut_BF,2));
%     Y = fft(chOut_BF(f,:), size(chOut_BF,2));
%     % chOut_equal(f,:) = conv(g(f,:), chOut_BF(f,:));
%     % chOut_equal(f,:) = cconv(g(f,:),chOut_BF(f,:),size(chOut_BF,2));
%     chOut_equal(f,:) = ifft(G.*Y);
% end


chOut_equal = (G).*chOut_BF(:,n_training+1:end);
chOut_equal = chOut_equal(:);




%{
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
%}




%% Scattergraph to see impact of BF

% No beamformer without equalization:
out = ofdmDemod1(chOut(:,1)); % first antenna
figure;

x = real(out);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(out);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
scatter(x,y);


% With beamformer without equalization
out = ofdmDemod1(arrOut);
figure;

x = real(out);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(out);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
scatter(x,y);

dataOut_beam = qamdemod(out,M1,'OutputType','bit');
dataOut_beam = dataOut_beam(:);
[numErrorsG_beam,berG_beam] = biterr(bitInput1,dataOut_beam(1:end))


% With beamformer with equalization
figure;

x = real(chOut_equal(:));
y = imag(chOut_equal(:));
scatter(x,y);

dataOut_beam = qamdemod(chOut_equal,M1,'OutputType','bit');
dataOut_beam = dataOut_beam(:);
[numErrorsG_beam,berG_beam] = biterr(bitInput1(length(bitInput1)-length(dataOut_beam)+1:end),dataOut_beam(1:end))


