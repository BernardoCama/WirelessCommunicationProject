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
Geometry.Nant = 16;
Geometry.BSarray = phased.URA('Size', [Geometry.Nant Geometry.Nant], ...
    'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'x');

% Getting position antenna array:
Geometry.BSAntennaPos = getElementPosition(Geometry. BSarray);

% Creating conformal antenna array:
Geometry.confarray = phased.ConformalArray('ElementPosition', Geometry.BSAntennaPos);


%% Generation of ODFM modulators and demodulators, M-QAM modulators and waveforms

% Number of ODFM symbols:
nSymbols1 = 100;

% Pilots symbols positioning at first antenna
pilot_indices1 = [2]';

% Band Carriers
NumGuardBandCarriers = [1;1];

% Nfft for OFDM modulation
nfft  = 64;

% CyclicPrefixLength
CyclicPrefixLength  = [4];

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
bitInput1 = randi([0 1], (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))) * nSymbols1 * log2(M1), 1);

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
bitInput2 = randi([0 1], (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))) * nSymbols2 * log2(M2), 1);

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
w1 = LOS(waveform1, Geometry.V1PosStart, Geometry.BSPos, Pars);
w2 = LOS(waveform2, Geometry.V2PosStart, Geometry.BSPos, Pars);

% Velocities of veichles:
vel1 = [0;0;0];
vel2 = [0;0;0];


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


%% OFDM demodulation and LMS Beamforming
% Signal after fft
chOut_OFDMdem = zeros( Geometry.Nant^2 , (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);

for ant = 1: size(chOut,2)
    chOut_OFDMdem(ant,:,:) = ofdmDemod1(chOut(:,ant));

end

chOut_OFDMdem_BF = zeros (size(chOut_OFDMdem,2), size(chOut_OFDMdem,3));
w = zeros (size(chOut_OFDMdem,2), size(chOut_OFDMdem,1));
for sc = 1: size(chOut_OFDMdem,2)

        nTrain = round(size(chOut_OFDMdem,3) / 2);

        [chOut_OFDMdem_BF(sc,:), w(sc,:)] = LMS_BF(Geometry, Pars, DoAs(:, 1), squeeze(chOut_OFDMdem(:,sc,:)).', dataInput1(sc, 1:nTrain).');
    
end




%%  Channel equalization
n_training = size(chOut_OFDMdem_BF,2);
    
max_iter = 100;

% Tap of the equalizer
g_len = 2;

chOut_OFDMdem_BF_equal = zeros (size(chOut_OFDMdem_BF));

for sc = 1: size(chOut_OFDMdem,2)
    
    g = Gradient_descent( chOut_OFDMdem_BF(sc,:),  dataInput1(sc,1:n_training), n_training, max_iter, g_len);

    chOut_OFDMdem_BF_ = conv(g,chOut_OFDMdem_BF(sc,:));

    chOut_OFDMdem_BF_equal(sc,:) = chOut_OFDMdem_BF_(1:length(chOut_OFDMdem_BF(sc,:)));

end


%% OFDM demodulation and QAM demodulation

% No beamformer with equalization:
chOut_OFDMdem_1 = squeeze(chOut_OFDMdem(1,:,:)); % first antenna
figure;

x = real(chOut_OFDMdem_1);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(chOut_OFDMdem_1);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
scatter(x,y);
title('Without beamforming, with equalization')

chOut_OFDMdem_1_QAMdem = qamdemod(chOut_OFDMdem_1,M1,'OutputType','bit');
chOut_OFDMdem_1_QAMdem = chOut_OFDMdem_1_QAMdem(:);
[numErrorsG_beam_noequal_nobeam,berG_beam_noequal_nobeam] = biterr(bitInput1,chOut_OFDMdem_1_QAMdem(1:end))




% With beamformer with equalization
chOut_OFDMdem_BF_equal = chOut_OFDMdem_BF_equal;
figure;

x = real(chOut_OFDMdem_BF_equal);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(chOut_OFDMdem_BF_equal);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
scatter(x,y);

chOut_OFDMdem_BF_equal_QAMdem = qamdemod(chOut_OFDMdem_BF_equal,M1,'OutputType','bit');
chOut_OFDMdem_BF_equal_QAMdem = chOut_OFDMdem_BF_equal_QAMdem(:);
[numErrorsG_beam,berG_beam] = biterr(bitInput1(length(bitInput1)-length(chOut_OFDMdem_BF_equal_QAMdem)+1:end),chOut_OFDMdem_BF_equal_QAMdem(1:end))
title('With beamforming and equalization')
