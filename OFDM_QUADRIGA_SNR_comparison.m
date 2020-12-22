%% Preparing 

clear all;
close all;
clc;

%% Defining geometry and pars

% CArrier frequency and wavelength
Pars.fc = 26e9;
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
Geometry.Nant = 8;
Geometry.BSarray = phased.URA('Size', [Geometry.Nant Geometry.Nant], ...
    'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'z');

% Getting position antenna array:
Geometry.BSAntennaPos = getElementPosition(Geometry. BSarray);

% Creating conformal antenna array:
Geometry.confarray = phased.ConformalArray('ElementPosition', Geometry.BSAntennaPos);



%% 3GPP Parameters
% Frequency spacing
Geometry.delta_f = 60e3;

% Symbol Time
Geometry.Ts = 17.84e-6;


%% Generation of ODFM modulators and demodulators, M-QAM modulators and waveforms

% Number of ODFM symbols:
nSymbols1 = 100;

% Pilots symbols positioning at first antenna
pilot_indices1 = [11]';

% Band Carriers
NumGuardBandCarriers = [1;1];

% Nfft for OFDM modulation
nfft  = 64;

% Cyclic prefix length:
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
%waveform1 = waveform1*3;

% Used by the channel to determine the delay in number of samples:
Fs1 = 180000;

% Pilot indices for second modulator:
pilot_indices2 = pilot_indices1 + 5;

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
waveform2 = waveform2*1;

% Used by the channel to determine the delay in number of samples:
Fs2 = 180000; 

% OFDM demodulators definition:
ofdmDemod1 = comm.OFDMDemodulator(ofdmMod1);
ofdmDemod2 = comm.OFDMDemodulator(ofdmMod2);

% Visualizing OFDM mapping:
% showResourceMapping(ofdmMod1);
% title('OFDM modulators (1 = 2)');

%% 3GPP Channel 
% Velocities of veichles:
vel1 = [0;0;0];
vel2 = [0;0;0];


% Create Layout Object
l = qd_layout;
l.set_scenario('QuaDRiGa_UD2D_LOS');

% Define tx and rx
txArr = qd_arrayant('omni');
rxArr = qd_arrayant('omni');
rxArr.no_elements = Geometry.Nant*Geometry.Nant ;
rxArr.element_position = Geometry.BSAntennaPos;

l.tx_array = txArr;
l.rx_array = rxArr;
l.no_rx = 1;
l.no_tx = 4;

% tx_track1 = qd_track('linear', Geometry.T1, pi/2);
tx_track1 = qd_track('linear', 0, pi/2);
tx_track1.name = 'trackV1';

% tx_track2 = qd_track('linear', Geometry.T2, pi);
tx_track2 = qd_track('linear', 0, pi);
tx_track2.name = 'trackV2';

tx_track1.initial_position = Geometry.V1PosStart';
tx_track2.initial_position = Geometry.V2PosStart';

l.tx_position = [Geometry.V1PosStart', Geometry.V2PosStart',...
    Geometry.I1Pos', Geometry.I2Pos'];
l.tx_track(1,1) = copy(tx_track1);
l.tx_track(1,2) = copy(tx_track2);
l.rx_position = Geometry.BSPos';


% Visualize model
% l.visualize();


% Run the model
l.set_pairing;
chan = l.get_channels;

% Veichle 1
chTaps1 = size(chan(1).delay); % [16 1 34 2]
TS = Geometry.Ts; 
WFlenght = size(waveform1,1);
chOut1 = zeros(chTaps1(1), WFlenght);
TsVect = 0:TS:TS*(WFlenght-1);

for antenna=1:1:chTaps1(1)
    for path=1:1:chTaps1(3)
        inX = TsVect - chan(1).delay(antenna, 1, path, 1);
        inY = interp1(TsVect, waveform1, inX, 'pchip');
        chOut1(antenna, :) = inY * chan(1).coeff(antenna, 1, path, 1)...
            +chOut1(antenna, :);
    end
end


% Veichle 2
chTaps2 = size(chan(2).delay); % [16 1 34 2]
TS = Geometry.Ts; 
WFlenght = size(waveform2,1);
chOut2 = zeros(chTaps2(1), WFlenght);
TsVect = 0:TS:TS*(WFlenght-1);

for antenna=1:1:chTaps2(1)
    for path=1:1:chTaps2(3)
        inX = TsVect - chan(2).delay(antenna, 1, path, 1);
        inY = interp1(TsVect, waveform2, inX, 'pchip');
        chOut2(antenna, :) = inY * chan(2).coeff(antenna, 1, path, 1)...
            +chOut2(antenna, :);
    end
end

% Output of the channel without noise:
chOut1 = transpose(chOut1);
chOut2 = transpose(chOut2);
chOut = chOut1 + chOut2;


%% Definition of the DoA MUSIC estimator:

estimator = phased.MUSICEstimator2D( ...
    'SensorArray', Geometry.BSarray, ...
    'OperatingFrequency', Pars.fc, ...
    'ForwardBackwardAveraging', true, ...
    'NumSignalsSource', 'Property', ...
    'DOAOutputPort', true, ...
    'NumSignals', 2, ...
    'AzimuthScanAngles', -180:.5:180, ...
    'ElevationScanAngles', 0:0.5:90);

%% Definition of working parameters and outputs

% SNR vector:
Pars.SNR = 1:10;

% Definition of output of the channel woth noise:
chOut_noise = zeros(size(waveform1, 1), Geometry.Nant^2, length(Pars.SNR));
chOut1_noise = zeros(size(waveform1, 1), Geometry.Nant^2, length(Pars.SNR));
chOut2_noise = zeros(size(waveform1, 1), Geometry.Nant^2, length(Pars.SNR));
noise = zeros(size(waveform1, 1), Geometry.Nant^2, length(Pars.SNR));

% Definition of output SNR vectors:
SNR_out_simple_BF = zeros(1, length(Pars.SNR));
SNR_out_nulling_BF = zeros(1, length(Pars.SNR));
SNR_out_mvdr_BF = zeros(1, length(Pars.SNR));
SNR_out_lms_BF = zeros(1, length(Pars.SNR));
SNR_out_mmse_BF = zeros(1, length(Pars.SNR));

% Set of outputs of the differen beamformers (overall signal):
chOut_simple_BF = zeros(size(waveform1, 1), length(Pars.SNR)); 
chOut_nulling_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut_mvdr_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut_lms_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut_mmse_BF = zeros(size(waveform1, 1), length(Pars.SNR));

% Set of outputs of the differen beamformers (signal from V1):
chOut1_simple_BF = zeros(size(waveform1, 1), length(Pars.SNR)); 
chOut1_nulling_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut1_mvdr_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut1_lms_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut1_mmse_BF = zeros(size(waveform1, 1), length(Pars.SNR));

% Set of outputs of the differen beamformers (signal from V2):
chOut2_simple_BF = zeros(size(waveform1, 1), length(Pars.SNR)); 
chOut2_nulling_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut2_mvdr_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut2_lms_BF = zeros(size(waveform1, 1), length(Pars.SNR));
chOut2_mmse_BF = zeros(size(waveform1, 1), length(Pars.SNR));

% Set of weigths of the differen beamformers:
w_simple_BF = zeros((Geometry.Nant)^2, length(Pars.SNR));
w_nulling_BF = zeros((Geometry.Nant)^2, length(Pars.SNR));
w_mvdr_BF = zeros((Geometry.Nant)^2, length(Pars.SNR));
w_lms_BF = zeros((Geometry.Nant)^2, length(Pars.SNR));
w_mmse_BF = zeros((Geometry.Nant)^2, length(Pars.SNR));

% Noise after BF:
noise_simple_BF = zeros(size(waveform1, 1), length(Pars.SNR)); 
noise_nulling_BF = zeros(size(waveform1, 1), length(Pars.SNR)); 
noise_mvdr_BF = zeros(size(waveform1, 1), length(Pars.SNR)); 
noise_lms_BF = zeros(size(waveform1, 1), length(Pars.SNR)); 
noise_mmse_BF = zeros(size(waveform1, 1), length(Pars.SNR)); 

%% Looping for finding weigths for different levels of SNR:

for i = 1 : length(Pars.SNR)
    
    %% Adding noise wrt the current SNR 
    
    chOut_noise(:, :, i) = awgn(chOut, Pars.SNR(i), 'measured');
    chOut1_noise(:, :, i) = awgn(chOut1, Pars.SNR(i), 'measured');
    chOut2_noise(:, :, i) = awgn(chOut2, Pars.SNR(i), 'measured');
    noise(:, :, i) = chOut_noise(:, :, i) - chOut(:, :);
    
    %% Estimation od the DoA:
    
    [~, DoA] = estimator(squeeze(chOut_noise(:, :, i)));
    
    %% Applying BF techniques
    
    % Simple BF:
    [chOut_simple_BF(:, i), w_simple_BF(:, i)] = Conventional_BF(Geometry, Pars, DoA(:, 1), squeeze(chOut_noise(:, :, i)));
    
    % Nullsteering BF:
    [chOut_nulling_BF(:, i), w_nulling_BF(:, i)] = Nullsteering_BF(Geometry, Pars, DoA, squeeze(chOut_noise(:, :, i)));
    
    % MVDR BF:
    [chOut_mvdr_BF(:, i), w_mvdr_BF(:, i)] = MVDR_BF(Geometry, Pars, DoA(:, 1), squeeze(chOut_noise(:, :, i)));
    
    % LMS BF:
    nTrain = round(length(chOut(:,1)) / 2);
    [chOut_lms_BF(:, i), w_lms_BF(:, i)] = LMS_BF(Geometry, Pars, DoA(:, 1), squeeze(chOut_noise(:, :, i)), waveform1(1:nTrain, :));
    
    % MMSE BF:
    nTrain = round(length(chOut(:,1)) / 2);
    [chOut_mmse_BF(:, i), w_mmse_BF(:, i)] = MMSE_BF(Geometry, Pars, squeeze(chOut_noise(:, :, i)), waveform1(1:nTrain, :));
    
    %% Passing the signals through the different BFs
    
    % Simple BF:
%     chOut1_simple_BF(:, i) = (chOut1_noise(:, :, i)' * (w_simple_BF(:, i)))';
%     chOut2_simple_BF(:, i) = (chOut2_noise(:, :, i)' * (w_simple_BF(:, i)))';
    chOut1_simple_BF(:, i) = (transpose((w_simple_BF(:, i))) * chOut1_noise(:, :, i)')';
    chOut2_simple_BF(:, i) = (transpose((w_simple_BF(:, i))) * chOut2_noise(:, :, i)')';
    noise_simple_BF(:, i) = (transpose((w_simple_BF(:, i))) *  noise(:, :, i)')';
    
    % Nullsteering BF:
%     chOut1_nulling_BF(:, i) = transpose(chOut1_noise(:, :, i)' * (w_nulling_BF(:, i)));
%     chOut2_nulling_BF(:, i) = transpose(chOut2_noise(:, :, i)' * (w_nulling_BF(:, i)));
    chOut1_nulling_BF(:, i) = transpose((w_nulling_BF(:, i))) * chOut1_noise(:, :, i)';
    chOut2_nulling_BF(:, i) = transpose((w_nulling_BF(:, i))) * chOut2_noise(:, :, i)';
    noise_nulling_BF(:, i) = transpose((w_nulling_BF(:, i))) * noise(:, :, i)';
    
    % MVDR BF:
%     chOut1_mvdr_BF(:, i) = transpose(chOut1_noise(:, :, i)' * (w_mvdr_BF(:, i)));
%     chOut2_mvdr_BF(:, i) = transpose(chOut2_noise(:, :, i)' * (w_mvdr_BF(:, i)));
    chOut1_mvdr_BF(:, i) = transpose((w_mvdr_BF(:, i))) * chOut1_noise(:, :, i)';
    chOut2_mvdr_BF(:, i) = transpose((w_mvdr_BF(:, i))) * chOut2_noise(:, :, i)';
    noise_mvdr_BF(:, i) = transpose((w_mvdr_BF(:, i))) * noise(:, :, i)';
  
    % LMS BF:
%     chOut1_lms_BF(:, i) = transpose(chOut1_noise(:, :, i)' * (w_lms_BF(:, i)));
%     chOut2_lms_BF(:, i) = transpose(chOut2_noise(:, :, i)' * (w_lms_BF(:, i)));
    chOut1_lms_BF(:, i) = transpose((w_lms_BF(:, i))) * chOut1_noise(:, :, i)';
    chOut2_lms_BF(:, i) = transpose((w_lms_BF(:, i))) * chOut2_noise(:, :, i)';
    noise_lms_BF(:, i) = transpose((w_lms_BF(:, i))) * noise(:, :, i)';
    
    % MMSE BF:
%     chOut1_mmse_BF(:, i) = transpose(chOut1_noise(:, :, i)' * (w_mmse_BF(:, i)));
%     chOut2_mmse_BF(:, i) = transpose(chOut2_noise(:, :, i)' * (w_mmse_BF(:, i)));
    chOut1_mmse_BF(:, i) = transpose((w_mmse_BF(:, i))) * chOut1_noise(:, :, i)';
    chOut2_mmse_BF(:, i) = transpose((w_mmse_BF(:, i))) * chOut2_noise(:, :, i)';
    noise_mmse_BF(:, i) = transpose((w_mmse_BF(:, i))) * noise(:, :, i)';
    
    %% Computation of the power of good (V1) and interfering (V2) singals after the BF and of SNR:
    
    % Simple BF:
    P1_simple_BF = sum(abs(fft(chOut1_simple_BF(:, i))).^2);
    P2_simple_BF = sum(abs(fft(chOut2_simple_BF(:, i))).^2);
    Ptot_simple_BF = sum(abs(fft(chOut_simple_BF(:, i))).^2);
    P_noise_simple_BF = sum(abs(fft(noise_simple_BF(:, i))).^2);
    %SNR_out_simple_BF(i) = P1_simple_BF / (P2_simple_BF + P_noise_simple_BF);
    SNR_out_simple_BF(i) = Ptot_simple_BF / (P_noise_simple_BF);
    %SNR_out_simple_BF(i) = P1_simple_BF / (P_noise_simple_BF);
    SNR_out_simple_BF(i) = 10 * log10(SNR_out_simple_BF(i));
    
    % Null-steering BF:
    P1_nulling_BF = sum(abs(fft(chOut1_nulling_BF(:, i))).^2);
    P2_nulling_BF = sum(abs(fft(chOut2_nulling_BF(:, i))).^2);
    Ptot_nulling_BF = sum(abs(fft(chOut_nulling_BF(:, i))).^2);
    P_noise_nulling_BF = sum(abs(fft(noise_nulling_BF(:, i))).^2);
    %SNR_out_nulling_BF(i) = P1_nulling_BF / (P2_nulling_BF + P_noise_nulling_BF);
    SNR_out_nulling_BF(i) = Ptot_nulling_BF / (P_noise_nulling_BF);
    %SNR_out_nulling_BF(i) = P1_nulling_BF / (P_noise_nulling_BF);
    SNR_out_nulling_BF(i) = 10 * log10(SNR_out_nulling_BF(i));
    
    % MVDR BF:
    P1_mvdr_BF = sum(abs(fft(chOut1_mvdr_BF(:, i))).^2);
    P2_mvdr_BF = sum(abs(fft(chOut2_mvdr_BF(:, i))).^2);
    Ptot_mvdr_BF = sum(abs(fft(chOut_mvdr_BF(:, i))).^2);
    P_noise_mvdr_BF = sum(abs(fft(noise_mvdr_BF(:, i))).^2);
    %SNR_out_mvdr_BF(i) = P1_mvdr_BF / (P2_mvdr_BF + P_noise_mvdr_BF);
    SNR_out_mvdr_BF(i) = Ptot_mvdr_BF / (P_noise_mvdr_BF);
    %SNR_out_mvdr_BF(i) = P1_mvdr_BF / (P_noise_mvdr_BF);
    SNR_out_mvdr_BF(i) = 10 * log10(SNR_out_mvdr_BF(i));
    
    % LMS BF:
    P1_lms_BF = sum(abs(fft(chOut1_lms_BF(:, i))).^2);
    P2_lms_BF = sum(abs(fft(chOut2_lms_BF(:, i))).^2);
    Ptot_lms_BF = sum(abs(fft(chOut_lms_BF(:, i))).^2);
    P_noise_lms_BF = sum(abs(fft(noise_lms_BF(:, i))).^2);
    %SNR_out_lms_BF(i) = P1_lms_BF / (P2_lms_BF + P_noise_lms_BF);
    SNR_out_lms_BF(i) = Ptot_lms_BF / (P_noise_lms_BF);
    %SNR_out_lms_BF(i) = P1_lms_BF / (P_noise_lms_BF);
    SNR_out_lms_BF(i) = 10 * log10(SNR_out_lms_BF(i));
    
    % MMSE BF:
    P1_mmse_BF = sum(abs(fft(chOut1_mmse_BF(:, i))).^2);
    P2_mmse_BF = sum(abs(fft(chOut2_mmse_BF(:, i))).^2);
    Ptot_mmse_BF = sum(abs(fft(chOut_mmse_BF(:, i))).^2);
    P_noise_mmse_BF = sum(abs(fft(noise_mmse_BF(:, i))).^2);
    %SNR_out_mmse_BF(i) = P1_mmse_BF / (P2_mmse_BF + P_noise_mmse_BF);
    SNR_out_mmse_BF(i) = Ptot_mmse_BF / (P_noise_mmse_BF);
    %SNR_out_mmse_BF(i) = P1_mmse_BF / (P_noise_mmse_BF);
    SNR_out_mmse_BF(i) = 10 * log10(SNR_out_mmse_BF(i));
    
end
