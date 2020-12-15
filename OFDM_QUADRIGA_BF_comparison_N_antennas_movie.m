%% DESCRIPTION
% In this script, we're showing the perfomace of different kinds of beamforming techniques
% varying the number of antennas at the receiver (BS) and keeping fixed the
% number of interferents (2). 
% The beamformers we're going to consider are:
% - Simple BF
% - Nullsteering BF
% - MVDR BF
% - LMS BF
% - MMSE BF

%% PREPARING

clear all;
close all;
clc;

%% DEFINING GEOMETRY AND PARS

% Carrier frequency and wavelength
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

% Vector of sizes of antenna arrays:
Geometry.Nant = [2 4 8 16];

% Definig the different antenna arrays with antenna spacing = lambda/2:
Geometry.BSarray_2x2 = phased.URA('Size', [Geometry.Nant(1) Geometry.Nant(1)], ...
    'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'z');

Geometry.BSarray_4x4 = phased.URA('Size', [Geometry.Nant(2) Geometry.Nant(2)], ...
    'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'z');

Geometry.BSarray_8x8 = phased.URA('Size', [Geometry.Nant(3) Geometry.Nant(3)], ...
    'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'z');

Geometry.BSarray_16x16 = phased.URA('Size', [Geometry.Nant(4) Geometry.Nant(4)], ...
    'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'z');

% Putting all antenna arrays into a vector:
Geometry.BSarray_all = {Geometry.BSarray_2x2 Geometry.BSarray_4x4 Geometry.BSarray_8x8 Geometry.BSarray_16x16};

% Geometry.Nant = [4 16];
% 
% Geometry.BSarray_4x4 = phased.URA('Size', [Geometry.Nant(1) Geometry.Nant(1)], ...
%     'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'z');
% 
% Geometry.BSarray_16x16 = phased.URA('Size', [Geometry.Nant(2) Geometry.Nant(2)], ...
%     'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'z');
% 
% % Putting all antenna arrays into a vector:
% Geometry.BSarray_all = {Geometry.BSarray_4x4 Geometry.BSarray_16x16};

%% %% 3GPP Parameters
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

% Used by the channel to determine the delay in number of samples:
Fs2 = 180000;

% OFDM demodulators definition:
ofdmDemod1 = comm.OFDMDemodulator(ofdmMod1);
ofdmDemod2 = comm.OFDMDemodulator(ofdmMod2);

% Visualizing OFDM mapping:
% showResourceMapping(ofdmMod1);
% title('OFDM modulators (1 = 2)');


%% DEFINITION OF OUTPUTS TO BE COMPARED

% Detected direction of arrivals of the first vehicle (only one that can be correctly detected in this case):
DoAs = zeros(2, length(Geometry.Nant));

% BER of the beamformers (only after beamformer and with equalization case, one column per BF type):
ber = ones(length(Geometry.Nant), 5);

% Matrices of the weigths of the antennas: one colums is a set of weigths, one matrix per type of BF:
w_simple_BF = zeros(max(Geometry.Nant)^2, length(Geometry.Nant));
w_nulling_BF = zeros(max(Geometry.Nant)^2, length(Geometry.Nant));
w_mvdr_BF = zeros(max(Geometry.Nant)^2, length(Geometry.Nant));
w_lms_BF = zeros(max(Geometry.Nant)^2, length(Geometry.Nant));
w_mmse_BF = zeros(max(Geometry.Nant)^2, length(Geometry.Nant));

% Signals after each typer of beamformer:
chOut_simple_BF = zeros(size(waveform1, 1), length(Geometry.Nant)); 
chOut_nulling_BF = zeros(size(waveform1, 1), length(Geometry.Nant));
chOut_mvdr_BF = zeros(size(waveform1, 1), length(Geometry.Nant));
chOut_lms_BF = zeros(size(waveform1, 1), length(Geometry.Nant));
chOut_mmse_BF = zeros(size(waveform1, 1), length(Geometry.Nant));

% Signals after beamformer and equalization:
chOut_simple_BF_equal = zeros(size(waveform1, 1), length(Geometry.Nant)); 
chOut_nulling_BF_equal = zeros(size(waveform1, 1), length(Geometry.Nant));
chOut_mvdr_BF_equal = zeros(size(waveform1, 1), length(Geometry.Nant));
chOut_lms_BF_equal = zeros(size(waveform1, 1), length(Geometry.Nant));
chOut_mmse_BF_equal = zeros(size(waveform1, 1), length(Geometry.Nant));

% Signals OFDM-demodulated after BF and equalization:
% Matrices dimension:(dataDubcarriers * nSymbols * antennavector):
dimensions = [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))), ...
    nSymbols1, length(Geometry.Nant)];

chOut_simple_BF_equal_OFDMdem = zeros(dimensions); 
chOut_nulling_BF_equal_OFDMdem = zeros(dimensions);
chOut_mvdr_BF_equal_OFDMdem = zeros(dimensions);
chOut_lms_BF_equal_OFDMdem = zeros(dimensions);
chOut_mmse_BF_equal_OFDMdem = zeros(dimensions);

% Signals OFDM-demodulated after BF but without equalization:
chOut_simple_BF_OFDMdem = zeros(dimensions); 
chOut_nulling_BF_OFDMdem = zeros(dimensions);
chOut_mvdr_BF_OFDMdem = zeros(dimensions);
chOut_lms_BF_OFDMdem = zeros(dimensions);
chOut_mmse_BF_OFDMdem = zeros(dimensions);

% Signals OFDM-demodulated without BF and without equalization:
chOut_OFDMdem = zeros(dimensions);

%% LOOP FOR PERFORMANCE COMPARISON

for ant = 1 : length(Geometry.Nant)
   
    %% DEFINING CURRENT ANTENNA ARRAY
    
    % Current BS antenna array:
    Geometry.BSarray = Geometry.BSarray_all{ant};
    
    % Getting position antenna array:
    Geometry.BSAntennaPos = getElementPosition(Geometry.BSarray);
    
    % Creating conformal antenna array:
    Geometry.confarray = phased.ConformalArray('ElementPosition', Geometry.BSAntennaPos);
    
    %% 3GPP CHANNEL
    
    % Velocities of veichles:
    vel1 = [0;0;0];
    vel2 = [0;0;0];
    
    % Create Layout Object
    l = qd_layout;
    l.set_scenario('QuaDRiGa_UD2D_LOS');
    
    % Define tx and rx
    txArr = qd_arrayant('omni');
    rxArr = qd_arrayant('omni');
    rxArr.no_elements = Geometry.Nant(ant) * Geometry.Nant(ant);
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
                + chOut1(antenna, :);
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
                + chOut2(antenna, :);
        end
    end
    
    
    % Summing contributes of both the signals:
    chOut = chOut1 + chOut2;
    chOut = chOut.';
    
    % Adding AWGN noise to waveform:
    Pars.SNR = 20; % in dB
    chOut_noise = awgn(chOut, Pars.SNR, 'measured');
    noise = chOut_noise - chOut;
    chOut = chOut_noise;
    
    %% ESTIMATION OF DoA (MUSIC ALGORITHM)
    
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
    [~, DoAs_tmp] = estimator(chOut);
    
    % DoA of the first vehicle:
    DoAs(:, ant) = DoAs_tmp(:, 1);
    
    % THIS IS FOR THIS SPECIFIC CASE -> NEED TO FIX FOR GENERAL CASE:
    % DoAs(1,:) = -(DoAs(1,:) - 180);
    % temp1 = DoAs(:,1);
    % DoAs(:,1) = DoAs(:,2);
    % DoAs(:,2) = temp1;
    
    % Plotting:
    % figure();
    % plotSpectrum(estimator);
    
    %% APPLICATION OF DIFFERENT BEAMFORMERS
    
    % Simple BF:
    [chOut_simple_BF(:, ant), w_simple_BF(1:Geometry.Nant(ant)^2, ant)] = ...
        Conventional_BF(Geometry, Pars, DoAs_tmp(:, 1), chOut);
    
    % Nullsteering BF:
    [chOut_nulling_BF(:, ant), w_nulling_BF(1:Geometry.Nant(ant)^2, ant)] = ...
        Nullsteering_BF(Geometry, Pars, DoAs_tmp, chOut);
    
    % MVDR BF:
    [chOut_mvdr_BF(:, ant), w_mvdr_BF(1:Geometry.Nant(ant)^2, ant)] = ...
        MVDR_BF(Geometry, Pars, DoAs_tmp(:, 1), chOut);
    
    % LMS BF:
    nTrain = round(length(chOut(:,1)) / 2);
    [chOut_lms_BF(:, ant), w_lms_BF(1:Geometry.Nant(ant)^2, ant)] = ...
        LMS_BF(Geometry, Pars, DoAs_tmp(:, 1), chOut, waveform1(1:nTrain, :));
    
    % MMSE BF:
    nTrain = round(length(chOut(:,1)) / 2);
    [chOut_mmse_BF(:, ant), w_mmse_BF(1:Geometry.Nant(ant)^2, ant)] = ...
        MMSE_BF(Geometry, Pars, chOut, waveform1(1:nTrain, :));

    %%  Channel equalization
    
    % OFDM symbol used to train
    n_training = size(chOut, 1);
    
    % Max number of iteration of the equalization algorithm:
    max_iter = 100;
    
    % Taps of the equalization filter:
    g_len = 1;
    
    % Gradient descent algorithm for estiamting the inverse filter of the channel:
    g_simple_BF = Gradient_descent(chOut_simple_BF(:, ant).',  waveform1(1:n_training).', n_training, max_iter, g_len);
    g_nulling_BF = Gradient_descent(chOut_nulling_BF(:, ant).',  waveform1(1:n_training).', n_training, max_iter, g_len);
    g_mvdr_BF = Gradient_descent(chOut_mvdr_BF(:, ant).',  waveform1(1:n_training).', n_training, max_iter, g_len);
    g_lms_BF = Gradient_descent(chOut_lms_BF(:, ant).',  waveform1(1:n_training).', n_training, max_iter, g_len);
    g_mmse_BF = Gradient_descent(chOut_mmse_BF(:, ant).',  waveform1(1:n_training).', n_training, max_iter, g_len);
    
    % Channel equalization:
    chOut_simple_BF_equal_ = conv(g_simple_BF, chOut_simple_BF(:, ant));
    chOut_simple_BF_equal(:, ant) = chOut_simple_BF_equal_(1:size(chOut_simple_BF, 1));
    
    chOut_nulling_BF_equal_ = conv(g_nulling_BF, chOut_nulling_BF(:, ant));
    chOut_nulling_BF_equal(:, ant) = chOut_nulling_BF_equal_(1:size(chOut_nulling_BF_equal, 1));
    
    chOut_mvdr_BF_equal_ = conv(g_mvdr_BF, chOut_mvdr_BF(:, ant));
    chOut_mvdr_BF_equal(:, ant) = chOut_mvdr_BF_equal_(1:size(chOut_mvdr_BF_equal, 1));
    
    chOut_lms_BF_equal_ = conv(g_lms_BF, chOut_lms_BF(:, ant));
    chOut_lms_BF_equal(:, ant) = chOut_lms_BF_equal_(1:size(chOut_lms_BF_equal, 1));
    
    chOut_mmse_BF_equal_ = conv(g_mmse_BF, chOut_mmse_BF(:, ant));
    chOut_mmse_BF_equal(:, ant) = chOut_mmse_BF_equal_(1:size(chOut_mmse_BF_equal), 1);
    
    %% OFDM DEMODULATION
    % In order to show the effects of the various typer of beamformer, we
    % consider the signals at the first antenna ODFM-demodulated at different steps:
    % 1) Without the BF and the equalization
    % 2) With the BF but without equalization
    % 3) With the BF and with the equalization
    
    % BF: no, equalization: no:
    chOut_OFDMdem(:, :, ant) = ofdmDemod1(chOut(:,1));
    
    % BF: yes, equalization: no:
    chOut_simple_BF_OFDMdem(:, :, ant) = ofdmDemod1(chOut_simple_BF(:, ant));
    chOut_nulling_BF_OFDMdem(:, :, ant) = ofdmDemod1(chOut_nulling_BF(:, ant));
    chOut_mvdr_BF_OFDMdem(:, :, ant) = ofdmDemod1(chOut_mvdr_BF(:, ant));
    chOut_lms_BF_OFDMdem(:, :, ant) = ofdmDemod1(chOut_lms_BF(:, ant));
    chOut_mmse_BF_OFDMdem(:, :, ant) = ofdmDemod1(chOut_mmse_BF(:, ant));
    
    % BF: no, equalization: yes:
    chOut_simple_BF_equal_OFDMdem(:, :, ant) = ofdmDemod1(chOut_simple_BF_equal(:, ant));
    chOut_nulling_BF_equal_OFDMdem(:, :, ant) = ofdmDemod1(chOut_nulling_BF_equal(:, ant));
    chOut_mvdr_BF_equal_OFDMdem(:, :, ant) = ofdmDemod1(chOut_mvdr_BF_equal(:, ant));
    chOut_lms_BF_equal_OFDMdem(:, :, ant) = ofdmDemod1(chOut_lms_BF_equal(:, ant));
    chOut_mmse_BF_equal_OFDMdem(:, :, ant) = ofdmDemod1(chOut_mmse_BF_equal(:, ant));
    
    %% QAM DEMODULATION AND BER COMPUTATION
    % In order to check if the beamformer and the equalization properly
    % work, we consider the BER after the QAM demodulation of the signal
    % that has been beamformed and equalized
    
    % QAM demodulation:
    chOut_simple_BF_equal_OFDMdem_QAMdem = ...
        qamdemod(chOut_simple_BF_equal_OFDMdem(:, :, ant), M1, 'OutputType', 'bit', 'UnitAveragePower', true);
    chOut_simple_BF_equal_OFDMdem_QAMdem = chOut_simple_BF_equal_OFDMdem_QAMdem(:);
    
    chOut_nulling_BF_equal_OFDMdem_QAMdem = ...
        qamdemod(chOut_nulling_BF_equal_OFDMdem(:, :, ant), M1, 'OutputType', 'bit', 'UnitAveragePower', true);
    chOut_nulling_BF_equal_OFDMdem_QAMdem = chOut_nulling_BF_equal_OFDMdem_QAMdem(:);
    
    chOut_mvdr_BF_equal_OFDMdem_QAMdem = ...
        qamdemod(chOut_mvdr_BF_equal_OFDMdem(:, :, ant), M1, 'OutputType', 'bit', 'UnitAveragePower', true);
    chOut_mvdr_BF_equal_OFDMdem_QAMdem =  chOut_mvdr_BF_equal_OFDMdem_QAMdem(:);
    
    chOut_lms_BF_equal_OFDMdem_QAMdem = ...
        qamdemod(chOut_lms_BF_equal_OFDMdem(:, :, ant), M1, 'OutputType', 'bit', 'UnitAveragePower', true);
    chOut_lms_BF_equal_OFDMdem_QAMdem = chOut_lms_BF_equal_OFDMdem_QAMdem(:);
    
    chOut_mmse_BF_equal_OFDMdem_QAMdem = ...
        qamdemod(chOut_mmse_BF_equal_OFDMdem(:, :, ant), M1, 'OutputType', 'bit', 'UnitAveragePower', true);
    chOut_mmse_BF_equal_OFDMdem_QAMdem = chOut_mmse_BF_equal_OFDMdem_QAMdem(:);
    
    % BER computation:
    [~, ber(ant, 1)] = ...
        biterr(bitInput1((length(bitInput1) - size(chOut_simple_BF_equal_OFDMdem_QAMdem, 1) + 1):end), ...
            chOut_simple_BF_equal_OFDMdem_QAMdem(1:end)); 
    
    [~, ber(ant, 2)] = ...
        biterr(bitInput1((length(bitInput1) - size(chOut_nulling_BF_equal_OFDMdem_QAMdem, 1) + 1):end), ...
            chOut_nulling_BF_equal_OFDMdem_QAMdem(1:end));
    
    [~, ber(ant, 3)] = ...
        biterr(bitInput1((length(bitInput1) - size(chOut_mvdr_BF_equal_OFDMdem_QAMdem, 1) + 1):end), ...
            chOut_mvdr_BF_equal_OFDMdem_QAMdem(1:end));
    
    [~, ber(ant, 4)] = ...
        biterr(bitInput1((length(bitInput1) - size(chOut_lms_BF_equal_OFDMdem_QAMdem, 1) + 1):end), ...
            chOut_lms_BF_equal_OFDMdem_QAMdem(1:end));
    
    [~, ber(ant, 5)] = ...
        biterr(bitInput1((length(bitInput1) - size(chOut_mmse_BF_equal_OFDMdem_QAMdem, 1) + 1):end), ...
            chOut_mmse_BF_equal_OFDMdem_QAMdem(1:end));
    
end

%% PLOTS FOR COMPARING THE WEIGTHS OF THE ANTENNAS USING ANTENNA ARRAYS OF DIFFERENT SIZES

% Preparing cells with the weigth of the antennas: one cell for antenna array containing the
% weights of the different BF in the order: simple, null-steering, mvdr, lms, mmse.
w_2x2 = {w_lms_BF(:, 1) w_nulling_BF(:, 1) w_mvdr_BF(:, 1) ...
    w_lms_BF(:, 1) w_mmse_BF(:, 1)};

w_4x4 = {w_lms_BF(:, 2) w_nulling_BF(:, 2) w_mvdr_BF(:, 2) ...
    w_lms_BF(:, 2) w_mmse_BF(:, 2)};

w_8x8 = {w_lms_BF(:, 3) w_nulling_BF(:, 3) w_mvdr_BF(:, 3) ...
    w_lms_BF(:, 3) w_mmse_BF(:, 3)};

w_16x16 = {w_lms_BF(:, 4) w_nulling_BF(:, 4) w_mvdr_BF(:, 4) ...
    w_lms_BF(:, 4) w_mmse_BF(:, 4)};

% Here, we create a video in which we show the weigths of the different
% kinds of BF techiques using differentantenna array configurations

% Createing a videowriter object for a new video file:
v = VideoWriter('Antenna_patterns.avi');
v.FrameRate = 1;
open(v);

% Creating video:
for bf = 1 : 5
    
    % Creating video:
    frame = figure();
    setappdata(gcf, 'SubplotDefaultAxesLocation', [0, 0, 1, 1]);
    set(gcf, 'WindowState', 'Maximized');
    pause(1);

    Rect = [0.02, 0.02, 0.95, 0.9];
    AxisPos = myPlotPos(2, 2, Rect);
    
    % plot1 = subplot(2,2,1) -> BF with 2x2 antenna array.
    W_2x2 = w_2x2{bf};
    axes('Position', AxisPos(1, :));
    pattern(Geometry.BSarray_all{1}, Pars.fc, [-180:180], 0, ...
        'PropagationSpeed', Pars.c, ...
        'Type', 'powerdb', ...
        'CoordinateSystem', 'rectangular', ...
        'Weights', W_2x2(1:2^2));
    %title('Weigths of the 2x2 antenna array', 'FontSize', 18);
    %fig_title = sprintf('Contellation revealed by 2x2 antenna array - %s', bf_name);
    %title(fig_title, 'FontSize', 18);
    
    % plot2 = subplot(2,2,2) -> BF with 4x4 antenna array.
    W_4x4 = w_4x4{bf};
    axes('Position', AxisPos(2, :));
    pattern(Geometry.BSarray_all{2}, Pars.fc, [-180:180], 0, ...
        'PropagationSpeed', Pars.c, ...
        'Type', 'powerdb', ...
        'CoordinateSystem', 'rectangular', ...
        'Weights', W_4x4(1:4^2));
    %title('Weigths of the antenna array', 'FontSize', 18);
    %fig_title = sprintf('Contellation revealed by 4x4 antenna array - %s', bf_name);
    %title(fig_title, 'FontSize', 18);
    
    % plot3 = subplot(2,2,3) -> BF with 8x8 antenna array.
    W_8x8 = w_8x8{bf};
    axes('Position', AxisPos(3, :));
    pattern(Geometry.BSarray_all{3}, Pars.fc, [-180:180], 0, ...
        'PropagationSpeed', Pars.c, ...
        'Type', 'powerdb', ...
        'CoordinateSystem', 'rectangular', ...
        'Weights', W_8x8(1:8^2));
    %title('Weigths of the 8x8 antenna array', 'FontSize', 18);
    %fig_title = sprintf('Contellation revealed by 18x8 antenna array - %s', bf_name);
    %title(fig_title, 'FontSize', 18);  
    
    % plot4 = subplot(2,2,4) -> BF with 16x16 antenna array.
    W_16x16 = w_4x4{bf};
    axes('Position', AxisPos(4, :));
    pattern(Geometry.BSarray_all{4}, Pars.fc, [-180:180], 0, ...
        'PropagationSpeed', Pars.c, ...
        'Type', 'powerdb', ...
        'CoordinateSystem', 'rectangular', ...
        'Weights', W_16x16(1:16^2));
    %title('Weigths of the 16x16 antenna array', 'FontSize', 18);
    %fig_title = sprintf('Contellation revealed by 16x16 antenna array - %s', bf_name);
    %title(fig_title, 'FontSize', 18);
    
    F = getframe(frame);
    writeVideo(v, F);
    writeVideo(v, F);
    pause(5)
    close all;
    
end
close(v);


% Opening figure for plotting the weigths as images:
figure();

% Plot of the weigths of BFs for 2x2 antenna array
subplot(2, 2, 1);
for i = 1 : 5
    
    hold on;
    w = w_2x2{i};
    pattern(Geometry.BSarray_all{1}, Pars.fc, [-180:180], 0, ...
        'PropagationSpeed', Pars.c, ...
        'Type', 'powerdb', ...
        'CoordinateSystem', 'rectangular', ...
        'Weights', w(1:2^2));
  
end
grid on;
title('Weights for different BF techniques using a 2x2 antenna array', 'FontSize', 18);
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');

% Plot of the weigths of BFs for 4x4 antenna array
subplot(2, 2, 2);
for i = 1 : 5
    
    hold on;
    w = w_4x4{i};
    pattern(Geometry.BSarray_all{2}, Pars.fc,[-180:180], 0, ...
        'PropagationSpeed', Pars.c, ...
        'Type', 'powerdb', ...
        'CoordinateSystem', 'rectangular', ...
        'Weights', w(1:4^2));
  
end
grid on;
title('Weights for different BF techniques using a 4x4 antenna array', 'FontSize', 18);
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');

% Plot of the weigths of BFs for 8x8 antenna array
subplot(2, 2, 3);
for i = 1 : 5
    
    hold on;
    w = w_8x8{i};
    pattern(Geometry.BSarray_all{3}, Pars.fc,[-180:180], 0, ...
        'PropagationSpeed', Pars.c, ...
        'Type', 'powerdb', ...
        'CoordinateSystem', 'rectangular', ...
        'Weights', w(1:8^2));
  
end
grid on;
title('Weights for different BF techniques using a 8x8 antenna array', 'FontSize', 18);
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');

% Plot of the weigths of BFs for 16x16 antenna array
subplot(2, 2, 4);
for i = 1 : 5
    
    hold on;
    w = w_16x16{i};
    pattern(Geometry.BSarray_all{4}, Pars.fc,[-180:180], 0, ...
        'PropagationSpeed', Pars.c, ...
        'Type', 'powerdb', ...
        'CoordinateSystem', 'rectangular', ...
        'Weights', w(1:16^2));
  
end
grid on;
title('Weights for different BF techniques using a 16x16 antenna array', 'FontSize', 18);
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');


%% PLOTS FOR COMPARING THE CONSTALLATIONS REVEALED BY DIFFERENT BEAMFORMERS WITH DIFFERENT NUMBER OF ANTENNAS
% Here we prepare a movie showing the constellations revealed by the
% different antenna arrays and different beamformers.

% First, we prepare 4 cell (one for each antenna array used) each one containing
% the costellation of the BFs used in the order: 
% simple, null-steering, mvdr, lms, mmse
BF_2x2 = {chOut_simple_BF_equal_OFDMdem(:, :, 1) chOut_nulling_BF_equal_OFDMdem(:, :, 1)...
    chOut_mvdr_BF_equal_OFDMdem(:, :, 1) chOut_lms_BF_equal_OFDMdem(:, :, 1)...
    chOut_mmse_BF_equal_OFDMdem(:, :, 1)};

BF_4x4 = {chOut_simple_BF_equal_OFDMdem(:, :, 2) chOut_nulling_BF_equal_OFDMdem(:, :, 2)...
    chOut_mvdr_BF_equal_OFDMdem(:, :, 2) chOut_lms_BF_equal_OFDMdem(:, :, 2)...
    chOut_mmse_BF_equal_OFDMdem(:, :, 2)};

BF_8x8 = {chOut_simple_BF_equal_OFDMdem(:, :, 3) chOut_nulling_BF_equal_OFDMdem(:, :, 3)...
    chOut_mvdr_BF_equal_OFDMdem(:, :, 3) chOut_lms_BF_equal_OFDMdem(:, :, 3)...
    chOut_mmse_BF_equal_OFDMdem(:, :, 3)};

BF_16x16 = {chOut_simple_BF_equal_OFDMdem(:, :, 4) chOut_nulling_BF_equal_OFDMdem(:, :, 4)...
    chOut_mvdr_BF_equal_OFDMdem(:, :, 4) chOut_lms_BF_equal_OFDMdem(:, :, 4)...
    chOut_mmse_BF_equal_OFDMdem(:, :, 4)};

% Createing a videowriter object for a new video file:
v = VideoWriter('Contellations.avi');
v.FrameRate = 1;
open(v);

% Loop for writing video:
for bf = 1 : 5
    
    % The current BF:
    switch bf
        case bf == 1
            bf_name = 'SIMPLE BF';
        case bf == 2
            bf_name = 'NULL-STEERING BF';
        case bf == 3
            bf_name = 'MVDR BF';
        case bf == 4
            bf_name = 'LMS BF';
        case bf == 5
            bf_name = 'MMSE BF';
    end
    
    % Creating video:
    frame = figure();
    setappdata(gcf, 'SubplotDefaultAxesLocation', [0, 0, 1, 1]);
    set(gcf, 'WindowState', 'Maximized');
    pause(1);

    Rect = [0.02, 0.02, 0.95, 0.9];
    AxisPos = myPlotPos(2, 2, Rect);
    
    % plot1 = subplot(2,2,1) -> BF with 2x2 antenna array.
    bf_2x2 = BF_2x2{bf};
    axes('Position', AxisPos(1, :));
    x = real(bf_2x2);
    x = reshape(x, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    y = imag(bf_2x2);
    y = reshape(y, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    scatter(x, y);
    title('Contellation revealed by 2x2 antenna array', 'FontSize', 18);
    %fig_title = sprintf('Contellation revealed by 2x2 antenna array - %s', bf_name);
    %title(fig_title, 'FontSize', 18);
    
    % plot2 = subplot(2,2,2) -> BF with 4x4 antenna array.
    bf_4x4 = BF_4x4{bf};
    axes('Position', AxisPos(2, :));
    x = real(bf_4x4);
    x = reshape(x, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    y = imag(bf_4x4);
    y = reshape(y, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    scatter(x, y);
    title('Contellation revealed by 4x4 antenna array', 'FontSize', 18);
    %fig_title = sprintf('Contellation revealed by 4x4 antenna array - %s', bf_name);
    %title(fig_title, 'FontSize', 18);
    
    % plot3 = subplot(2,2,3) -> BF with 8x8 antenna array.
    bf_8x8 = BF_8x8{bf};
    axes('Position', AxisPos(3, :));
    x = real(bf_8x8);
    x = reshape(x, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    y = imag(bf_8x8);
    y = reshape(y, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    scatter(x, y);
    title('Contellation revealed by 8x8 antenna array', 'FontSize', 18);
    %fig_title = sprintf('Contellation revealed by 18x8 antenna array - %s', bf_name);
    %title(fig_title, 'FontSize', 18);  
    
    % plot4 = subplot(2,2,4) -> BF with 16x16 antenna array.
    bf_16x16 = BF_4x4{bf};
    axes('Position', AxisPos(4, :));
    x = real(bf_16x16);
    x = reshape(x, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
    y = imag(bf_16x16);
    y = reshape(y, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
    scatter(x, y);
    title('Contellation revealed by 16x16 antenna array', 'FontSize', 18);
    %fig_title = sprintf('Contellation revealed by 16x16 antenna array - %s', bf_name);
    %title(fig_title, 'FontSize', 18);
    
    F = getframe(frame);
    writeVideo(v, F);
    writeVideo(v, F);
    pause(5)
    close all;
    
end
close(v);


% Here we plot 4 figures, one for each tested dimension of the antenna
% array (2x2, 4x4, 8x8, 16x16).
% Each figure contains the constellation revealed after the BF and
% equalization of the 5 BF used.
figure();

% Plot of the constellation revealed by the 2x2 array
subplot(2, 2, 1);
for i = 1 : 5
    
    hold on;
    x = real(BF_2x2{i});
    x = reshape(x, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    y = imag(BF_2x2{i});
    y = reshape(y, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    scatter(x, y);
    
end
grid on;
title('Contellation revealed by 2x2 antenna array', 'FontSize', 18);
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');

% Plot of the constellation revealed by the 4x4 array
subplot(2, 2, 2);
for i = 1 : 5
    
    hold on;
    x = real(BF_4x4{i});
    x = reshape(x, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    y = imag(BF_4x4{i});
    y = reshape(y, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    scatter(x, y);
    
end
grid on;
title('Contellation revealed by 4x4 antenna array', 'FontSize', 18);
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');

% Plot of the constellation revealed by the 8x8 array
subplot(2, 2, 3);
for i = 1 : 5
    
    hold on;
    x = real(BF_8x8{i});
    x = reshape(x, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    y = imag(BF_8x8{i});
    y = reshape(y, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    scatter(x, y);
    
end
grid on;
title('Contellation revealed by 8x8 antenna array', 'FontSize', 18);
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');

% Plot of the constellation revealed by the 16x16 array
subplot(2, 2, 4);
for i = 1 : 5
    
    hold on;
    x = real(BF_16x16{i});
    x = reshape(x, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    y = imag(BF_16x16{i});
    y = reshape(y, [(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1, 1]);
    scatter(x, y);
    
end
grid on;
title('Contellation revealed by 16x16 antenna array', 'FontSize', 18);
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');



%% BER COMPARISON
% Here we show the different BERs for the different kinds of BF and
% different antenna arrays.

figure();
x = [4 16 64 256];
markers = {'--*', '--x', '--+', '--^', '--s'};

for i = 1 : 5
    
    hold on;
    y = ber(:, i);
    plot(x, y, markers{i}, 'MarkerSize', 10, 'LineWidth', 1.5);
    
end
grid on;
title('BER for different kinds of BF');
legend('Simple BF', 'Null-steering BF', 'MVDR BF', 'LMS BF', 'MMSE BF');
