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
Geometry.Nant = 16;
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
CPlen = [4];

% First OFDM modulator:
ofdmMod1 = comm.OFDMModulator('FFTLength', nfft, ...
    'NumGuardBandCarriers', NumGuardBandCarriers, ... % Default values
    'InsertDCNull', false, ...
    'CyclicPrefixLength', CPlen, ...
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
    'CyclicPrefixLength', CPlen, ...
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


chOut = chOut1 + chOut2;
% chOut = chOut2;
chOut = chOut.';

% 
% % Calucation of received wavefrom1 (attention to dimension of waveforms):
% chOut = collectPlaneWave(Geometry.BSarray, [w1 w2], ...
%         [Geometry.DOAV1Start', Geometry.DOAV2Start'], Pars.fc);
%     




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
% DoAs(1,:) = -(DoAs(1,:) - 180);
% temp1 = DoAs(:,1);
% DoAs(:,1) = DoAs(:,2);
% DoAs(:,2) = temp1;

% Plotting:
% figure();
% plotSpectrum(estimator);

%% Beamformer 0 SIMPLE, 1 NULLING, 2 MVDR, 3 LMS, 4 MMSE
Type = 0;

switch Type
    
    case 0
        [arrOut,w] = Conventional_BF(Geometry, Pars, DoAs(:,1), chOut);

    case 1     
        [arrOut,w] = Nullsteering_BF(Geometry, Pars, DoAs, chOut);
        
    case 2       
        [arrOut,w] = MVDR_BF(Geometry, Pars, DoAs(:,1), chOut);
        
    case 3
        % We use the first half of the received signal for the LMS algorithm, 
        % finding then the weights to be assigned to each antenna and applying the
        % to the received singal for better reception.

        % Training sequence length:
        nTrain = round(length(chOut(:,1)) / 2);

        [arrOut, w] = LMS_BF(Geometry, Pars, DoAs(:, 1), chOut, waveform1(1:nTrain, :));
    
    case 4   
        % Training sequence length:
        nTrain = round(length(chOut(:,1)) / 2);
        
        [arrOut,w] = MMSE_BF(Geometry, Pars, chOut, waveform1(1:nTrain, :));
        
end


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




 
%%  Channel equalization

% OFDM symbol used to train
n_training = length(arrOut);

max_iter = 100;

% Tap of the equalizer
g_len = 1;

g = Gradient_descent( arrOut.',  waveform1(1:n_training).', n_training, max_iter, g_len);

arrOut_ = conv(g,arrOut);

arrOut_equal = arrOut_(1:length(arrOut));

 

%% OFDM demodulation and QAM demodulation

% No beamformer without equalization:
out = ofdmDemod1(chOut(:,1)); % first antenna
figure;

x = real(out);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(out);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
scatter(x,y);
title('no BF, no equal')

dataOut_beam_noequal = qamdemod(out,M1,'OutputType','bit');
dataOut_beam_noequal = dataOut_beam_noequal(:);
[numErrorsG_beam_noequal_nobeam,berG_beam_noequal_nobeam] = biterr(bitInput1,dataOut_beam_noequal(1:end))




% With beamformer without equalization
out = ofdmDemod1(arrOut);
figure;

x = real(out);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(out);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
scatter(x,y);

dataOut_beam_noequal = qamdemod(out,M1,'OutputType','bit');
dataOut_beam_noequal = dataOut_beam_noequal(:);
[numErrorsG_beam_noequal,berG_beam_noequal] = biterr(bitInput1,dataOut_beam_noequal(1:end))
title('si BF, no equal')


% With beamformer with equalization
chOut_equal= ofdmDemod1(arrOut_equal);
figure;

x = real(chOut_equal);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(chOut_equal);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
scatter(x,y);

dataOut_beam = qamdemod(chOut_equal,M1,'OutputType','bit');
dataOut_beam = dataOut_beam;
dataOut_beam = dataOut_beam(:);
[numErrorsG_beam,berG_beam] = biterr(bitInput1(length(bitInput1)-length(dataOut_beam)+1:end),dataOut_beam(1:end))
title('si BF, si equal')



%{
chOut = chOut1 + chOut2;
numberBF = 1 SIMPLE, 2 NULLING, 3 MVDR, 4 LMS, 5 MMSE
Nant = 1 4, 2 8, 3 16, 4 32
g_len = 1 1, 2 2, 3 34
M = 1 4, 2 16
Bers = zeros(numberBF, Nant, g_len, M)
Bers (1,1,1,1) = 0.2623
Bers (1,2,1,1) = 0.0051
Bers (1,3,1,1) = 2.4590e-04
Bers (1,1,2,1) = 0.023
Bers (1,1,3,1) = 0
Bers (1,1,1,2) = 0.5
Bers (1,3,3,1) = 0
Bers (1,3,3,2) = 0.25

Bers (2,1,1,1) = 0.0037
Bers (2,2,1,1) = 0
Bers (2,3,1,1) = 0
Bers (2,1,2,1) = 0.5
Bers (2,1,3,1) = 0.2546
Bers (2,1,1,2) = 0.5
Bers (2,3,3,1) = 0
Bers (2,3,3,2) = 0.25

Bers (3,3,3,1) = 0
Bers (3,3,3,2) = 0.45

Bers (4,3,1,1) = 0
Bers (4,3,3,1) = 0.25
Bers (4,3,1,2) = 0.25

Bers (5,3,1,1) = 0
Bers (5,4,1,1) = 0
Bers (5,3,1,2) = 0.27
Bers (5,4,1,2) = 0.25

%}
