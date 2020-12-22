%% DESCRIPTION
% In this script, we're showing the perfomace of the LMS beamformer
% in frequency domain while tracking a vehicle in presence of an
% interference.

%% Preparing 

clear all;
close all;
clc;

set(0,'DefaultTextFontSize',18)
set(0,'DefaultLineLineWidth',2);
%set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesFontSize',16)

%% Defining geometry and pars

% CArrier frequency and wavelength
Pars.fc = 2.6e9;
Pars.c = physconst('LightSpeed');
Pars.lambda = Pars.c / Pars.fc;

% BS position (macrocell with high 25m):
Geometry.BSPos = [0, 0, 25];

% Simulation Time
Geometry.TotTime = 12; % [s]

% Number of packets
Geometry.N_Pack = 24;

% Number of Tx
Geometry.N_vehicle = 2;

% First veichle (V1):
Geometry.V1PosStart = [70, -100, 1.5]; % start [m]
Geometry.V1PosEnd = [70, 100, 1.5];    % end [m]
Geometry.V1Positions = Positions(Geometry.V1PosStart, Geometry.V1PosEnd, Geometry.N_Pack);

% Second veichle (V2):
Geometry.V2PosStart = [200, -50, 1.5]; % start [m]
Geometry.V2PosEnd = [0, -50, 1.5];    % end [m]
Geometry.V2Positions = Positions(Geometry.V2PosStart, Geometry.V2PosEnd, Geometry.N_Pack);



% Velocities of veichles:
Geometry.vel1 = [0;200;0].* 1/12; % 16.67 [m/s] = 60 [km/h]
Geometry.vel2 = [-200;0;0].* 1/12; % 16.67 [m/s] = 60 [km/h]

% Coherence Time
Geometry.fd = Pars.fc* max([sqrt(sum(Geometry.vel1.^2)) ; sqrt(sum(Geometry.vel2.^2))])/Pars.c; % [Hz]
Geometry.Tc = 1/(20*Geometry.fd); % [s]

% Frequency spacing
Geometry.delta_f = 60e3; % [Hz]

% Symbol Time
Geometry.Ts = 17.84e-6; % [s]

% Coherence Time in Symbols
Geometry.Tc_symb = floor(Geometry.Tc/Geometry.Ts);

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




%% Paramters of Modulations

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



    
%% DEFINITION OF OUTPUTS TO BE COMPARED

% Detected direction of arrivals:
DoAs_tot = zeros(Geometry.N_Pack, Geometry.N_vehicle, 2);

% Error of Music estimator on azimuth angle
DoAs_error = zeros(Geometry.N_Pack, Geometry.N_vehicle, 2);

% Real DoAs
DoAs_tot_eff = zeros(Geometry.N_Pack, Geometry.N_vehicle, 2);


% BER of vehicle1
ber_tot_v1 = ones(Geometry.N_Pack,1);
% BER of vehicle2
ber_tot_v2 = ones(Geometry.N_Pack,1);


% Signals OFDM-demodulated of vehicle1
chOut_OFDMdem_v1_tot = zeros( Geometry.N_Pack ,Geometry.Nant^2 , (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);

% Signals OFDM-demodulated and squeezed of vehicle1 from antenna 1
chOut_OFDMdem_v1_1_tot = zeros( Geometry.N_Pack, (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);

% Signals OFDM-demodulated after BF but without equalization of vehicle1:
chOut_OFDMdem_v1_BF_tot = zeros( Geometry.N_Pack, (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);

% Signals OFDM-demodulated after BF but with equalization of vehicle1:
chOut_OFDMdem_v1_BF_equal_tot = zeros( Geometry.N_Pack, (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);


% Signals OFDM-demodulated of vehicle2
chOut_OFDMdem_v2_tot = zeros( Geometry.N_Pack ,Geometry.Nant^2 , (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);

% Signals OFDM-demodulated and squeezed of vehicle2 from antenna 1
chOut_OFDMdem_v2_1_tot = zeros( Geometry.N_Pack, (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);

% Signals OFDM-demodulated after BF but without equalization of vehicle2:
chOut_OFDMdem_v2_BF_tot = zeros( Geometry.N_Pack, (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);

% Signals OFDM-demodulated after BF but with equalization of vehicle2:
chOut_OFDMdem_v2_BF_equal_tot = zeros( Geometry.N_Pack, (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);





for n=1:Geometry.N_Pack
    
% Adjust position of target
Geometry.V1PosStart = [Geometry.V1Positions(:,n)]';
Geometry.V2PosStart = [Geometry.V2Positions(:,n)]';

% Updated DoA = [AoA ZoA] (ZoA = 90 - elevation angle) for the two vehicles:
Geometry.AOAV1Start = AoA(Geometry.BSPos, Geometry.V1PosStart);
Geometry.ZOAV1Start = ZoA(Geometry.BSPos, Geometry.V1PosStart);
Geometry.DOAV1Start = [Geometry.AOAV1Start Geometry.ZOAV1Start]; % DoA of V1

Geometry.AOAV2Start = AoA(Geometry.BSPos, Geometry.V2PosStart);
Geometry.ZOAV2Start = ZoA(Geometry.BSPos, Geometry.V2PosStart);
Geometry.DOAV2Start = [Geometry.AOAV2Start Geometry.ZOAV2Start]; % DOA of V2 
   
Geometry.DOAV1Start(1) = wrapTo180(Geometry.DOAV1Start(1) -180);
Geometry.DOAV2Start(1) = wrapTo180(Geometry.DOAV2Start(1) -180);
Geometry.DOAV1Start(2) = 90 - Geometry.DOAV1Start(2);
Geometry.DOAV2Start(2) = 90 - Geometry.DOAV2Start(2);

DoAs_tot_eff(n, :, :) = [Geometry.DOAV1Start'  Geometry.DOAV2Start'];

%% Generation of ODFM modulators and demodulators, M-QAM modulators and waveforms

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

% OFDM demodulators definition:
ofdmDemod1 = comm.OFDMDemodulator(ofdmMod1);
ofdmDemod2 = comm.OFDMDemodulator(ofdmMod2);

% Visualizing OFDM mapping:
% showResourceMapping(ofdmMod1);
% title('OFDM modulators (1 = 2)');



%% 3GPP Channel 
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
l.no_tx = 2;

% tx_track1 = qd_track('linear', Geometry.T1, pi/2);
tx_track1 = qd_track('linear', 0, pi/2);
tx_track1.name = 'trackV1';

% tx_track2 = qd_track('linear', Geometry.T2, pi);
tx_track2 = qd_track('linear', 0, pi);
tx_track2.name = 'trackV2';

tx_track1.initial_position = Geometry.V1PosStart';
tx_track2.initial_position = Geometry.V2PosStart';

l.tx_position = [Geometry.V1PosStart', Geometry.V2PosStart'];
% l.tx_track(1,1) = copy(tx_track1);
% l.tx_track(1,2) = copy(tx_track2);
l.rx_position = Geometry.BSPos';


% Visualize model
%channel = l.visualize();
%title('Scenario')
channel = PlotScenario(Geometry);
saveas(channel,sprintf('Quadriga%d.png',n))


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

% Adding AWGN noise to waveform:
Pars.SNR = 20; % in dB

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
    'AzimuthScanAngles', -180:.5:180, ...
    'ElevationScanAngles', 0:0.5:90);

% Estimation od DoA of singal in output from the channel:
[~, DoAs] = estimator(chOut);
if n == 1
    DoAs_tot(n,:,:) = DoAs_tot_eff(n,:,:);
else
    % Compare with DoAs of previous estimation and assign DoAs to correct
    % vehicles
    N_positions = 2;
    DoAs = reorderDoAs(squeeze(DoAs_tot_eff(n-1,:,:)), DoAs, N_positions);
    DoAs_tot(n,:,:) = DoAs;
end

% Using real angles
DoAs = squeeze(DoAs_tot_eff(n,:,:));

% Plotting and Saving images:
fig = plotSpectrum(estimator);
saveas(fig,sprintf('DOAs%d.png',n))

% Compute error of Music estimator
DoAs_error(n,:,:) = DoAs_tot(n,:,:) - DoAs_tot_eff(n,:,:);



%% OFDM demodulation and LMS Beamforming in Frequency
% Signal after fft
chOut_OFDMdem_v1 = zeros( Geometry.Nant^2 , (nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers))),  nSymbols1);
chOut_OFDMdem_v2 = zeros( Geometry.Nant^2 , (nfft - (length(pilot_indices2) + sum(NumGuardBandCarriers))),  nSymbols2);


% Demodulation the Signal from each Antenna
for ant = 1: size(chOut,2)
    
    chOut_OFDMdem_v1(ant,:,:) = ofdmDemod1(chOut(:,ant));

end
% chOut_OFDMdem_v1_tot(n,:,:,:) = chOut_OFDMdem_v1;
for ant = 1: size(chOut,2)
    
    chOut_OFDMdem_v2(ant,:,:) = ofdmDemod2(chOut(:,ant));

end
% chOut_OFDMdem_v2_tot(n,:,:,:) = chOut_OFDMdem_v2;


% Signal after fft and Beamforming
chOut_OFDMdem_v1_BF = zeros (size(chOut_OFDMdem_v1,2), size(chOut_OFDMdem_v1,3));
chOut_OFDMdem_v2_BF = zeros (size(chOut_OFDMdem_v2,2), size(chOut_OFDMdem_v2,3));


% Weights for each Antenna at each Sub-Carrier
w_v1 = zeros (size(chOut_OFDMdem_v1,2), size(chOut_OFDMdem_v1,1));
w_v2 = zeros (size(chOut_OFDMdem_v2,2), size(chOut_OFDMdem_v2,1));


% Apply Beamforming to each Sub-Carrier
for sc = 1: size(chOut_OFDMdem_v1,2)

        nTrain = round(size(chOut_OFDMdem_v1,3) / 2);

        [chOut_OFDMdem_v1_BF(sc,:), w_v1(sc,:)] = LMS_BF(Geometry, Pars, DoAs(:, 1), squeeze(chOut_OFDMdem_v1(:,sc,:)).', dataInput1(sc, 1:nTrain).');
    
end
% chOut_OFDMdem_v1_BF_tot(n,:,:) = chOut_OFDMdem_v1_BF;
for sc = 1: size(chOut_OFDMdem_v2,2)

        nTrain = round(size(chOut_OFDMdem_v2,3) / 2);

        [chOut_OFDMdem_v2_BF(sc,:), w_v2(sc,:)] = LMS_BF(Geometry, Pars, DoAs(:, 2), squeeze(chOut_OFDMdem_v2(:,sc,:)).', dataInput2(sc, 1:nTrain).');
    
end
% chOut_OFDMdem_v2_BF_tot(n,:,:) = chOut_OFDMdem_v2_BF;


%%  Channel equalization
n_training_v1 = size(chOut_OFDMdem_v1_BF,2);
n_training_v2 = size(chOut_OFDMdem_v2_BF,2);
    
max_iter = 100;

% Tap of the equalizer
g_len = 2;

chOut_OFDMdem_v1_BF_equal = zeros (size(chOut_OFDMdem_v1_BF));
chOut_OFDMdem_v2_BF_equal = zeros (size(chOut_OFDMdem_v2_BF));

% Apply Gradient-Descent algorithm to each Sub-Carrier
for sc = 1: size(chOut_OFDMdem_v1,2)
    
    g = Gradient_descent( chOut_OFDMdem_v1_BF(sc,:),  dataInput1(sc,1:n_training_v1), n_training_v1, max_iter, g_len);

    chOut_OFDMdem_v1_BF_ = conv(g,chOut_OFDMdem_v1_BF(sc,:));

    chOut_OFDMdem_v1_BF_equal(sc,:) = chOut_OFDMdem_v1_BF_(1:length(chOut_OFDMdem_v1_BF(sc,:)));

end
chOut_OFDMdem_v1_BF_equal_tot(n,:,:) = chOut_OFDMdem_v1_BF_equal;
for sc = 1: size(chOut_OFDMdem_v2,2)
    
    g = Gradient_descent( chOut_OFDMdem_v2_BF(sc,:),  dataInput2(sc,1:n_training_v2), n_training_v2, max_iter, g_len);

    chOut_OFDMdem_v2_BF_ = conv(g,chOut_OFDMdem_v2_BF(sc,:));

    chOut_OFDMdem_v2_BF_equal(sc,:) = chOut_OFDMdem_v2_BF_(1:length(chOut_OFDMdem_v2_BF(sc,:)));

end
chOut_OFDMdem_v2_BF_equal_tot(n,:,:) = chOut_OFDMdem_v2_BF_equal;


%% QAM demodulation

% No beamformer without equalization v1:
chOut_OFDMdem_v1_1 = squeeze(chOut_OFDMdem_v1(1,:,:)); % first antenna
chOut_OFDMdem_v1_1_tot(n,:,:) = chOut_OFDMdem_v1_1;
figure;

x = real(chOut_OFDMdem_v1_1);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(chOut_OFDMdem_v1_1);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
scatter(x,y);
title('Without beamforming, without equalization')

chOut_OFDMdem_v1_1_QAMdem = qamdemod(chOut_OFDMdem_v1_1,M1,'OutputType','bit', 'UnitAveragePower', true);
chOut_OFDMdem_v1_1_QAMdem = chOut_OFDMdem_v1_1_QAMdem(:);
[numErrorsG_beam_noequal_nobeam,berG_beam_noequal_nobeam] = biterr(bitInput1,chOut_OFDMdem_v1_1_QAMdem(1:end))


% No beamformer without equalization v2:
chOut_OFDMdem_v2_1 = squeeze(chOut_OFDMdem_v2(1,:,:)); % first antenna
chOut_OFDMdem_v2_1_tot(n,:,:) = chOut_OFDMdem_v2_1;
figure;

x = real(chOut_OFDMdem_v2_1);
x = reshape(x,[(nfft - (length(pilot_indices2) + sum(NumGuardBandCarriers)))*nSymbols2,1]);
y = imag(chOut_OFDMdem_v2_1);
y = reshape(y,[(nfft - (length(pilot_indices2) + sum(NumGuardBandCarriers)))*nSymbols2,1]);
scatter(x,y);
title('Without beamforming, without equalization')

chOut_OFDMdem_v2_1_QAMdem = qamdemod(chOut_OFDMdem_v2_1,M2,'OutputType','bit', 'UnitAveragePower', true);
chOut_OFDMdem_v2_1_QAMdem = chOut_OFDMdem_v2_1_QAMdem(:);
[numErrorsG_beam_noequal_nobeam,berG_beam_noequal_nobeam] = biterr(bitInput2,chOut_OFDMdem_v2_1_QAMdem(1:end))

 

% With beamformer with equalization v1:
figure;

x = real(chOut_OFDMdem_v1_BF_equal);
x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
y = imag(chOut_OFDMdem_v1_BF_equal);
y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]); 
scatter(x,y);
title('With beamforming and equalization')

chOut_OFDMdem_v1_BF_equal_QAMdem = qamdemod(chOut_OFDMdem_v1_BF_equal,M1,'OutputType','bit', 'UnitAveragePower', true);
chOut_OFDMdem_v1_BF_equal_QAMdem = chOut_OFDMdem_v1_BF_equal_QAMdem(:);
[numErrorsG_beam,berG_beam] = biterr(bitInput1(length(bitInput1)-length(chOut_OFDMdem_v1_BF_equal_QAMdem)+1:end),chOut_OFDMdem_v1_BF_equal_QAMdem(1:end))

ber_tot_v1(n) = berG_beam;



% With beamformer with equalization v2:
figure;

x = real(chOut_OFDMdem_v2_BF_equal);
x = reshape(x,[(nfft - (length(pilot_indices2) + sum(NumGuardBandCarriers)))*nSymbols2,1]);
y = imag(chOut_OFDMdem_v2_BF_equal);
y = reshape(y,[(nfft - (length(pilot_indices2) + sum(NumGuardBandCarriers)))*nSymbols2,1]); 
scatter(x,y);
title('With beamforming and equalization')

chOut_OFDMdem_v2_BF_equal_QAMdem = qamdemod(chOut_OFDMdem_v2_BF_equal,M2,'OutputType','bit', 'UnitAveragePower', true);
chOut_OFDMdem_v2_BF_equal_QAMdem = chOut_OFDMdem_v2_BF_equal_QAMdem(:);
[numErrorsG_beam,berG_beam] = biterr(bitInput2(length(bitInput2)-length(chOut_OFDMdem_v2_BF_equal_QAMdem)+1:end),chOut_OFDMdem_v2_BF_equal_QAMdem(1:end))

ber_tot_v2(n) = berG_beam;


close all
end



%% PLOTS THE RESULTS OF TRACKING
% Saves variables
%save('Variables1')
%load('Variables1');

% Create a VideoWriter object for a new video file.
v = VideoWriter('File3.avi');
v.FrameRate = 1;
open(v)

% Plot of the constellations, the DoAs and the Scenario
for n = 1 : Geometry.N_Pack
    
    frame = figure();
    setappdata(gcf, 'SubplotDefaultAxesLocation', [0, 0, 1, 1]);
    set(gcf,'WindowState','Maximized')
    pause(0.1)

    
    Rect    = [0.03, 0.03, 0.95, 0.94];
    AxisPos = PlotPos(2, 2, Rect);
      
    % plot3 = subplot(2,2,1)
    axes('Position', AxisPos(1, :));
    x = real(chOut_OFDMdem_v1_BF_equal_tot(n,:,:));
    x = reshape(x,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]);
    y = imag(chOut_OFDMdem_v1_BF_equal_tot(n,:,:));
    y = reshape(y,[(nfft - (length(pilot_indices1) + sum(NumGuardBandCarriers)))*nSymbols1,1]); 
    scatter(x,y);   
    title(sprintf('With beamforming and equalization V1, BER = %0.5g', ber_tot_v1(n)), 'FontSize', 18)
    
    
    % plot2 = subplot(2,2,2)
    axes('Position', AxisPos(2, :));
    img1 = imread(sprintf('DOAs%d.png',n));
    imshow(img1)
    title('DoAs', 'FontSize', 18)
    
    
    % plot3 = subplot(2,2,3)
    axes('Position', AxisPos(3, :));
    x = real(chOut_OFDMdem_v2_BF_equal_tot(n,:,:));
    x = reshape(x,[(nfft - (length(pilot_indices2) + sum(NumGuardBandCarriers)))*nSymbols2,1]);
    y = imag(chOut_OFDMdem_v2_BF_equal_tot(n,:,:));
    y = reshape(y,[(nfft - (length(pilot_indices2) + sum(NumGuardBandCarriers)))*nSymbols2,1]); 
    scatter(x,y);   
    title(sprintf('With beamforming and equalization V2, BER = %0.5g', ber_tot_v2(n)), 'FontSize', 18)
    
    % plot4 = subplot(2,2,4)
    axes('Position', AxisPos(4, :));
    img2 = imread(sprintf('Quadriga%d.png',n));
    imshow(img2)
    title('Scenario', 'FontSize', 18)
    
    F = getframe(frame);
    writeVideo(v,F);
    writeVideo(v,F);
    pause(0.1)
    close all
    
end
close(v)



