clear all, clc, close all

%% Geometry and Pars
Pars.fc = 1e9;
Pars.c = physconst('LightSpeed');
Pars.lambda = Pars.c/Pars.fc;

Geometry.BSPos = [0,0,25]; % macrocell with high 25m
Geometry.V1PosStart = [70,-100,1.5]; % start Veichile 1
Geometry.V1PosEnd = [70,100,1.5]; % end Veichile 1
Geometry.V2PosStart = [200,-50,1.5]; % start Veichile 2
Geometry.V2PosEnd = [10,-50,1.5]; % end Veichile 2

Geometry.I1Pos = [10,-210,1.5]; 
Geometry.I2Pos = [-150,100,1.5];

% CreateScenarioAndVisualize(Geometry, Pars);
Geometry.T1 = dist3D(Geometry.V1PosStart, Geometry.V1PosEnd);  % distance covered by V1
Geometry.T2 = dist3D(Geometry.V2PosStart, Geometry.V2PosEnd);  % distance covered by V2

Geometry.DistV1Start = dist3D(Geometry.V1PosStart, Geometry.BSPos); % initial distance between BS and V1
Geometry.DistV2Start = dist3D(Geometry.V2PosStart, Geometry.BSPos); % initial distance between BS and V1


% DoA = [AoA ZoA] at beginning 
% (Direction of Arrival) = (Angle of Arriv. Zenith of Arriv.)
% Zenith = 90 - Elevation Angle
Geometry.AOAV1Start = AoA(Geometry.BSPos, Geometry.V1PosStart);
Geometry.ZOAV1Start = ZoA(Geometry.BSPos, Geometry.V1PosStart);
Geometry.DOAV1Start = [Geometry.AOAV1Start Geometry.ZOAV1Start]; % DOA of V1

Geometry.AOAV2Start = AoA(Geometry.BSPos, Geometry.V2PosStart);
Geometry.ZOAV2Start = ZoA(Geometry.BSPos, Geometry.V2PosStart);
Geometry.DOAV2Start = [Geometry.AOAV2Start Geometry.ZOAV2Start]; % DOA of V2


% Defining 4x4 antenna array BS, separated by lambda/2
Geometry.BSarray = phased.URA('Size',[4 4],...
    'ElementSpacing', [Pars.lambda/2 Pars.lambda/2], 'ArrayNormal', 'x');

% Position of elements of antenna array
Geometry.BSAntennaPos = getElementPosition(Geometry.BSarray);

Geometry.confarray = phased.ConformalArray('ElementPosition',Geometry.BSAntennaPos);
viewArray(Geometry.confarray);


%% Waveform, Modulators and Demodulators Generation
% OFDM configuration:
ofdmMod1 = comm.OFDMModulator('FFTLength', 12, ...
    'NumGuardBandCarriers', [1;1], ...
    'InsertDCNull', false, ...
    'CyclicPrefixLength', [0], ...
    'Windowing', false, ...
    'NumSymbols', 140, ...
    'NumTransmitAntennas', 1, ...
    'PilotInputPort', true, ...
    'PilotCarrierIndices', [11]);

M = 4; 	 % Modulation order
% input bit source:
in1 = randi([0 1], 2520, 1);

dataInput1 = qammod(in1, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
ofdmInfo1 = info(ofdmMod1);
ofdmSize1 = ofdmInfo1.DataInputSize;
dataInput1 = reshape(dataInput1, ofdmSize1);

% waveform generation:
pilotInput1 = ones(1, 140, 1);
waveform1 = ofdmMod1(dataInput1, pilotInput1);

Fs1 = 180000; 	



% OFDM configuration:
ofdmMod2 = comm.OFDMModulator('FFTLength', 12, ...
    'NumGuardBandCarriers', [1;1], ...
    'InsertDCNull', false, ...
    'CyclicPrefixLength', [0], ...
    'Windowing', false, ...
    'NumSymbols', 140, ...
    'NumTransmitAntennas', 1, ...
    'PilotInputPort', true, ...
    'PilotCarrierIndices', [2]);

M = 4; 	 % Modulation order
% input bit source:
in2 = randi([0 1], 2520, 1);

dataInput2 = qammod(in2, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
ofdmInfo2 = info(ofdmMod2);
ofdmSize2 = ofdmInfo2.DataInputSize;
dataInput2 = reshape(dataInput2, ofdmSize2);

% waveform generation:
pilotInput2 = ones(1, 140, 1);
waveform2 = ofdmMod2(dataInput2, pilotInput2);

Fs2 = 180000; 		


% Demodulators
ofdmDemod1 = comm.OFDMDemodulator(ofdmMod1);
ofdmDemod2 = comm.OFDMDemodulator(ofdmMod2);


%% Channnel generation
% Calculation on waveform
chOut = collectPlaneWave(Geometry.BSarray, [waveform1 waveform2],... % attention to dimension of waveforms
        [Geometry.DOAV1Start', Geometry.DOAV2Start'], Pars.fc);
    

% Add AWGN
Pars.SNR = 20; % dB
chOut = awgn(chOut, Pars.SNR, 'measured');




%% DOA Estimation with MUSIC
estimator = phased.MUSICEstimator2D(...
    'SensorArray',Geometry.BSarray,...
    'OperatingFrequency',Pars.fc,...
    'ForwardBackwardAveraging',true,...
    'NumSignalsSource','Property',...
    'DOAOutputPort',true,'NumSignals',2,...
    'AzimuthScanAngles',-90:.1:90,...
    'ElevationScanAngles',0:0.5:90);
[~,doas] = estimator(chOut);
doas(1,:) = -(doas(1,:)-180);
temp1 = doas(:,1);
doas(:,1) = doas(:,2);
doas(:,2) = temp1;

figure
plotSpectrum(estimator);



%% Simple beamformer
beamformer = phased.PhaseShiftBeamformer(...
    'SensorArray',Geometry.BSarray,...
    'OperatingFrequency',Pars.fc,'PropagationSpeed',Pars.c,...
    'Direction',doas(:,1),'WeightsOutputPort',true);
% beamformer = phased.PhaseShiftBeamformer(...
%     'SensorArray',Geometry.BSarray,...
%     'OperatingFrequency',Pars.fc,'PropagationSpeed',Pars.c,...
%     'Direction',Geometry.DOAV1Start','WeightsOutputPort',true);
[arrOut,w] = beamformer(chOut);


% % Plot Output of Beamformer
% figure;
% plot([0:1/Fs1:length(abs(arrOut))/Fs1-1/Fs1],abs(arrOut)); axis tight;
% title('Output of Beamformer');
% xlabel('Time (s)');ylabel('Magnitude (V)');


% Plot array pattern at azimuth = 0°
figure;
pattern(Geometry.BSarray,Pars.fc,[-180:180],0,...
    'PropagationSpeed',Pars.c,...
    'Type','powerdb',...
    'CoordinateSystem','polar','Weights',w)

figure;
pattern(Geometry.BSarray,Pars.fc,[-180:180],0,...
    'PropagationSpeed',Pars.c,...
    'Type','powerdb',...
    'CoordinateSystem','rectangular','Weights',w)



%% Scattergraph to see impact of BF

% Without beamformer
out = ofdmDemod1(chOut(:,1)); % first antenna
figure;

x = real(out);
x = reshape(x,[9*140,1]);
y = imag(out);
y = reshape(y,[9*140,1]);
scatter(x,y);

dataOut_notbeam = qamdemod(out,M,'OutputType','bit');
dataOut_notbeam = dataOut_notbeam(:);
[numErrorsG_notbeam,berG_notbeam] = biterr(in1,dataOut_notbeam)


% With beamformer
out = ofdmDemod1(arrOut);
figure;

x = real(out);
x = reshape(x,[9*140,1]);
y = imag(out);
y = reshape(y,[9*140,1]);
scatter(x,y);

dataOut_beam = qamdemod(out,M,'OutputType','bit');
dataOut_beam = dataOut_beam(:);
[numErrorsG_beam,berG_beam] = biterr(in1,dataOut_beam)



