clear all, clc, close all


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


% Define waveform
Fsin = 600;
Ts = 1e-5;
Fsample = 1/Ts; % 10 khz
TsVect = 0:Ts:5/Fsin; % up to 5 periods
sinusoid_waveform = sin(2*pi*Fsin*TsVect);





% Create Layout Object
l = qd_layout;
l.set_scenario('QuaDRiGa_UD2D_LOS');

% Define tx and rx
txArr = qd_arrayant('omni');
rxArr = qd_arrayant('omni');
rxArr.no_elements = 16;
rxArr.element_position = Geometry.BSAntennaPos;

l.tx_array = txArr;
l.rx_array = rxArr;
l.no_rx = 1;
l.no_tx = 4;
tx_track1 = qd_track('linear', Geometry.T1, pi/2);
tx_track1.name = 'trackV1';
tx_track2 = qd_track('linear', Geometry.T2, pi);
tx_track2.name = 'trackV2';
tx_track1.initial_position = Geometry.V1PosStart';
tx_track2.initial_position = Geometry.V2PosStart';
l.tx_position = [Geometry.V1PosStart', Geometry.V2PosStart',...
    Geometry.I1Pos', Geometry.I2Pos'];
l.tx_track(1,1) = copy(tx_track1);
l.tx_track(1,2) = copy(tx_track2);
l.rx_position = Geometry.BSPos';


% Visualize model
l.visualize();


% Run the model
l.set_pairing;
chan = l.get_channels;


% We have 4 channels:
% H (4=Rx,1=Tx) = [H11 H21 H31 H41]'

% First channel
% size(chan(1).coeff) = 16x1x34x2
% size(squeeze(chan(1).coeff(:,1,:,1))) = 16x34
% squeeze(chan(1).coeff(:,1,:,1)) = [c1_1(LOS) c1_2 .. c1_34 
%                                   [c2_1(LOS) c2_2 .. c2_34
%                                   ....               c11_34]                   
% Send sinusoid in chan(1) and compute Output
chTaps = size(chan(1).delay); % [16 1 34 2]
TS = Ts; % sampling time could be different for diff waveforms
WFlenght = size(sinusoid_waveform);
chOut = zeros(chTaps(1), WFlenght(2));
TsVect = 0:TS:TS*(WFlenght(2)-1);

for antenna=1:1:chTaps(1)
    for path=1:1:chTaps(3)
        inX = TsVect - chan(1).delay(antenna, 1, path, 1);
        inY = interp1(TsVect, sinusoid_waveform, inX, 'pchip');
        chOut(antenna, :) = inY * chan(1).coeff(antenna, 1, path, 1)...
            +chOut(antenna, :);
    end
end




% Add AWGN
% Pars.SNR = 20; % dB
% chOut = awgn(chOut, Pars.SNR, 'measured');

% chOut (16, 834) values of the sinusoid received in each of the 16
% antennas from the 1 tx


% PROVE
% inX2 = TsVect - chan(1).delay(1:chTaps(1), 1, 1:chTaps(3), 1);
% inY2 = interp1(TsVect, sinusoid_waveform, inX2, 'pchip');
% chOut2(1:chTaps(1), :) = inY2 * chan(1).coeff(1:chTaps(1), 1, 1:chTaps(3), 1)...
%     +chOut2(1:chTaps(1), :);
% pagemtimes(inY2,squeeze(permute(chan(1).coeff(1:chTaps(1), 1, 1:chTaps(3), 1),[3 2 1])))
