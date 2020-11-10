clear all, clc, close all

ris = zeros(3,1000);

for idx=1:1:170

Pars.fc = 1e9;
Pars.c = physconst('LightSpeed');
Pars.lambda = Pars.c/Pars.fc;

Geometry.BSPos = [0,0,25]; % macrocell with high 25m
Geometry.V1PosStart = [idx,0,1.5]; % start Veichile 1 [70,-100,1.5]
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

txPos = Geometry.BSPos;
htx = txPos(3);

rxPos = Geometry.V1PosStart;
hrx = rxPos(3);

% invert z coordinate
rxPosR = rxPos;
rxPosR(3) = -rxPosR(3);


% LOS distance
l = dist3D(txPos, rxPos);

% RELFECTED distance (rxPosR!)
x = dist3D(txPos, rxPosR);

% 2D distance d
d = dist2D(txPos, rxPos);

% DeltaD = relfected - los
DeltaD = x - l;


% power of sinusoid = A^2/2 (this case A=1), Gt=Gr=1 isotropic antenna
PrTheory = (1/2) * (1) * (Pars.lambda/(4*pi*l))^2*...
    (abs(1-exp(1i*(2*pi*DeltaD/(Pars.lambda)))))^2;

% Large d approx DeltaD = x - l = ~ 2*htx*hrx/d
PrApprox = (1/2) * (1) * (Pars.lambda/(4*pi*l))^2*...
    (abs(1-exp(1i*(2*pi*2*htx*hrx/(d*Pars.lambda)))))^2;

% very different since 2*htx*hrx/d = 15.708, DeltaD = 3 and lambda = 0.2997
[PrTheory, PrApprox];



% Using Matlab Model TwoRayChannel
pos1 = txPos';
pos2 = rxPos';
vel1 = [0;0;0]; % tx and rx not moving
vel2 = [0;0;0];
swav = sinusoid_waveform';

channel = phased.TwoRayChannel('SampleRate', Fsample,...
    'GroundReflectionCoefficient', -1, 'OperatingFrequency', Pars.fc,...
    'CombinedRaysOutput', true);
prop_signal = channel([swav,swav], pos1, pos2, vel1, vel2);

RxPow = mean(abs(prop_signal).^2);
[RxPow, PrTheory, PrApprox];

ris (1:3,idx) = [RxPow PrTheory PrApprox];

end

semilogy(ris(1,:));
hold on;
semilogy(ris(2,:));
hold on;
semilogy(ris(3,:));
legend('RxPow', 'PrTheory', 'PrApprox');


