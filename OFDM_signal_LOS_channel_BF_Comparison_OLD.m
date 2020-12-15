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
Geometry.Nant = 8;
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
    chOut_no_noise = collectPlaneWave(Geometry.BSarray, [w1 w2], ...
        [Geometry.DOAV1Start', Geometry.DOAV2Start'], Pars.fc);
for i = 1:11 
    SNR(i) = i+9; % in dB
    % Adding AWGN noise to waveform: 
    chOut_noise = awgn(chOut_no_noise, 0, 'measured');
    %noise = chOut_noise - chOut_no_noise;
    noise=10*log10(sum(abs(fft(chOut_noise(:,1))).^2)/sum(abs(fft(chOut_no_noise(:,1))).^2));
    chOut= chOut_noise;
    
   
    %SNR_IN(i)=(rms(chOut(:,1))/rms(noise(:,1)))^2;
    %SNR_IN(i)=abs(mean(chOut.^2))/abs(mean(noise.^2));
    SNR_IN(i)=(sum(abs(fft(chOut)).^2)/sum(abs(fft(noise)).^2));
    SNR_IN_dB(i)=floor(10*log10(SNR_IN(i)));
    
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




%% Beamformer 0 SIMPLE, 1 NULLING, 2 MVDR, 3 LMS, 4 MMSE
    for j=1:5
            
        k=j-1;

        switch k

            case 0
                [chOut_BF,w] = Conventional_BF(Geometry, Pars, DoAs(:,1), chOut);
                [chOut_no,w] = Conventional_BF(Geometry, Pars, DoAs(:,1), chOut_no_noise);
                noise=chOut_BF-chOut_no;
                [int_BF,w] = Conventional_BF(Geometry, Pars, DoAs(:,1), int);
            case 1     
                [chOut_BF,w] = Nullsteering_BF(Geometry, Pars, DoAs, chOut);

            case 2       
                [chOut_BF,w] = MVDR_BF(Geometry, Pars, DoAs(:,1), chOut);

            case 3
                % We use the first half of the received signal for the LMS algorithm, 
                % finding then the weights to be assigned to each antenna and applying the
                % to the received singal for better reception.

                % Training sequence length:
                nTrain = round(length(chOut(:,1)) / 2);

                [chOut_BF, w] = LMS_BF(Geometry, Pars, DoAs(:, 1), chOut, waveform1(1:nTrain, :));

            case 4   
                % Training sequence length:
                nTrain = round(length(chOut(:,1)) / 2);

                [chOut_BF,w] = MMSE_BF(Geometry, Pars, chOut, waveform1(1:nTrain, :));
                
        end
    SNR_OUT(i,j)=(sum(abs(fft(chOut_BF)).^2)/(sum(abs(fft(noise(:))).^2)+(sum(abs(fft(int_BF)).^2))));
    SNR_OUT_dB(i,j)=10*log10(SNR_OUT(i,j));        
    %SNR_OUT(i,j)=(rms(chOut_BF)/rms(noise(:)))^2;
    %SNR_OUT_dB(i,j)=10*log10(SNR_OUT(i,j));  
    end
end

%% Plot the results

