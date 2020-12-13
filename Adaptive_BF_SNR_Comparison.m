%% Preparing 

clear all;
close all;
clc;

%% Defining Pars

% CArrier frequency and wavelength
Pars.fc = 2.4e+9; %[Hz] Frequency
Pars.c = physconst('LightSpeed');% Lightspeed
Pars.lambda = Pars.c / Pars.fc; %[m] Wavelength

%% Defining Geometry

Geometry.Nant = 8; %Number of antennas for each array.
Geometry.AntSpacing=0.5; %[m] x lambda
Geometry.NSource=1; %Number of arrays
   
% spherical coordinates of the directivity function of single element
Geometry.tetadir = -pi/2 : pi/256 : pi/2; % elevation - positive half-space [-pi/2 to pi/2]
Geometry.tetadir(1,end) = Geometry.tetadir(1,end) - 0.001;
Geometry.phidir = -pi : pi/64 : pi; % azimuth [-pi to pi]
Geometry.phidir(1,end) = Geometry.phidir(1,end) - 0.001;

Geometry.tetads = [-89,0,89] * pi / 180;
% Geometry.tetads = (-89 : 1 : 89) * pi / 180;  %Orientation of the Array


%% Adaptive Beamforming Data
    
% 2D
source.x = 0;
source.y = 100;
source.teta = pi/2 - angle(source.x+1i*source.y);
[s_corr]=steervector(source.teta,Geometry.Nant,Pars.lambda*0.5,Pars.fc);
source.s = s_corr.';

init.x = 0;
init.y = 1;
[s_corr]=steervector(0,Geometry.Nant,Pars.lambda*0.5,Pars.fc);
init.s = s_corr.';

% conventional BF
w = init.s;
% reference
d = w'*init.s;

N_iter = 800;

SNR = [0:10]; % input SNR

N_interf = 1;
N_interf_test=1; %use the same value as N_interf
ang_sep = 20; %Angular separation
SNRout = zeros(5,length(SNR)); %SNR output with interference
SNRout0 = zeros(5,length(SNR)); %SNR output without interference

%% No interference
if(N_interf == 0)
    Ps = 1; % signal power
    S0 = [source.s];
    U0 = Ps * eye(1);
    Rs0 = S0*U0*S0';
    p = source.s;
   for i = 1:length(SNR)
            sigma2n = Ps / (10^(SNR(i)/10));
            Rn = sigma2n * eye(Geometry.Nant);
            Ru = Rs0 + Rn;

            % conventional BF
            w = source.s;
            Pn = w'*Rn*w; %Noise power
            Psout = w'*Rs0*w; %Output signal power
            SNRout0(1,i) = 10*log10(Psout/Pn);
            w_simple = w/Geometry.Nant;
            

            % null-steering BF
            g1 = [1,zeros(1,N_interf)]';
            % S0 = S+1e-4*ones(size(S));
            w = g1'*S0'*inv(S0*S0'+0.01*eye(Geometry.Nant));
            w = w';
            Pn = w'*Rn*w; %Noise power
            Psout = w'*Rs0*w; %Output signal power
            SNRout0(2,i) = 10*log10(Psout/Pn);
            w_null = w;
            
            % MVDR BF
            w = inv(Ru)*source.s/(source.s'*inv(Ru)*source.s);
            Pn = w'*Rn*w;
            Psout = w'*Rs0*w;
            SNRout0(3,i) = 10*log10(Psout/Pn);
                        
            % MMSE BF
            w = inv(Ru)*p;
            w_mmse = w;
            Pn = w'*Rn*w; %Noise power
            Psout = w'*Rs0*w; %Output signal power
            SNRout0(4,i) = 10*log10(Psout/Pn);
                        
            % LMS BF
            mu = 1/(256*trace(Ru));
            w_iter = zeros(Geometry.Nant,N_iter);
            SNR_iter = zeros(1,N_iter);
            SNR_iter0 = zeros(1,N_iter);
            initial_cond = 1; % 0, 1, 2
            if (initial_cond == 0)
                init_ang = -pi/2 + pi*rand;
                [s_corr]=steervector(init_ang,Geometry.Nant,Pars.lambda*0.5,Pars.fc);
                w_iter(:,1) = s_corr.';
            elseif (initial_cond == 1)
                w_iter(:,1) = init.s;
            elseif (initial_cond == 2)
                w_iter(:,1) = w_mmse*Geometry.Nant;
            end
            w = w_iter(:,1);
            for iter = 2:N_iter
                u_corr = source.s + sqrt(sigma2n/2)*(randn(Geometry.Nant,1)+1i*randn(Geometry.Nant,1));
                e_corr = (w'*u_corr - d);
                w = w_iter(:,iter-1) - mu*u_corr*conj(e_corr);
                w_iter(:,iter) = w;
                Pn = w'*Rn*w; %Noise power
                Psout = w'*Rs0*w;%Output signal power
                SNR_iter0(1,iter) = Psout/Pn;
                
            end
            SNR_LMS(1,i) = (real(SNR_iter(1,N_iter)));
            SNRout0(5,i) = 10*log10(real(SNR_iter0(1,N_iter)));
            w_lms = w_iter(:,end)/Geometry.Nant;

   end
%Print the results
figure();    
plot(SNR, SNRout0(1,:),'o');
hold on;
plot(SNR, SNRout0(2,:),'s');
hold on;
plot(SNR, SNRout0(3,:),'+');
hold on;
plot(SNR, SNRout0(4,:),'*');
hold on;
plot(SNR, SNRout0(5,:),'p');
hold off;
legend('SIMPLE','NULL-STEERING', 'MVDR', 'MMSE', 'LMS');
xlabel('SNR (Input)[dB]'); 
ylabel('SNR (Output)[dB]');
title('Adaaptive Beam-forming performance with interference')
end

%% With interfernce
if(N_interf>0)
    
    for it = 1:N_interf_test
        
        interf.s = [];
        interf.teta = [];
        for iint = 1:N_interf
            if (N_interf_test == 1)
                interf.x = [1];
                interf.y = [100];
                % interf.teta = pi/2 - angle(interf.x+1i*interf.y);
                if N_interf==1
                    interf.teta(1,iint) = source.teta + ang_sep*pi/180;
                else
                    interf.teta(1,iint) = -pi/2 + pi*rand;
                end
                % source.teta = pi/2 - angle(source.x+1i*source.y);
            else
                interf.teta(1,iint) = -pi/2 + pi*rand;
            end
            [s_corr]=steervector(interf.teta(1,iint),Geometry.Nant,Pars.lambda*0.5,Pars.fc);
            interf.s = [interf.s,s_corr.'];
        end
                
        Ps = 1; % signal power

        % total signal
        S = [source.s,interf.s];
        U = Ps * eye(1+N_interf);
        Rs = S*U*S';
        %
        % signal
        S0 = [source.s];
        U0 = Ps * eye(1);
        Rs0 = S0*U0*S0';
        p = source.s;
        % interference
        if (N_interf > 0)
            Si = [interf.s];
            Ui = Ps * eye(N_interf);
            Rsi = Si*Ui*Si';
        else
            Rsi = zeros(Geometry.Nant,Geometry.Nant);
        end

        for isnr = 1:length(SNR)
            
            sigma2n = Ps / (10^(SNR(isnr)/10));
            Rn = sigma2n * eye(Geometry.Nant);

            Ru = Rs + Rn;

            % simple BF
            w = source.s;
            Pn = w'*Rn*w; %Noise power
            Psout = w'*Rs0*w;%Output signal power
            Piout = w'*Rsi*w;%Interference power
            SNRout(1,isnr) = SNRout(1,isnr) + Psout/(Pn+Piout);
            w_simple = w/Geometry.Nant;
            %

            % null-steering BF
            g1 = [1,zeros(1,N_interf)]';
            % S0 = S+1e-4*ones(size(S));
            w = g1'*S'*inv(S*S'+0.01*eye(Geometry.Nant));
            w = w';
            Pn = w'*Rn*w; %Noise power
            Psout = w'*Rs0*w; %Output signal power
            Piout = w'*Rsi*w;%Interference power
            SNRout(2,isnr) = SNRout(2,isnr) + Psout/(Pn+Piout);
            w_null = w;
            
            % MVDR BF
            w = inv(Ru)*source.s/(source.s'*inv(Ru)*source.s);
            Pn = w'*Rn*w; %Noise power
            Psout = w'*Rs0*w; %Output signal power
            Piout = w'*Rsi*w; %Interference power
            SNRout(3,isnr) = SNRout(3,isnr) + Psout/(Pn+Piout);

            % MMSE BF
            w = inv(Ru)*p;
            w_mmse = w;
            Pn = w'*Rn*w; %Noise power
            Psout = w'*Rs0*w; %Output signal power
            Piout = w'*Rsi*w; %Interference power
            SNRout(4,isnr) = SNRout(4,isnr) + Psout/(Pn+Piout);
            SNR_MMSE(it,isnr) = Psout/(Pn+Piout);
            
            % LMS BF
            mu = 1/(256*trace(Ru));
            w_iter = zeros(Geometry.Nant,N_iter);
            SNR_iter = zeros(1,N_iter);
            SNR_iter0 = zeros(1,N_iter);
            initial_cond = 2; % 0, 1, 2(Good)
            if (initial_cond == 0)
                init_ang = -pi/2 + pi*rand;
                [s_corr]=steervector(init_ang,Geometry.Nant,RB.lambda*0.5,RB.f);
                w_iter(:,1) = s_corr.';
            elseif (initial_cond == 1)
                w_iter(:,1) = init.s;
            elseif (initial_cond == 2)
                w_iter(:,1) = w_mmse*Geometry.Nant;
            end
            w = w_iter(:,1);
            for iter = 2:N_iter
                if (N_interf > 0)
                    u_corr = source.s + sum(interf.s,2) + sqrt(sigma2n/2)*(randn(Geometry.Nant,1)+1i*randn(Geometry.Nant,1));
                else
                    u_corr = source.s + sqrt(sigma2n/2)*(randn(Geometry.Nant,1)+1i*randn(Geometry.Nant,1));
                end
                e_corr = (w'*u_corr - d);
                w = w_iter(:,iter-1) - mu*u_corr*conj(e_corr);
                w_iter(:,iter) = w;
                Pn = w'*Rn*w; %Noise power
                Psout = w'*Rs0*w; %Output signal power
                Piout = w'*Rsi*w; %Interference signal power
                SNR_iter0(1,iter) = Psout/Pn;
                SNR_iter(1,iter) = Psout/(Pn+Piout);
            end
            SNR_LMS(it,isnr) = real(SNR_iter(1,N_iter));
            SNRout(5,isnr) = SNRout(5,isnr) + SNR_LMS(it,isnr);
            w_lms = w_iter(:,end)/Geometry.Nant;

        end
    end
    for j=1:5
        for k=1:length(SNR)
            SNRout(j,k)= real(10*log10(SNRout(j,k)));
        end
    end

    %Print the results

    figure();    
    plot(SNR, SNRout(1,:),'o');
    hold on;
    plot(SNR, SNRout(2,:),'s');
    hold on;
    plot(SNR, SNRout(3,:),'+');
    hold on;
    plot(SNR, SNRout(4,:),'*');
    hold on;
    plot(SNR, SNRout(5,:),'p');
    hold off;
    legend('SIMPLE','NULL-STEERING', 'MVDR', 'MMSE', 'LMS');
    xlabel('SNR (Input)[dB]'); 
    ylabel('SNR (Output)[dB]'); 
    title('Adaaptive Beam-forming performance with interference');

    %Plot AF -->FIX NEEDED
    if (N_interf == 1)
    % teta_corr = -pi/2:pi/512:pi/2;
    % [vc]=steervector(Geometry.tetadir,Geometry.Nant,Pars.lambda*0.5,Pars.fc);
    % AF_simple = w_simple(length(Geometry.Nant),:)*vc.';
    % AF_null=w_null(length(Geometry.Nant),:)*vc.';
    % AF_MMSE=w_mmse(length(Geometry.Nant),:)*vc.';
    % AF_LMS=w_lms(length(Geometry.Nant),:)*vc.';
    % 
    % % figure();
    % plot(Geometry.tetadir*180/pi,10*log10(abs(AF_simple)));
    % hold on;
    % plot(Geometry.tetadir*180/pi,10*log10(abs(AF_null)));
    % hold on;
    % plot(Geometry.tetadir*180/pi,10*log10(abs(AF_MMSE)));
    % hold on;
    % plot(Geometry.tetadir*180/pi,10*log10(abs(AF_LMS)));
    % hold off;
    % 
    % xlabel('\theta');
    % ylabel('AF [dB]');
    end
end



