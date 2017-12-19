clear all
close all
clc

% An example of parameters estimation of chirps with time-varying amplitude
% Written by: Xiangxia Meng
% Paper: "Estimation of chirp signals with time-varying amplitudes"
% Authors: Xiangxia Meng, Andreas Jakobsson, Xiukun Li, Yahui Lei

%% Signal Parameters
f0 = 0.05;
B  = 0.15;

N = 60;
k = B/N;
t = 0:N-1;
t = t(:);

s0  = exp(1i*2*pi*t*f0 + 1i*pi*t.^2*k);
amp = 0.3*sin(0.08*t + 0.1) + linspace(1,0.5,N)';
e   = randn(N,1) + 1i*randn(N,1);
s   = diag(s0*amp.') + 0.05*e;

figure('Name','Original Signal')
plot(real(s))
xlabel('Time(s)')
ylabel('Amplitude(V)')
axis([-inf inf -1.5 1.5])

%% Signal Dictionary -1
Nf  = 11;
f0d = linspace(0,0.5,Nf)';
sd  = [];
fd  = [];
for i = 1:Nf
    Bd = f0d - f0d(i);
    kd = Bd/N;
    fd = [fd,[f0d(i)*ones(1,Nf);kd']];
    sd = [sd,exp(1i*2*pi*t*f0d(i)*ones(1,Nf) + 1i*pi*t.^2*kd')];   
end
Z = sd;
clear sd

%% Spline Basis
Nk    = 3;
knots = linspace(0,2,Nk+2);
knots = knots(1:end-1);

Gamma = ones(N,1);
Slin  = linspace(0,2,N)';
for i = 1:length(knots)
    Gamma = [Gamma,max(Slin-knots(i),0)];
end
for i = 1:length(knots)
    Gamma = [Gamma,max(Slin-knots(i),0).^2];
end
for i = 1:length(knots)
    Gamma = [Gamma,max(Slin-knots(i),0).^3];
end
for i = 1:length(knots)
    Gamma = [Gamma,max(Slin-knots(i),0).^4];
end

%% ADMM
alpha   = 6;
beta    = 4;
rho     = 0.5;
S = Est_Chirps (s,Z,Gamma,alpha,beta,rho);

S1 = sqrt(sum(S.* conj(S),1));
[~,Gin] = max(S1);

%% ADMM by updating dictionary
fd1  = fd(:,Gin);
Df0d = f0d(2) - f0d(1);
DBd  = Df0d;        
for j = 1:3
    
    fdr0 = linspace(fd1(1)-2*Df0d,fd1(1)+2*Df0d,101);
    fdr1 = fd1(1) + fd1(2)*N;
    fdr  = [fdr0;(fdr1-fdr0)/N];
    Zr   = exp(1i*2*pi*t*fdr0 + 1i*pi*t.^2*(fdr1-fdr0)/N);
    
    rho     = 2;
    alpha   = 5;
    beta    = 5;
    S = Est_Chirps (s,Zr,Gamma,alpha,beta,rho);
    
    SR = sqrt(sum(Gamma*S.* conj(Gamma*S),1));
    [~,Ginr] = max(SR);
    fd1= fdr(:,Ginr);
    Z1 = Zr(:,Ginr);
    S1 = S(:,Ginr);
    
    Zs = Z1;
    for i = 1:size(Gamma,2)
        ZG(:,i) = Zs.*Gamma(:,i);
    end
    S1s = inv(ZG'*ZG)*ZG'*s;
end
    
figure('Name','Estimated IF')
plot(ones(size(t))*fd1(1,:)+t*fd1(2,:),'--');
hold on
plot(ones(size(t))*f0'+t*k',':');
hold off
legend('Estimated','True')
xlabel('Time(s)')
ylabel('Frequency(Hz)')
axis([-inf inf 0 0.3])

Amped1 = abs(Gamma*S1s);
figure('Name','Estimated Amplitude')
plot(Amped1,'--');
hold on
plot(amp);
hold off
legend('Estimated','True')
xlabel('Time(s)')
ylabel('Amplitude(V)')
axis([-inf inf 0 2])
