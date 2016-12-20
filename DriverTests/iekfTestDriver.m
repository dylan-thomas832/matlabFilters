%% EKF Tricycle Estimation

%% Housekeeping
clearvars; close all; clc;

% Custom scripts for keeping organization
asv;
addPathsMBE;

%% Constants, Inputs, Parameters, and Initialization

% Load time & measurement histories
cart_EKF_meas();

% East positions of radars
lradara = -1; % meters
lradarb = 1; % meters

% Measurement noise standard deviations
sig_rhoa = 0.002; % meters
sig_rhob = 0.002; % meters

% Continuous time process noise intensities
qtilsteer = 0.10; % rad^2/s
qtilspeed = 21.25; % m^2/s^3

% System parameters
b = 0.1; % meters
tausteer = 0.25; % sec
tauspeed = 0.60; % sec
meanspeed = 2.1; % m/s

% Calculate measurement & process noise discrete covariances
Rk = diag([sig_rhoa^2 sig_rhob^2]);
Qk = diag([qtilsteer qtilspeed])/(thist(2)-thist(1));

% Initialization
[xguess,Pguess] = cartInit0(zhist,thist);

%% Filter setup

% Continuous dynamics
ffunc = 'f_cart';
hfunc = 'h_cart';

% Runge-Kutta iterations
nRK = 20;

% Control vector
uk = zeros(length(thist),1);

iekf = batch_iEKF(ffunc,hfunc,'CD',xguess,Pguess,uk,zhist,thist,Qk,Rk,nRK,3);
iekf = iekf.doFilter();

%% Results
hold on
plot(iekf.xhathist(2,:),iekf.xhathist(3,:))
grid on
xlim([-3 2])
ylim([1 6])

mean_eta = mean(iekf.eta_nuhist)
Nk = length(thist);
r1 = chi2inv(0.05/2,size(zhist,2)*Nk)/Nk
r2 = chi2inv(1-0.05/2,size(zhist,2)*Nk)/Nk