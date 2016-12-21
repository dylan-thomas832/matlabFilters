%% Tricycle Estimation Driver
% *Author: Dylan Thomas*
%
% This script tests the nonlinear batch filters using the tricycle problem.
% The filters it tests are (EKF,iEKF,ESRIF,iESRIF,UKF,PF,RegPF).

%% Housekeeping
clearvars; close all; clc;

% Custom scripts for keeping organization
asv;
addPaths;

%% User defined variables

% Filter to test (EKF/iEKF/ESRIF/iESRIF/UKF/PF/RegPF/All)
filter = 'RegPF';
% Initial state estimate supplied to filter (0/1)
kInit = 1;
% Number of Runge Kutta iterations for dynamics model conversion (5-100)
nRK = 10;
% Number of measurement update iterations (1-100)
Niter = 5;
% Lower limit on Gauss-Newton search in measurement updates
alphaLim = 0.001;
% Number of particles to generate in PF (100 - 100000)
Np = 500;
% Number of minimum effective particles (10% - 50% of Np)
NT = 200;
% Flag on resample scheme (0-none, 1-resample1, 2-resample2)
resample = 1;

%% Filter setup
% Get problem information

% Load time & measurement histories
cart_EKF_meas();
% Continuous dynamics & Discrete measurements
ffunc = 'f_cart';
hfunc = 'h_cart';
modelType = 'CD';
% No controls
uhist = [];

%% Constants, Inputs, Parameters, and Initialization

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
R = diag([sig_rhoa^2 sig_rhob^2]);
Q = diag([qtilsteer qtilspeed])/(thist(2)-thist(1));

% Initialization
[xguess,Pguess] = cartInit(kInit,zhist,thist);

%% Input Structure Setup
% Create a structure to input into the filters
inputStruct.fmodel = ffunc;
inputStruct.hmodel = hfunc;
inputStruct.modelFlag = modelType;
inputStruct.kInit = kInit;
inputStruct.xhatInit = xguess;
inputStruct.PInit = Pguess;
inputStruct.uhist = uhist;
inputStruct.zhist = zhist;
inputStruct.thist = thist;
inputStruct.Q = Q;
inputStruct.R = R;
%% Filter(s) Implementation

switch filter
    case 'EKF'
        inputStruct.optArgs = {nRK};
        filt = batch_EKF({inputStruct});
        filt = filt.doFilter();
    case 'iEKF'
        inputStruct.optArgs = {nRK,Niter,alphaLim};
        filt = batch_iEKF({inputStruct});
        filt = filt.doFilter();
    case 'ESRIF'
        inputStruct.optArgs = {nRK};
        filt = batch_ESRIF({inputStruct});
        filt = filt.doFilter();
    case 'iESRIF'
        % fix
        inputStruct.optArgs = {nRK};
        filt = batch_iESRIF({inputStruct});
        filt = filt.doFilter();
    case 'UKF'
        % Uses optimal tuning parameters
        inputStruct.optArgs = {nRK};
        filt = batch_UKF({inputStruct});
        filt = filt.doFilter();
    case 'PF'
        inputStruct.optArgs = {nRK,Np,resample};
        filt = batch_PF({inputStruct});
        filt = filt.doFilter();
    case 'RegPF'
        inputStruct.optArgs = {nRK,Np,NT,resample};
        filt = batch_RegPF({inputStruct});
        filt = filt.doFilter();
    case 'All'
        error('Comparison of all filters not yet implemented')
    otherwise
        filterIDs = 'EKF | iEKF | ESRIF | iESRIF | UKF | PF | RegPF | All';
        error('test:filterID','Filter type is incorrect. Choose an available filter option to test: \n\n%s',filterIDs)
end

%% Results
hold on
plot(filt.xhathist(2,(kInit+1:end)),filt.xhathist(3,(kInit+1:end)))
grid on
xlim([-3 2])
ylim([1 6])

switch filter
    case 'PF'
        break
    case 'RegPF'
        break
    otherwise
        mean_eta = mean(filt.eta_nuhist(kInit+2:end))
        Nk = length(thist)-kInit;
        r1 = chi2inv(0.05/2,size(zhist,2)*Nk)/Nk
        r2 = chi2inv(1-0.05/2,size(zhist,2)*Nk)/Nk
end