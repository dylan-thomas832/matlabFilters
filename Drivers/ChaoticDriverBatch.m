%% Chaotic System Estimation Driver
% *Author: Dylan Thomas*
%
% This script tests the nonlinear batch filters using a 1D chaotic system.
% The filters it tests are (EKF,iEKF,ESRIF,iESRIF,UKF,PF,RegPF).

%% Housekeeping
clearvars; close all; clc;

% Custom scripts for keeping organization
asv;
addPaths;

%% User defined variables

% Filter to test (EKF/iEKF/ESRIF/iESRIF/UKF/PF/RegPF/All)
filter = 'EKF';
% Initial state estimate supplied to filter (0)
kInit = 0;
% Number of Runge Kutta iterations for dynamics model conversion (5-100)
nRK = 50;
% Number of measurement update iterations (1-100)
Niter = 5;
% Lower limit on Gauss-Newton search in measurement updates
alphaLim = 0.001;
% Number of particles to generate in PF (100 - 100000)
Np = 1000;
% Number of minimum effective particles (10% - 50% of Np)
NT = 800;
% Flag on resample scheme (0-none, 1-resample1, 2-resample2)
resample = 1;

%% Filter setup
% Get problem information

% Load data to be filtered
load measdata_pfexample
% Continuous dynamics & Discrete measurements
ffunc = 'f_ps8prob4';
hfunc = 'h_ps8prob4';
modelType = 'DD';
% No controls
uhist = [];
% No time-history
thist = [];

%% Input Structure Setup
% Create a structure to input into the filters
inputStruct.fmodel = ffunc;
inputStruct.hmodel = hfunc;
inputStruct.modelFlag = modelType;
inputStruct.kInit = kInit;
inputStruct.xhatInit = xhat0;
inputStruct.PInit = P0;
inputStruct.uhist = uhist;
inputStruct.zhist = zkhist;
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
plot(filt.xhathist(kInit+1:end),'b*')
grid on
axis tight

switch filter
    case 'PF'
    case 'RegPF'
    otherwise
        mean_eta = mean(filt.eta_nuhist(kInit+2:end))
        Nk = size(zkhist,1)-kInit;
        r1 = chi2inv(0.05/2,size(zkhist,2)*Nk)/Nk
        r2 = chi2inv(1-0.05/2,size(zkhist,2)*Nk)/Nk
end