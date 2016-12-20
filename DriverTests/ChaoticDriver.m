%% Chaotic System Estimation Driver
% This script tests the nonlinear batch filters using a 1D chaotic system.
% The filters it tests are (EKF,iEKF,ESRIF,iESRIF,UKF,PF,RegPF).

%% Housekeeping
clearvars; close all; clc;

% Custom scripts for keeping organization
asv;
addPaths;

%% User defined variables

% Filter to test (EKF/iEKF/ESRIF/iESRIF/UKF/PF/RegPF/All)
filter = 'RegPF';
% Initial state estimate supplied to filter (0)
kInit = 0;
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

%% Filter(s) Implementation

switch filter
    case 'EKF'
        filt = batch_EKF(ffunc,hfunc,modelType,kInit,xhat0,P0,uhist,zkhist,thist,Q,R,nRK);
        filt = filt.doFilter();
    case 'iEKF'
        filt = batch_iEKF(ffunc,hfunc,modelType,kInit,xhat0,P0,uhist,zkhist,thist,Q,R,nRK,Niter,alphaLim);
        filt = filt.doFilter();
    case 'ESRIF'
        filt = batch_ESRIF(ffunc,hfunc,modelType,kInit,xhat0,P0,uhist,zkhist,thist,Q,R,nRK);
        filt = filt.doFilter();
    case 'iESRIF'
        filt = batch_iESRIF(ffunc,hfunc,modelType,kInit,xhat0,P0,uhist,zkhist,thist,Q,R,nRK);
        filt = filt.doFilter();
    case 'UKF'
        % Uses optimal tuning parameters
        filt = batch_UKF(ffunc,hfunc,modelType,kInit,xhat0,P0,uhist,zkhist,thist,Q,R,nRK);
        filt = filt.doFilter();
    case 'PF'
        filt = batch_PF(ffunc,hfunc,modelType,kInit,xhat0,P0,uhist,zkhist,thist,Q,R,nRK,Np,resample);
        filt = filt.doFilter();
    case 'RegPF'
        filt = batch_RegPF(ffunc,hfunc,modelType,kInit,xhat0,P0,uhist,zkhist,thist,Q,R,nRK,Np,NT,resample);
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