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
filter = 'ESRIF';
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
uk = [];
% No time-history
tk = [];

%% Input Structure Setup
% Create a structure to input into the filters
inputStruct.fmodel = ffunc;
inputStruct.hmodel = hfunc;
inputStruct.modelFlag = modelType;
inputStruct.k = 0;
inputStruct.xhatk = xhat0;
inputStruct.Pk = P0;
inputStruct.uk = uk;
inputStruct.zkp1 = zkhist(1,:)';
inputStruct.tk = tk;
inputStruct.dt = tk;
inputStruct.Qk = Q;
inputStruct.Rkp1 = R;
inputStruct.optArgs = {nRK};

%%
% Determine number of samples and states, and setup arrays to store results
nx = size(xhat0,1);
kmax = size(zkhist,1);
kmaxp1 = kmax + 1;
xhatkhist = zeros(nx,kmaxp1);
Pkhist = zeros(nx,nx,kmaxp1);
eta_nuhist = zeros(kmax,1);

% Store initial estimate and its error covariance
xhatkhist(:,1) = xhat0;
Pkhist(:,:,1) = P0;


ekf = step_ESRIF({inputStruct});
% Main loop which executes EKF 
for k = 0:(kmax-1)
    kp1 = k + 1;
    ekf.zkp1 = zkhist(kp1,:)';
    ekf = ekf.doFilter();
    
    kp2 = kp1 + 1;
    xhatkhist(:,kp2) = ekf.xhatkp1;
    Pkhist(:,:,kp2) = ekf.Pkp1;
    eta_nuhist(kp1) = ekf.eta_nukp1;
    
    ekf.k = kp1;
    ekf.xhatk = ekf.xhatkp1;
    ekf.Pk = ekf.Pkp1;
    
end

%% Results
hold on
plot(xhatkhist,'b*')
grid on
axis tight


mean_eta = mean(eta_nuhist)
Nk = size(zkhist,1);
r1 = chi2inv(0.05/2,size(zkhist,2)*Nk)/Nk
r2 = chi2inv(1-0.05/2,size(zkhist,2)*Nk)/Nk