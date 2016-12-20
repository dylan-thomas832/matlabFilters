%%% AOE 6984 - Model Based Estimation - Final Exam
% Assignment 8 - Problem 4
%
% Author: Dylan Thomas
%
% This script tests an Extended Kalman Filter.

% Housekeeping
clearvars; close all; clc;

% Custom scripts for keeping organization
asv;
addPaths;
format long

%%% Implement Extended Kalman Filter

% Define dynamics function
ffunc = 'f_ps8prob4';
% Define measurement model function
hfunc = 'h_ps8prob4';
% Load data to be filtered
load measdata_pfexample

% % Determine number of samples and states, and setup arrays to store results
% nx = size(xhat0,1);
% K = size(zkhist,1);
% Kp1 = K + 1;
% xhathist = zeros(Kp1,nx);
% Phist = zeros(nx,nx,Kp1);
% 
% % Store initial estimate and its error covariance
% xhathist(1,:) = xhat0;
% Phist(:,:,1) = P0;
% % Initialize the state estimate and its covariance for the filter 
% xhatkp1 = xhat0;
% Pkp1 = P0;
% uk = 0;
% Qk = Q;
% Rkp1 = R;
% 
% % Main loop which executes EKF 
% for kk = 0:(K-1)
%     xhatk = xhatkp1;
%     Pk = Pkp1;
%     kkp1 = kk + 1;
%     zkp1 = zkhist(kkp1,:)';
%     [xhatkp1,Pkp1,~] = ekf_1step(xhatk,Pk,kk,uk,zkp1,Qk,Rkp1,fmodel,hmodel);
%     kkp2 = kkp1 + 1;
%     xhathist(kkp2,:) = xhatkp1';
%     Phist(:,:,kkp2) = Pkp1;
% end

%%% Implement Particle Filter
Np = 50;
nRK = 20;
uhist = zeros(size(zkhist));
thist = uhist;
% [xhatkhist_pf,Pkhist_pf] = pf(xhat0,P0,uhist,zkhist,Qk,Rkp1,fmodel,hmodel,Np);

pf = batch_RegPF(ffunc,hfunc,'DD',xhat0,P0,uhist,zkhist,thist,Q,R,nRK,Np,50,1);
pf = pf.doFilter();

%% Results
hold on
plot(pf.xhathist,'b.')
grid on
axis tight
