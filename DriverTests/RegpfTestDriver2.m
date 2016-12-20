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

%%% Implement Particle Filter
Np = 500;
nRK = 10;
uhist = zeros(size(zkhist));
thist = uhist;

kInit = 0;
pf = batch_RegPF(ffunc,hfunc,'DD',kInit,xhat0,P0,uhist,zkhist,thist,Q,R,nRK,Np,200,1);
pf = pf.doFilter();

%% Results
hold on
plot(pf.xhathist,'b.')
grid on
axis tight
