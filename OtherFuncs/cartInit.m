function [x0,P0] = cartInit(kInit,zhist,thist)
% Tricycle problem initialization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Computes initial state estimate and covariance for the tricycle problem.
%
% This function uses the first two measurements to generate an initial
% guess. This assumes the first two measurements have no error, and assumes
% the cart has a positive north position. Can be dangerous if the sample
% times are too close together. See Problem Set 3 - Problem 5 for details.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Inputs:
%
%   zhist           = [z(1)';z(2)';z(3)';...;z(kmax)'], the (kmax)x(nz) 
%                   array that stores the measurement time history. Note 
%                   that the state estimate xhat(k+1,:)' and the 
%                   measurement vector zhist(k,:)' correspond to the same 
%                   time.
%
%   thist           Vector of the discrete time history.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outputs:
%
%   xhat0           The (nx)x1 initial state estimate.
%
%   P0              The (nx)x(nx) symmetric, positive definite initial 
%                   state estimation error covariance matrix.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% Get first two measurements
rhoa1 = zhist(1,1);
rhob1 = zhist(1,2);
rhoa2 = zhist(2,1);
rhob2 = zhist(2,2);

%%%% a
y1r = (lradarb^2-lradara^2 + rhoa1^2-rhob1^2)/(2*(lradarb - lradara));
y2r = sqrt(rhob1^2 - (y1r-lradarb)^2);
y1r(2) = (lradarb^2-lradara^2 + rhoa2^2-rhob2^2)/(2*(lradarb - lradara));
y2r(2) = sqrt(rhob2^2 - (y1r(2)-lradarb)^2);

%%%% b
vr = [(y1r(2)-y1r(1))/(thist(2)-thist(1));(y2r(2)-y2r(1))/(thist(2)-thist(1))];

%%%% c
if kInit == 0
    yr0 = [y1r(1);y2r(1)]-vr*thist(1);
elseif kInit == 1
    yr0 = [y1r(1);y2r(1)];
else
    error('kInit should only be 0 or 1 here')
end

%%%% d
psi0 = atan2(vr(2),vr(1));
vr0 = norm(vr);

%%%% e
psidot0 = 0;

% Initial estimate
x0 = [psi0; yr0; psidot0; vr0];

% Initial covariance
P0 = zeros(5,5);
P0(1,1) = (2*sig_rhoa*sig_rhob)/(vr0^2*(thist(2)-thist(1))^2);
P0(2,2) = sig_rhoa*sig_rhob;
P0(3,3) = sig_rhoa*sig_rhob;
P0(4,4) = (qtilsteer*tausteer)*0.5;
P0(5,5) = (2*sig_rhoa*sig_rhob)/(thist(2)-thist(1))^2;
end