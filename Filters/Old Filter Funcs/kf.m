function [xhat,P] = kf(xhat0,P0,uhist,zhist,Q,R,F,Gamma,G,H)
% Linear Kalman Filter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function performs Kalman Filter state estimation for the LTI
% discrete-time system model:
%
%       x(k+1) = F*x(k) + G*u(k) + Gamma*v(k)
%       z(k)   = H*x(k) + w(k)
% 
% Where v(k) and w(k) are uncorrelated, zero-mean, white-noise, Gaussian
% random processes with covariances E[vk*vk'] = Qk & E[wk*wk'] = Rk.
% The Kalman Filter starts from the a posteriori estimate and its
% covariance at sample 0, xhat0 and P0, and it performs dynamic propagation
% and measurement update for samples k = 1 to k = kmax.
%
% The KF starts from the a posteriori estimate & covariance at sample 0, 
%   xhat0 and P0, and it performs dynamic propagation and measurement 
%   update for samples k = 1 to k = kmax:
%
%   1) Takes in the a posteriori estimate at sample k.
%
%   2) Dynamically propagating it to sample k+1 to get the apriori
%      estimate.
%
%   3) Updates the k+1 state estimate with the measurement model to obatin
%      the a posteriori state estimate at sample k+1 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Inputs:
%
%   F       The (nx)x(nx) state tranisiton matrix for the time-invariant
%           linear system.
% 
%   Gamma   The (nx)x(nv) process noise gain matrix for the
%           time-invariant linear system.
%
%   G       the (nx)x(nu) input gain matrix for the time-invariant linear
%           system.
% 
%   H       The (nz)x(nx) output gain matrix for the time-invariant linear
%           system.
% 
%   Q       The (nv)x(nv) symmetric, positive definite process noise
%           covariance matrix.
% 
%   R       The (nz)x(nz) symmetric, positive definite measurement noise
%           covariance matrix.
%
%   xhat0   The (nx)x1 initial state estimate.
%
%   P0      The (nx)x(nx) symmetric, positive definite initial state
%           estimation error covariance matrix.
%
%   uhist   = [u(1)';z(2)';...;u(kmax)'], the (kmax),(nu) array that stores
%           the control input time history.
%
%   zhist   = [z(1)';z(2)';z(3)';...;z(kmax)'], the (kmax)x(nz) array that
%           stores the measurement time history. Note that the state
%           estimate xhat(k+1,:)' and the measurement vector zhist(k,:)'
%           correspond to the same time.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outputs:
%
%   xhat    = [xhat(0)';xhat(1)';xhat(2)';...;xhat(kmax)'], the
%           (kmax+1)x(nx) array that stores the estimated time history for
%           the state vector. Note Matlab does not allow zero indices.
%
%   P       The (nx)x(nx)x(kmax+1) array that stores the estimation error
%           covariance time history as computed by the Kalman Filter. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get problem dimensions and setup the output arrays
nx = size(xhat0,1);
kmax = size(zhist,1);
kmaxp1 = kmax + 1;
xhat = zeros(kmaxp1,nx);
P = zeros(nx,nx,kmaxp1);

% Initialize quantities for use in the main loop and store the first a
% posteriori estimate and its error covariance matrix.
xhatk = xhat0;
Pk = P0;
xhat(1,:) = xhatk';
P(:,:,1) = Pk;

% Iterate through kmax samples, first doing dynamic propagation, then doing
% the measurement update
for k = 0:(kmax-1)
   
   kp1 = k + 1;
   % Propagation from sample k to sample k+1
   xbarkp1 = F*xhatk + G*uhist(kp1,:)';
   Pbarkp1 = F*Pk*(F') + Gamma*Q*(Gamma');
   
   % Measurement update at sample k+1
   zbarkp1 = H*xbarkp1;
   zkp1 = zhist(kp1,:)';
   nukp1 = zkp1 - zbarkp1;
   Skp1 = H*Pbarkp1*(H') + R;
   Wkp1 = (Pbarkp1*(H'))/Skp1;
   xhatkp1 = xbarkp1 + Wkp1*nukp1;
   Pkp1 = Pbarkp1 - Wkp1*Skp1*(Wkp1');
   
   % Store results
   kp2 = kp1 + 1;
   xhat(kp2,:) = xhatkp1';
   P(:,:,kp2) = Pkp1;
   
   % Prepare for next sample
   xhatk = xhatkp1;
   Pk = Pkp1;
end



