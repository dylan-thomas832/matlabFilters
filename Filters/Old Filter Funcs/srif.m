function [xhat,P] = srif(xhat0,P0,uhist,zhist,Q,R,F,Gamma,G,H)
% Linear Square Root Information Filter

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
% NOTE: This implementation uses the SRIF technique.
%
% The ESRIF starts from the a posteriori estimate & covariance at sample 0, 
%   xhat0 and P0. It transforms them into the square root information form.
%   It performs dynamic propagation and measurement update for samples 
%   k = 1 to k = kmax in the SRIF framework:
%
%   1) Takes in the SRIF a posteriori estimate at sample k.
%
%   2) Dynamically propagating it to sample k+1 to get the SRIF a priori
%      estimate.
%
%   3) Updates the k+1 SRIF estimate with the measurement model to obatin
%      the SRIF a posteriori estimate at sample k+1.
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
nv = size(Q,1);
kmax = size(zhist,1);
xhat = zeros(kmax+1,nx);
P = zeros(nx,nx,kmax+1);

% Determine the square-root information matrix for the process noise, and
% transform the measurements to have an error with an identity covariance.
Rwwk = inv(chol(Q)');
Ra = chol(R);
Rainvtr = inv(Ra');
Ha = Rainvtr*H;
zahist = zhist*(Rainvtr');

% Initialize quantities for use in the main loop and store the first a
% posteriori estimate and its error covariance matrix.
Rxx0 = inv(chol(P0)');
zx0 = Rxx0*xhat0;
Rxxk = Rxx0;
zxk = zx0;
xhat(1,:) = xhat0';
P(:,:,1) = P0;

% Compute inv(F) and inv(F)*Gamma for later use
Finv = inv(F);
FinvGamma = Finv*Gamma;

% Iterate through kmax samples, first doing dynamic propagation, then doing
% the measurement update
for k = 0:(kmax-1)
   
   kp1 = k + 1;
   % Dynamic propagation
   Rbig = [             Rwwk, zeros(nv,nx); ...
           (-Rxxk*FinvGamma),   Rxxk*Finv];
       
   [Taktr,Rdum] = qr(Rbig);
   Tak = Taktr';
   zdum = Tak*[zeros(nv,1);zxk+Rxxk*Finv*G*uhist(kp1,:)'];
   idumxvec = [(nv+1):(nv+nx)]';
   Rbarxxkp1 = Rdum(idumxvec,idumxvec);
   zbarxkp1 = zdum(idumxvec,1);
   
   % Measurement update at sample k+1
   [Tbkp1tr,Rdum] = qr([Rbarxxkp1;Ha]);
   Tbkp1 = Tbkp1tr';
   zdum = Tbkp1*[zbarxkp1;zahist(kp1,:)'];
   idumxvec = [1:nx]';
   Rxxkp1 = Rdum(idumxvec,idumxvec);
   zxkp1 = zdum(idumxvec,1);
   
   % Compute and store the state estimate and its estimation error
   % covariance at sample k + 1
   Rxxkp1inv = inv(Rxxkp1);
   xhatkp1 = Rxxkp1inv*zxkp1;
   Pkp1 = Rxxkp1inv*(Rxxkp1inv');
   kp2 = kp1 + 1;
   xhat(kp2,:) = xhatkp1';
   P(:,:,kp2) = Pkp1;
   
   % Prepare for next sample
   zxk = zxkp1;
   Rxxk = Rxxkp1;
end



