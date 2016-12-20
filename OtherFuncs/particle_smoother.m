function [xstar,Pstar,Wnorm] = particle_smoother(xhat0,P0,uhist,zhist,Qk,Rkp1,fmodel,hmodel,Np)
% Particle Smoother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% This function performs Particle Smoothing state estimation for the 
% nonlinear dsicrete-time system model:
%
%       x(k+1) = fmodel[k,x(k),u(k),v(k)]
%       z(k)   = hmodel[k,x(k)] + w(k)
% 
% Where v(k) and w(k) are uncorrelated, zero-mean, white-noise, Gaussian
% random processes with covariances E[vk*vk'] = Qk & E[wk*wk'] = Rk.
% The functions f and h define the dynamics and measurement equations of
% the system. They are generally nonlinear, and the EKF linearizes them
% about the a posteriori and a priori state estimates respectively.
%
% Algorithm Overview:
%
%   1) Starts from the a posteriori estimate & covariance at sample 0, 
%      and it generates Np particles at sample time 0.
%
%   2) It open-loop simulates each particle from sample 0 to sample kmax
%      creating Np particle state trajectories.
%
%   3) It calculates the measurement error at each sample time for each
%      particle trajectory.
%
%   4) It sums the squared errors over the entire trajectory to create
%      weights for each particle trajectory.
%
%   5) It calculates the smoothed state and covariance estimation
%      time-histories by weighting each particle state at each sample time.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Inputs:
%
%   xhat0           The (nx)x1 initial state estimate.
%
%   P0              The (nx)x(nx) symmetric, positive definite initial 
%                   state estimation error covariance matrix.
%
%   uhist           = [u(1)';u(2)';u(3)';...;u(kmax)'], the (kmax)x(nu) 
%                   array that stores the control time history. Note 
%                   that the state estimate xhat(k+1,:)' and the 
%                   control vector uhist(k,:)' correspond to the same time.
%
%   zhist           = [z(1)';z(2)';z(3)';...;z(kmax)'], the (kmax)x(nz) 
%                   array that stores the measurement time history. Note 
%                   that the state estimate xhat(k+1,:)' and the 
%                   measurement vector zhist(k,:)' correspond to the same 
%                   time.
%
%   Qk              The (nv)x(nv) symmetric, positive definite process
%                   noise covariance matrix.
% 
%   Rk              The (nz)x(nz) symmetric, positive definite measurement 
%                   noise covariance matrix.
%
%   fmodel          Name of the nonlinear dynamics function defined by the
%                   user. It takes the time, state estimate, control
%                   vector, and process noise at sample k.
%
%   hmodel          Name of the nonlinear measurement function defined by
%                   the user. It takes the state estimate at sample k.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Optional Inputs:
%
%   Np              Number of particles to generate and use for smoothing 
%                   by generating Np particle trajectories (Default 100).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outputs:
%
%   xstar   The (kmax+1)x(nx) array that stores the smoothed time history 
%           for the state vector. Note Matlab does not allow zero indices.
%
%   Pstar   The (nx)x(nx)x(kmax+1) array that stores the smoothed error
%           covariance time history as computed by the Particle Smoother.
%
%   Wnorm   The (Npx1) vector that stores the normalized weights.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check number of inputs, assign defaults if not supplied with input.
switch nargin
    case 8
       Np = 100;
    case 9
    otherwise 
        error('Not enough input arguments.')
end
% Check optional inputs are sensible, error otherwise
if Np < 10
    error('Number of particle trajectories is too small.')
end

% Get problem dimensions
nx = size(xhat0,1);
nz = size(Rkp1,1);
kmax = size(zhist,1);
% Allocation of array variables
xstar = zeros(nx,kmax+1);
Pstar = zeros(nx,nx,kmax+1);
X_script = zeros(nx,kmax+1,Np);
Jlog = zeros(Np,1);

% Initialize the a posteriori smoothed state estimate and error covariance.
xstar(:,1) = xhat0;
Pstar(:,:,1) = P0;

% Calculate inverse of Rkp1 before loop.
invRkp1 = inv(Rkp1);

% Loop to generate particle trajectories and measurement error sums.
for ii = 1:Np
    % Truth model simulation function
    [X_script(:,:,ii),~,ztrue,~] = pf_truthmodel(xhat0,P0,Qk,Rkp1,uhist,fmodel,hmodel,kmax);

    % Calculate the measurement error at each sample for this particle.
    ztilde = zhist' - ztrue;
    % Reinitialize sum measurement errors squared
    Jlog(ii) = 0;
    for k = 1:kmax
        % Sum the measurement errors over the entire particle trajectory.
        Jlog(ii) = Jlog(ii) + ztilde(:,k)'*invRkp1*ztilde(:,k);
    end
end
% Store the final log-weight for each particle trajectory.
Wlog = -0.5*Jlog;
% Find the max log-weight and subtract from the rest to prevent underflow.
% Normalize the weights.
Wlogmax = max(Wlog);
W = exp(Wlog - Wlogmax);
Wnorm = W/(sum(W));

% Calculate & store the smoothed state estimates.
for k = 2:kmax+1
    xstar(:,k) = squeeze(X_script(:,k,:))'*Wnorm;
end
% Calculate & store the smoothed error covariances.
for k = 1:kmax+1
    Pdum = zeros(nx,nx,Np);
    for ii=1:Np
        Pdum(:,:,ii) = (X_script(:,k,ii) - xstar(:,k))*(X_script(:,k,ii) - xstar(:,k))';
        Pdum(:,:,ii) = Pdum(:,:,ii)*Wnorm(ii);
    end
    Pstar(:,:,k) = sum(Pdum,3);
end

end

