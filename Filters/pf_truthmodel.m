function [xkhist,vkhist,zktrue,zkhist] = pf_truthmodel(xhat0,P0,Qk,Rkp1,uhist,fmodel,hmodel,kmax)
% Particle Filter Truth-Model Simulation
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  This function performs a truth-model simulation for the discrete-time 
%  system model:
%
%       x(k+1) = fmodel[k,x(k),u(k),v(k)]
%       z(k)   = hmodel[k,x(k)] + w(k)
%
%  Where v(k) and w(k) are uncorrelated, zero-mean, white-noise
%  Gaussian random processes with covariances E[v(k)*v(k)'] = Q and
%  E[w(k)*w(k)'] = R.  The simulation starts from a true x(0)
%  that is drawn from the Gaussian distribution with mean xhat0
%  and covariance P0.  The simulation starts at time k = 0 and
%  lasts until time k = kmax.
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Inputs:
%
%   fmodel          Name of the nonlinear dynamics function defined by the
%                   user. It takes the time, state estimate, control
%                   vector, and process noise at sample k.
%
%   hmodel          Name of the nonlinear measurement function defined by
%                   the user. It takes the state estimate at sample k.
%
%   Q               The (nv)x(nv) symmetric positive definite process noise 
%                   covariance matrix.
%
%   R               The (nz)x(nz) symmetric positive definite measurement 
%                   noise covariance matrix.
%
%   xhat0           The (nx)x1 initial state estimate.
%
%   P0              The (nx)x(nx) symmetric positive definite initial state
%                   estimation error covariance matrix.
%
%   uhist           The (kmax)x(nu) array that stores the control time 
%                   history. Note that the state estimate xhat(k+1) and the 
%                   control vector uhist(k) correspond to the same time.
%
%   kmax            The maximum discrete-time index of the simulation.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Outputs:
%
%   xkhist          The (nx)x(kmax+1) array that gives the truth state time
%                   history. xkhist(k+1,:) gives the state at sample k.
%
%   vkhist          The (nv)x(kmax) array that gives the truth process 
%                   noise time history which affects the state transition
%                   from sample k to sample k+1.
%
%   zktrue          The (nz)x(kmax) array that gives the measurement time 
%                   history if there were no noise.
%
%   zkhist          The (nz)x(kmax) array that gives the noisy measurement
%                   time history.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
%  Get problem dimensions.
%
nx = size(xhat0,1);
nv = size(Qk,1);
nz = size(Rkp1,1);
%
%  Calculate the appropriate matrix square root of P0 for use in
%  randomly generating an initial state.
%
sqrtP0 = chol(P0)';
%
%  Calculate the appropriate matrix square roots of Qk and Rkp1 for use in
%  randomly generating the process and measurement noise vectors.
%
sqrtQ = chol(Qk)';
sqrtR = chol(Rkp1)';
%
%  Compute vkhist and set up output arrays xkhist, zkhist, zktrue. vkhist
%  is initialized as the process noise time history. zkhist is initialized
%  as just measurement noise.
%
xkhist = zeros(nx,(kmax+1));
vkhist = sqrtQ*randn(nv,kmax);
zktrue = zeros(nz,kmax);
zkhist = sqrtR*randn(nz,kmax);
%
%  Generate the initial truth state with the aid of a random number
%  generator and store it.
%
x0 = xhat0 + sqrtP0*randn(nx,1);
xkhist(:,1) = x0;
%
%  Iterate through the kmax samples, propagating the state with
%  randomly generated process noise of the correct covariance and
%  corrupting the measurements with randomly generated measurement
%  noise of the correct covariance.
%
for k = 1:kmax
    % Define previous and next sample times as well as appropriate state,
    % control, and process noise vector for the current sample.
    km1 = k - 1;
    kp1 = k + 1;
    xkm1 = xkhist(:,k);
    vkm1 = vkhist(:,k);
    ukm1 = uhist(k,:)';
    % Propogate the state to the next sample
    [xkhist(:,kp1),~,~] = fmodel(xkm1,ukm1,vkm1,km1);
    % Determine the noise-less measurements (AKA truth)
    [zktrue(:,k),~] = hmodel(xkhist(:,kp1),k);
    % Determine the noisy meaurements
    zkhist(:,k) = zktrue(:,k) + zkhist(:,k);
end
end