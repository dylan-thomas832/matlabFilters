function [xhat,P,eta_nu] = ekf(xhat0,P0,uhist,zhist,thist,Q,R,f,h,modelFlag,varargin)
% Extended Kalman Filter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function performs Extended Kalman Filter state estimation for the 
% nonlinear continuous-time system model:
%
%       xdot(t) = f[t,x(t),u(t),v(t)]
%       z(t)    = h[t,x(t)] + w(t)
% 
% The function converts the continuous-time model to discrete-time
% difference equations using a fourth-order Runge-Kutta scheme:
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
% 
% The linearized gain matrices are as follows:
%
%       Fk      = dfmodel/dxk   @[k,xhat(k),u(k),0]
%       Gammak  = dfmodel/dvk   @[k,xhat(k),u(k),0]
%       Hk      = dhmodel/dxkp1 @[k+1,xbar(k+1)]
%
% The EKF starts from the a posteriori estimate & covariance at sample 0, 
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
%   f               Name of the nonlinear dynamics function defined by the
%                   user. It takes the time, state estimate, control
%                   vector, and process noise at sample k.
%
%   h               Name of the nonlinear measurement function defined by
%                   the user. It takes the state estimate at sample k.
% 
%   Q               The (nv)x(nv) symmetric, positive definite process
%                   noise covariance matrix.
% 
%   R               The (nz)x(nz) symmetric, positive definite measurement 
%                   noise covariance matrix.
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
%   thist           Vector of the discrete time history.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Optional Inputs:
%
%   nRK             Number of Runge-Kutta iterations (Default 100).
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

% Check number of inputs, assign defaults if not supplied with input.
nRK = argumentCheck(varargin);

% Get problem dimensions and setup the output arrays
nx = size(xhat0,1);
nv = size(Q,1);
kmax = size(zhist,1);
xhat = zeros(kmax+1,nx);
P = zeros(nx,nx,kmax+1);
eta_nu = zeros(size(thist));

% Initialize quantities for use in the main loop and store the first a
% posteriori estimate and its error covariance matrix.
xhatk = xhat0;
Pk = P0;
xhat(1,:) = xhatk';
P(:,:,1) = Pk;
tk = 0;
vk = zeros(nv,1);

% Iterate through kmax samples, first doing dynamic propagation, then doing
% the measurement update
for k = 0:(kmax-1)
   
   % Propagation from sample k to sample k+1
   kp1=k+1;
   tkp1 = thist(kp1);
   if strcmp(modelFlag,'CD')
       [xbarkp1,F,Gamma] = c2dnonlinear(xhatk,uhist(kp1,:)',vk,tk,tkp1,nRK,f,1);
   elseif strcmp(modelFlag,'DD')
       [xbarkp1,F,Gamma] = feval(f,xhatk,uhist(kp1,:)',vk,k);
   else
       error('Incorrect flag for the dynamics-measurement models')
   end
   Pbarkp1 = F*Pk*(F') + Gamma*Q*(Gamma');
   
   % Measurement update at sample k+1
   [zbarkp1,H] = feval(h,xbarkp1,1);
   zkp1 = zhist(kp1,:)';
   nukp1 = zkp1 - zbarkp1;
   Skp1 = H*Pbarkp1*(H') + R;
   Wkp1 = (Pbarkp1*(H'))/Skp1;
   xhatkp1 = xbarkp1 + Wkp1*nukp1;
   Pkp1 = Pbarkp1 - Wkp1*Skp1*(Wkp1');
   eta_nukp1 = nukp1'*inv(Skp1)*nukp1;
   
   % Store results
   kp2 = kp1 + 1;
   xhat(kp2,:) = xhatkp1';
   P(:,:,kp2) = Pkp1;
   eta_nu(kp1) = eta_nukp1;
   
   % Prepare for next sample
   xhatk = xhatkp1;
   Pk = Pkp1;
   tk = tkp1;
end

end


function nRK = argumentCheck(args)

switch nargin
    case 0
        nRK = 100;
    case 1
        nRK = args{1};
    otherwise 
        error('Not enough input arguments')
end
% Check optional inputs are sensible, error otherwise
if nRK < 5
    error('Number of Runge-Kutta iterations should be larger than 5')
end

end

