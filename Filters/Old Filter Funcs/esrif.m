function [xhat,P,eta_nu] = esrif(xhat0,P0,uhist,zhist,thist,Q,R,f,h,modelFlag,varargin)
% Extended Square Root Information Filter

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
% The linearized gain matrices are as follows:
%
%       Fk      = dfmodel/dxk   @[k,xhat(k),u(k),0]
%       Gammak  = dfmodel/dvk   @[k,xhat(k),u(k),0]
%       Hk      = dhmodel/dxkp1 @[k+1,xbar(k+1)]
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

% Determine the square-root information matrix for the process noise, and
% transform the measurements to have an error with an identity covariance.
Rvvk = inv(chol(Q)');
Ra = chol(R);
Rainvtr = inv(Ra');
zahist = zhist*(Rainvtr');

% Initialize quantities for use in the main loop and store the first a
% posteriori estimate and its error covariance matrix.
Rxx0 = inv(chol(P0)');
xhatk = xhat0;
Rxxk = Rxx0;
xhat(1,:) = xhat0';
P(:,:,1) = P0;
tk = 0;
vk = zeros(nv,1);

% Iterate through kmax samples, first doing dynamic propagation, then doing
% the measurement update
for k = 0:(kmax-1)
   
   % Propagation from sample k to sample k+1
   kp1 = k+1;
   tkp1 = thist(kp1);
   if strcmp(modelFlag,'CD')
       [xbarkp1,F,Gamma] = c2dnonlinear(xhatk,uhist(kp1,:)',vk,tk,tkp1,nRK,f,1);
   elseif strcmp(modelFlag,'DD')
       [xbarkp1,F,Gamma] = feval(f,xhatk,uhist(kp1,:)',vk,k);
   else
       error('Incorrect flag for the dynamics-measurement models')
   end
   Finv = inv(F);
   FinvGamma = F\Gamma;
   % QR Factorize
   Rbig = [             Rvvk, zeros(nv,nx); ...
           (-Rxxk*FinvGamma),      Rxxk/F];
   [Taktr,Rdum] = qr(Rbig);
   Tak = Taktr';
   zdum = Tak*[zeros(nv,1);Rxxk*Finv*xbarkp1];
   % Retrieve SRIF terms at k+1 sample
   idumxvec = [(nv+1):(nv+nx)]';
   Rbarxxkp1 = Rdum(idumxvec,idumxvec);
   zetabarxkp1 = zdum(idumxvec,1);

   % Measurement update at sample k+1
   % Linearize measurement function about xbar(k+1)
   [zbarkp1,H] = feval(h,xbarkp1,1);
   % Transform ith H(k) matrix and non-homogeneous measurement terms
   Ha = Rainvtr*H;
   zEKF = zahist(kp1,:)' - Rainvtr*zbarkp1 + Ha*xbarkp1;
   % QR Factorize
   [Tbkp1tr,Rdum] = qr([Rbarxxkp1;Ha]);
   Tbkp1 = Tbkp1tr';
   zdum = Tbkp1*[zetabarxkp1;zEKF];
   % Retrieve k+1 SRIF terms
   idumxvec = [1:nx]';
   Rxxkp1 = Rdum(idumxvec,idumxvec);
   zetaxkp1 = zdum(idumxvec,1);
   zetarkp1 = zdum(nx+1:end);
   
   % Compute and store the state estimate and covariance at sample k + 1
   Rxxkp1inv = inv(Rxxkp1);
   xhatkp1 = Rxxkp1\zetaxkp1;
   Pkp1 = Rxxkp1inv*(Rxxkp1inv');
   kp2 = kp1 + 1;
   xhat(kp2,:) = xhatkp1';
   P(:,:,kp2) = Pkp1;
   eta_nu(kp1) = zetarkp1'*zetarkp1;
   % Prepare for next sample
   Rxxk = Rxxkp1;
   xhatk = xhatkp1;
   tk = tkp1;
end

end

function nRK = argumentCheck(args)

switch nargin
    case 10
        nRK = 100;
    case 11
        nRK = args{1};
    otherwise 
        error('Not enough input arguments')
end
% Check optional inputs are sensible, error otherwise
if nRK < 5
    error('Number of Runge-Kutta iterations should be larger than 5')
end

end