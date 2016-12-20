function [xhat,P,eta_nu] = ukf(xhat0,P0,uhist,zhist,thist,Q,R,f,h,modelFlag,varargin)
% Unscented Kalman Filter (Sigma Points filter)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function performs Unscented Kalman Filter state estimation for the 
% nonlinear continuous-time dynamic system model:
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
% random processes with covariances E[v(k)*v(k)'] = Q & E[w(k)*w(k)'] = R.
% The functions f and h define the dynamics and measurement equations of
% the system. They are generally nonlinear, and the EKF linearizes them
% about the a posteriori and a priori state estimates respectively.
%
%
% The Unscented Kalman Filter starts from the a posteriori estimate and its
% covariance at sample 0, xhat0 and P0, and it performs dynamic propagation
% and measurement update for samples k = 1 to k = kmax.
%
% This UKF uses the sigma-points tuning parameters kappa, alpha, and beta
% from the paper:
%
%   Wan, E.A., and van der Merwe, R., "The Unscented Kalman Filter," in
%   Kalman Filtering and Neural Networks, S. Haykin, ed., Wiley, (New York,
%   2001), Chapter 7, pp. 221-280.
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
%   alpha           UKF tuning parameter. Spread of the sigma points. Wan &
%                   van der Merwe recommend 10e-5 to 1 (Default 0.01).
%
%   beta            UKF tuning parameter. When x is sample from a Gaussian
%                   distribution, the optimal value is two according to Wan
%                   & van der Merwe (Default 2).
%
%   kappa           UKF tuning parameter. Wan & van der Merwe recommend it
%                   to be three minus the number of states (Default 3-nx).
%
%   lambda          UKF scaling parameter. Wan & van der Merwe recommend an
%                   equation for this parameter (Default: see paper or eqn)
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

% Get problem dimensions
nx = size(xhat0,1);
nv = size(Q,1);
% Check the inputs, and assign them to appropriate variables.
[nRK,alpha,beta,lambda] = argumentCheck(nx,nv,varargin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup the output arrays
kmax = size(zhist,1);
sigptk = zeros(nx+nv,1+2*(nx+nv));
xsigptbark = zeros(nx,1+2*(nx+nv));
zsigptbark = zeros(nv,1+2*(nx+nv));
xhat = zeros(kmax+1,nx);
P = zeros(nx,nx,kmax+1);
eta_nu = zeros(size(thist));

% Initialize quantities for use in the main loop and store the first a
% posteriori estimate and its error covariance matrix.
xhatk = xhat0;
xhat(1,:) = xhat0';
P(:,:,1) = P0;
Svk = chol(Q)';
Sxk = chol(P0)';

% ???
tk = 0;

% Filter weights for mean and covariances
W_m0 = lambda/(nx+nv+lambda);
W_mi = 1/(2*(nx+nv+lambda));

W_c0 = W_m0 + (1-alpha^2+beta);
W_ci = W_mi;

% Iterate through kmax samples, first doing dynamic propagation, then doing
% the measurement update
for k = 0:(kmax-1)
    
    kp1 = k + 1;
    tkp1 = thist(kp1);
    % Sigma point generation loop
    for ii = 1:((2*nx+2*nv)+1)
        % Sigma point calculations
        if ii == 1
            sigptk(:,ii) = [xhatk ; zeros(nv,1)];
        elseif ii > 1 && ii <= (nx + 1)
            sigptk(:,ii) = [xhatk + sqrt(nx+nv+lambda)*Sxk(:,ii-1) ; zeros(nv,1)];
        elseif ii > (nx + 1) && ii <= (2*nx + 1)
            sigptk(:,ii) = [xhatk - sqrt(nx+nv+lambda)*Sxk(:,(ii-1-nx)) ; zeros(nv,1)];
        elseif ii > (2*nx + 1) && ii <= (2*nx + nv + 1)
            sigptk(:,ii) = [xhatk; sqrt(nx+nv+lambda)*Svk(:,(ii-1-2*nx))];
        elseif ii > (2*nx + nv + 1) && ii <= (2*nx + 2*nv + 1)
            sigptk(:,ii) = [xhatk; -sqrt(nx+nv+lambda)*Svk(:,(ii-1-2*nx-nv))];
        else
            error('out of sigma points bounds')
        end
        
        % Propogate sigma points through model dynamics
        if strcmp(modelFlag,'CD')
            [xsigptbark(:,ii),~,~] = c2dnonlinear(sigptk((1:nx),ii),uhist(kp1,:)',sigptk((nx+1):(nx+nv),ii),tk,tkp1,nRK,f,0);
        elseif strcmp(modelFlag,'DD')
            [xsigptbark(:,ii),~,~] = feval(f,sigptk((1:nx),ii),uhist(kp1,:)',sigptk((nx+1):(nx+nv),ii),k);
        else
            error('Incorrect flag for the dynamics-measurement models')
        end
        % Propogate measurement model with a priori sigma points
        [zsigptbark(:,ii),~] = feval(h,xsigptbark(:,ii),0);
        
        % Running summation for state and measurement means
        if ii == 1
            xbarkp1 = W_m0*xsigptbark(:,ii);
            zbarkp1 = W_m0*zsigptbark(:,ii);
        else
            xbarkp1 = xbarkp1 + W_mi*xsigptbark(:,ii);
            zbarkp1 = zbarkp1 + W_mi*zsigptbark(:,ii);
        end
    end
    
    
    Pbarkp1 = W_c0*(xsigptbark(:,1)-xbarkp1)*(xsigptbark(:,1)-xbarkp1)';
    Pxzkp1 = W_c0*(xsigptbark(:,1)-xbarkp1)*(zsigptbark(:,1)-zbarkp1)';
    Pzzkp1 = W_c0*(zsigptbark(:,1)-zbarkp1)*(zsigptbark(:,1)-zbarkp1)' + R;
    % Covariance running summation loop
    for ii = 2:((2*nx+2*nv)+1)
        Pbarkp1 = Pbarkp1 + W_ci*(xsigptbark(:,ii)-xbarkp1)*(xsigptbark(:,ii)-xbarkp1)';
        Pxzkp1 = Pxzkp1 + W_ci*(xsigptbark(:,ii)-xbarkp1)*(zsigptbark(:,ii)-zbarkp1)';
        Pzzkp1 = Pzzkp1 + W_ci*(zsigptbark(:,ii)-zbarkp1)*(zsigptbark(:,ii)-zbarkp1)';
    end
    
    % a posteriori estimate and covariance from LMMSE eqns
    zkp1 = zhist(kp1,:)';
    nukp1 = zkp1-zbarkp1;
    xhatkp1 = xbarkp1 + Pxzkp1*inv(Pzzkp1)*(zkp1-zbarkp1);
    Pkp1 = Pbarkp1 - Pxzkp1*inv(Pzzkp1)*Pxzkp1';
    eta_nukp1 = nukp1'*inv(Pzzkp1)*nukp1;
    
    % Store results
    kp2 = kp1 + 1;
    xhat(kp2,:) = xhatkp1';
    P(:,:,kp2) = Pkp1;
    eta_nu(kp1) = eta_nukp1;
    
    % Prepare for next sample
    xhatk = xhatkp1;
    tk = tkp1;
    Sxk = chol(Pkp1)';
end
end

% Check number of inputs, assign defaults if not supplied with input.
function [nRK,alpha,beta,lambda] = argumentCheck(nx,nv,args)
% Switch case on number of optional arguments
switch length(args)
    case 0
        nRK = 100;
        % Scaling parameter
        alpha = 0.01;
        % Optimal for Gaussian distribution samples
        beta = 2;
        kappa = 3-nx;
        % UKF scaling parameter.
        lambda = alpha^2*(nx+nv+kappa) - (nx+nv);
    case 1
        nRK = args{1};
        alpha = 0.01;
        beta = 2;
        kappa = 3-nx;
        lambda = alpha^2*(nx+nv+kappa) - (nx+nv);
    case 2
        nRK = args{1};
        alpha = args{2};
        beta = 2;
        kappa = 3-nx;
        lambda = alpha^2*(nx+nv+kappa) - (nx+nv);
    case 3
        nRK = args{1};
        alpha = args{2};
        beta = args{3};
        kappa = 3-nx;
        lambda = alpha^2*(nx+nv+kappa) - (nx+nv);
    case 4
        nRK = args{1};
        alpha = args{2};
        beta = args{3};
        kappa = args{4};
        lambda = alpha^2*(nx+nv+kappa) - (nx+nv);
    case 5
        nRK = args{1};
        alpha = args{2};
        beta = args{3};
        lambda = args{5};
    otherwise 
        error('Not enough input arguments')
end
% Check optional inputs are sensible, error otherwise
if nRK < 5
    error('Number of Runge-Kutta iterations should be larger than 5')
elseif (alpha > 1 || alpha <= 0)
    error('alpha should be between 0 and 1')
end
end