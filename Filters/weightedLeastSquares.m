%%% Weighted Least-Squares Estimation
function [x_hat,J_opt,check] = weightedLeastSquares(z,H,R)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Dylan Thomas - October 18, 2016
%
% Function to calculate the weighted least-squares estimate of x using
% square root method for increased computational accuracy
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Inputs
% 
%       z: (n_z by 1)   Vector of measurements
%       H: (n_z by n_x) Measurement influence matrix
%       R: (n_z by n_z) Measurement error covariance matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Outputs
%
%   x_hat: (n_x by 1)   Estimated state vector
%   J_opt: scalar       Optimal value of cost function
%   check: scalar       Check of 1st-order necessary conditions
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get size of state vector
nx = size(H,2);
% Cholesky factorize, and transform inputs to "square variants"
Ra = chol(R);
za = (Ra')\z;
Ha = (Ra')\H;
% QR Factorize Ha into Qb & Rb. Inefficient...
[Qb,~] = qr(Ha);
[~,Rb] = qr(Ha,0);
% Convert measurements, and split
zb = Qb'*za;
zb1 = zb(1:nx);
zb2 = zb(nx+1:end);
% Find the optimal state estimate using Rb and zb1
x_hat = Rb\zb1;
% Find the optimal cost function value with zb2
J_opt = zb2'*zb2;
% Check that the norm is small at the solution. Poor check if x = 0;
check = norm(-H'*(R\(z-H*x_hat)))/norm(-H'*(R\z));
end