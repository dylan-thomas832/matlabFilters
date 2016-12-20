function [hk,dhkdxk] = h_ps8prob4(xk,idervflag)

% This function models the measurements of the first filtering example:
% 
%   zk = h(xk,k) + wk = xk + xk^2 + xk^3 + wk
%
%
% Inputs:
%
%   xk
%
%   k
%
%
% Outputs:
%
%   fk
% 
%   dhkdxk
%

% Compute hk
hk = xk + xk^2 + xk^3;

if idervflag == 0
    dhkdxk = [];
    return
end

% Compute the required Jacobian
dhkdxk = 1 + 2*xk + 3*xk^2;
