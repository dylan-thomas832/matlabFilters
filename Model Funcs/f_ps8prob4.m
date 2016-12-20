function [fk,dfkdxk,dfkdvk] = f_ps8prob4(xk,uk,vk,k)

% This function models the dynamics of the first filtering example:
% 
%   xkp1 = f(xk,vk,k) = 2*atan(xk) + 0.5*cos(pi*k/3) + vk.
%
%
% Inputs:
%
%   xk
%
%   vk
%
%   k
%
%
% Outputs:
%
%   fk
% 
%   dfkdxk
%
%   dfkdvk
%

% Compute fk
atanxk = atan(xk);
fk = 2*atanxk + 0.5*cos(pi*k/3) + vk;

% Compute the required Jacobians
dfkdxk = 2/(1 + xk^2);
dfkdvk = 1;
