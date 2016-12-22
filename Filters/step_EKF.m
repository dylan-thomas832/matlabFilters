%% Extended Kalman Filter (EKF) Class
% *Author: Dylan Thomas*
%
% This class implements an extended Kalman filter.
%
% *Note: the user must instantiate the class with no properties, as a
% structure with fieldnames matching all input properties, or all input
% properties separately.*

%% TODO:
%
% # Get rid of inv( ) warnings
% # Add continuous measurement model functionality?

%% EKF Class Definition
classdef step_EKF < stepFilter
% Inherits stepFilter abstract class

%% EKF Properties
% *Inputs:*
    properties
        
        nRK         % scalar:
                    %
                    % The Runge Kutta iterations to perform for 
                    % coverting dynamics model from continuous-time 
                    % to discrete-time. Default value is 20 RK 
                    % iterations.
    end
    
%% EKF Methods
    methods
        % EKF constructor
        % Input order:
        % fmodel,hmodel,modelFlag,k,xhatk,k,uk,zkp1,tk,dt,Qk,Rkp1,nRK
        function EKFobj = step_EKF(varargin)
            % Prepare for superclass constructor
            if nargin == 0
                fprintf('Instantiating empty step EKF class\n\n')
                super_args = {};
            elseif nargin == 1
                fprintf('Instantiating step EKF class\n\n')
                super_args = varargin{1};
            elseif nargin < 12
                error('Not enough input arguments')
            else
                fprintf('Instantiating step EKF class\n\n')
                super_args = cell(1,13);
                for jj = 1:13
                    super_args{jj} = varargin{jj};
                end
            end
            % stepFilter superclass constructor
            EKFobj@stepFilter(super_args{:});
            % Only do if intantiated class is not empty
            if nargin > 0
                % Extra argument checker method
                EKFobj = argumentsCheck(EKFobj);
            end
        end
        
        % This method checks the extra input arguments for EKF class
        function EKFobj = argumentsCheck(EKFobj)
            % Switch on number of extra arguments.
            switch length(EKFobj.optArgs)
                case 0
                    EKFobj.nRK = 20;
                case 1
                    EKFobj.nRK = EKFobj.optArgs{1};
                otherwise
                    error('Too many input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if EKFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            end
        end
        
        % This method initializes the EKF class filter
        function EKFobj = initFilter(EKFobj)
            % Do nothing
        end
        
        % This method performs EKF class filter estimation
        function EKFobj = doFilter(EKFobj)
            % Filter initialization method
            EKFobj = initFilter(EKFobj);
            
            % Perform dynamic propagation and measurement update
            [xbarkp1,Pbarkp1] = dynamicProp(EKFobj);
            EKFobj = measUpdate(EKFobj,xbarkp1,Pbarkp1);
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function [xbarkp1,Pbarkp1] = dynamicProp(EKFobj)
            vk = zeros(EKFobj.nv,1);
            % Check model types and get sample k+1 a priori state estimate.
            if strcmp(EKFobj.modelFlag,'CD')
                [xbarkp1,F,Gamma] = c2dnonlinear(EKFobj.xhatk,EKFobj.uk,vk,EKFobj.tk,EKFobj.tkp1,EKFobj.nRK,EKFobj.fmodel,1);
            elseif strcmp(EKFobj.modelFlag,'DD')
                [xbarkp1,F,Gamma] = feval(EKFobj.fmodel,EKFobj.xhatk,EKFobj.uk,vk,EKFobj.k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
            % Get sample k a priori error covariance
            Pbarkp1 = F*EKFobj.Pk*(F') + Gamma*EKFobj.Qk*(Gamma');
        end
        
        % Measurement update method at sample k+1.
        function EKFobj = measUpdate(EKFobj,xbarkp1,Pbarkp1)
            % Linearized at sample k+1 a priori state estimate.
            [zbarkp1,H] = feval(EKFobj.hmodel,xbarkp1,EKFobj.k+1,1);
            % Innovations, innovation covariance, and filter gain.
            nukp1 = EKFobj.zkp1 - zbarkp1;
            Skp1 = H*Pbarkp1*(H') + EKFobj.Rkp1;
            Wkp1 = (Pbarkp1*(H'))/Skp1;
            % LMMSE sample k+1 a posteriori state estimate and covariance.
            EKFobj.xhatkp1 = xbarkp1 + Wkp1*nukp1;
            EKFobj.Pkp1 = Pbarkp1 - Wkp1*Skp1*(Wkp1');
            % Innovation statistic
            EKFobj.eta_nukp1 = nukp1'*inv(Skp1)*nukp1;
        end
    end
end