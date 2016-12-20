%% Extended Kalman Filter (EKF) Class
% *Author: Dylan Thomas*
%
% This class implements an extended Kalman filter.
%

%% TODO:
%
% # Get rid of inv( ) warnings
% # Add continuous measurement model functionality?

%% EKF Class Definition
classdef batch_EKF < batchFilter
% Inherits batchFilter abstract class

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
        function EKFobj = batch_EKF(fmodel,hmodel,modelFlag,kInit,xhatInit,PInit,uhist,zhist,thist,Q,R,varargin)
            % Prepare for superclass constructor
            if nargin == 0
                super_args = cell(1,12);
            elseif nargin < 11
                error('Not enough input arguments')
            else
                super_args{1}   = fmodel;
                super_args{2}   = hmodel;
                super_args{3}   = modelFlag;
                super_args{4}   = kInit;
                super_args{5}   = xhatInit;
                super_args{6}   = PInit;
                super_args{7}   = uhist;
                super_args{8}   = zhist;
                super_args{9}   = thist;
                super_args{10}  = Q;
                super_args{11}  = R;
                super_args{12}  = varargin;
            end
            % batchFilter superclass constructor
            EKFobj@batchFilter(super_args{:});
            % Extra argument checker method
            EKFobj = argumentsCheck(EKFobj);
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
                    error('Not enough input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if EKFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            end
        end
        
        % This method initializes the EKF class filter
        function [EKFobj,xhatk,Pk,tk,vk] = initFilter(EKFobj)
            % Setup the output arrays
            EKFobj.xhathist     = zeros(EKFobj.nx,EKFobj.kmax+1);
            EKFobj.Phist        = zeros(EKFobj.nx,EKFobj.nx,EKFobj.kmax+1);
            EKFobj.eta_nuhist   = zeros(size(EKFobj.thist));
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            xhatk                                = EKFobj.xhatInit;
            Pk                                   = EKFobj.PInit;
            EKFobj.xhathist(:,EKFobj.kInit+1)    = EKFobj.xhatInit;
            EKFobj.Phist(:,:,EKFobj.kInit+1)     = EKFobj.PInit;
            vk                                   = zeros(EKFobj.nv,1);
            % Make sure correct initial tk is used.
            if EKFobj.kInit == 0
                tk = 0;
            else
                tk = EKFobj.thist(EKFobj.kInit);
            end
        end
        
        % This method performs EKF class filter estimation
        function EKFobj = doFilter(EKFobj)
            % Filter initialization method
            [EKFobj,xhatk,Pk,tk,vk] = initFilter(EKFobj);
            
            % Main filter loop.
            for k = EKFobj.kInit:(EKFobj.kmax-1)
                % Prepare loop
                kp1 = k+1;
                tkp1 = EKFobj.thist(kp1);
                uk = EKFobj.uhist(kp1,:)';
                
                % Perform dynamic propagation and measurement update
                [xbarkp1,Pbarkp1] = dynamicProp(EKFobj,xhatk,Pk,uk,vk,tk,tkp1,k);
                [xhatkp1,Pkp1,eta_nukp1] = measUpdate(EKFobj,xbarkp1,Pbarkp1,kp1);
                
                % Store results
                kp2 = kp1 + 1;
                EKFobj.xhathist(:,kp2) = xhatkp1;
                EKFobj.Phist(:,:,kp2) = Pkp1;
                EKFobj.eta_nuhist(kp1) = eta_nukp1;
                % Prepare for next sample
                xhatk = xhatkp1;
                Pk = Pkp1;
                tk = tkp1;
            end
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function [xbarkp1,Pbarkp1] = dynamicProp(EKFobj,xhatk,Pk,uk,vk,tk,tkp1,k)
            % Check model types and get sample k a priori state estimate.
            if strcmp(EKFobj.modelFlag,'CD')
                [xbarkp1,F,Gamma] = c2dnonlinear(xhatk,uk,vk,tk,tkp1,EKFobj.nRK,EKFobj.fmodel,1);
            elseif strcmp(EKFobj.modelFlag,'DD')
                [xbarkp1,F,Gamma] = feval(EKFobj.fmodel,xhatk,uk,vk,k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
            % Get sample k a priori error covariance
            Pbarkp1 = F*Pk*(F') + Gamma*EKFobj.Q*(Gamma');
        end
        
        % Measurement update method at sample k+1.
        function [xhatkp1,Pkp1,eta_nukp1] = measUpdate(EKFobj,xbarkp1,Pbarkp1,kp1)
            % Linearized at sample k+1 a priori state estimate.
            [zbarkp1,H] = feval(EKFobj.hmodel,xbarkp1,kp1,1);
            zkp1 = EKFobj.zhist(kp1,:)';
            % Innovations, innovation covariance, and filter gain.
            nukp1 = zkp1 - zbarkp1;
            Skp1 = H*Pbarkp1*(H') + EKFobj.R;
            Wkp1 = (Pbarkp1*(H'))/Skp1;
            % LMMSE sample k+1 a posteriori state estimate and covariance.
            xhatkp1 = xbarkp1 + Wkp1*nukp1;
            Pkp1 = Pbarkp1 - Wkp1*Skp1*(Wkp1');
            % Innovation statistics
            eta_nukp1 = nukp1'*inv(Skp1)*nukp1;
        end
    end
end