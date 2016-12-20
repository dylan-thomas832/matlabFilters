%% Iterated, Extended Kalman Filter (iEKF) Class
% *Author: Dylan Thomas*
%
% This class implements an iterated, extended Kalman filter.
%

%% TODO:
%
% # Determine validity of measurement update iteration (not consistent with Niter >2)
% # Demystify beginning sample/tk = 0 problem (kinit maybe?)
% # Get rid of inv( ) warnings
% # Add continuous measurement model functionality?

%% iEKF Class Definition
classdef batch_iEKF < batchFilter
% Inherits batchFilter abstract class

%% iEKF Properties
% *Inputs:*
    properties
        
        nRK         % scalar:
                    %
                    % The Runge Kutta iterations to perform for 
                    % coverting dynamics model from continuous-time 
                    % to discrete-time. Default value is 20 RK 
                    % iterations.
                    
        Niter       % scalar:
                    %
                    % The number of measurement update iterations to 
                    % run through. Default value is 5.
                    
        alphalim    % scalar:
                    %
                    % The lower limit for the Gauss-Newton step 
                    % change. This limit decreases until the cost
                    % function is decreased for the current state 
                    % estimate. If the cost function is not 
                    % decreased, then the alpha value is decreased
                    % to create a different state estimate. Default
                    % value is 0.01.
    end
    
%% iEKF Methods
    methods
        % iEKF constructor
        function iEKFobj = batch_iEKF(fmodel,hmodel,modelFlag,xhat0,P0,uhist,zhist,thist,Q,R,varargin)
            % Prepare for superclass constructor
            if nargin == 0
                super_args = cell(1,11);
            elseif nargin < 10
                error('Not enough input arguments')
            else
                super_args{1}   = fmodel;
                super_args{2}   = hmodel;
                super_args{3}   = modelFlag;
                super_args{4}   = xhat0;
                super_args{5}   = P0;
                super_args{6}   = uhist;
                super_args{7}   = zhist;
                super_args{8}   = thist;
                super_args{9}   = Q;
                super_args{10}  = R;
                super_args{11}  = varargin;
            end
            % batchFilter superclass constructor
            iEKFobj@batchFilter(super_args{:});
            % Extra argument checker method
            iEKFobj = argumentsCheck(iEKFobj);
        end
        
        % This method checks the extra input arguments for iEKF class
        function iEKFobj = argumentsCheck(iEKFobj)
            % Switch on number of extra arguments.
            switch length(iEKFobj.optArgs)
                case 0
                    iEKFobj.nRK = 20;
                    iEKFobj.Niter = 5;
                    iEKFobj.alphalim = 0.01;
                case 1
                    iEKFobj.nRK = iEKFobj.optArgs{1};
                    iEKFobj.Niter = 5;
                    iEKFobj.alphalim = 0.01;
                case 2
                    iEKFobj.nRK = iEKFobj.optArgs{1};
                    iEKFobj.Niter = iEKFobj.optArgs{2};
                    iEKFobj.alphalim = 0.01;
                case 3
                    iEKFobj.nRK = iEKFobj.optArgs{1};
                    iEKFobj.Niter = iEKFobj.optArgs{2};
                    iEKFobj.alphalim = iEKFobj.optArgs{3};
                otherwise
                    error('Not enough input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if iEKFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            elseif (iEKFobj.Niter > 1000 || iEKFobj.Niter < 1)
                error('The number of measurement update iterations is too large')
            elseif (iEKFobj.alphalim >= 1 || iEKFobj.alphalim <= 0)
                error('Lower bound on alpha should be between 0 and 1')
            end
        end
        
        % This method initializes the iEKF class filter
        function [iEKFobj,xhatk,Pk,tk,vk] = initFilter(iEKFobj)
            % Setup the output arrays
            iEKFobj.xhathist     = zeros(iEKFobj.nx,iEKFobj.kmax+1);
            iEKFobj.Phist        = zeros(iEKFobj.nx,iEKFobj.nx,iEKFobj.kmax+1);
            iEKFobj.eta_nuhist   = zeros(size(iEKFobj.thist));
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            xhatk                   = iEKFobj.xhat0;
            Pk                      = iEKFobj.P0;
            iEKFobj.xhathist(:,1)    = iEKFobj.xhat0;
            iEKFobj.Phist(:,:,1)     = iEKFobj.P0;
            tk                      = 0;
            vk                      = zeros(iEKFobj.nv,1);
        end
        
        % This method performs iEKF class filter estimation
        function iEKFobj = doFilter(iEKFobj)
            % Filter initialization method
            [iEKFobj,xhatk,Pk,tk,vk] = initFilter(iEKFobj);
            
            % Main filter loop.
            for k = 0:(iEKFobj.kmax-1)
                % Prepare loop
                kp1 = k+1;
                tkp1 = iEKFobj.thist(kp1);
                
                % Perform dynamic propagation and measurement update
                [xbarkp1,Pbarkp1] = dynamicProp(iEKFobj,xhatk,Pk,vk,tk,tkp1,k);
                [xhatkp1,Pkp1,eta_nukp1] = measUpdate(iEKFobj,xbarkp1,Pbarkp1,kp1);
                
                % Store results
                kp2 = kp1 + 1;
                iEKFobj.xhathist(:,kp2) = xhatkp1;
                iEKFobj.Phist(:,:,kp2) = Pkp1;
                iEKFobj.eta_nuhist(kp1) = eta_nukp1;
                % Prepare for next sample
                xhatk = xhatkp1;
                Pk = Pkp1;
                tk = tkp1;
            end
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function [xbarkp1,Pbarkp1] = dynamicProp(iEKFobj,xhatk,Pk,vk,tk,tkp1,k)
            % Check model types and get sample k a priori state estimate.
            if strcmp(iEKFobj.modelFlag,'CD')
                [xbarkp1,F,Gamma] = c2dnonlinear(xhatk,iEKFobj.uhist(k+1,:)',vk,tk,tkp1,iEKFobj.nRK,iEKFobj.fmodel,1);
            elseif strcmp(iEKFobj.modelFlag,'DD')
                [xbarkp1,F,Gamma] = feval(iEKFobj.fmodel,xhatk,iEKFobj.uhist(k+1,:)',vk,k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
            % Get sample k a priori error covariance
            Pbarkp1 = F*Pk*(F') + Gamma*iEKFobj.Q*(Gamma');
        end
        
        % Measurement update method at sample k+1.
        function [xhatkp1,Pkp1,eta_nukp1] = measUpdate(iEKFobj,xbarkp1,Pbarkp1,kp1)
            % Linearized at sample k+1 a priori state estimate.
            [zbarkp1,H] = feval(iEKFobj.hmodel,xbarkp1,1);
            zkp1 = iEKFobj.zhist(kp1,:)';
            % Innovations, innovation covariance, and filter gain.
            nukp1 = zkp1-zbarkp1;
            Skp1 = H*Pbarkp1*(H') + iEKFobj.R;
            Wkp1 = (Pbarkp1*(H'))/Skp1;
            % LMMSE sample k+1 a posteriori state estimate and covariance.
            Pi = Pbarkp1 - Wkp1*Skp1*(Wkp1');
            xhatip1 = xbarkp1 + Wkp1*nukp1;
            
            % Initialize the measurement update loop
            xhati = xbarkp1;
            for i = 1:iEKFobj.Niter-1
                % Linearized at ith MAP state estimate.
                [zbari,Hi] = feval(iEKFobj.hmodel,xhati,1);
                % Innovations & innovation covariance.
                nukp1 = zkp1-zbari;
                Skp1 = Hi*Pbarkp1*(Hi') + iEKFobj.R;
                % ith MAP error covariance.
                Pi = Pbarkp1 - Pbarkp1*(Hi')/(Hi*Pbarkp1*(Hi') + iEKFobj.R)*Hi*Pbarkp1;
                % (i+1)th MAP state estimate.
                xhatip1 = xhati + Pi*(Hi')*inv(iEKFobj.R)*nukp1 - Pi*inv(Pbarkp1)*(xhati-xbarkp1);
                % Prepare for the next iteration.
                xhati = xhatip1;
            end
            % Calculate the a posteriori state estimate & error covariance.
            Pkp1 = Pi;
            xhatkp1 = xhatip1;
            % Innovation statistics
            eta_nukp1 = nukp1'*inv(Skp1)*nukp1;
        end
    end
end