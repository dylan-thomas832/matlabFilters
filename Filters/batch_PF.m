%% Particle Filter (PF) Class
% *Author: Dylan Thomas*
%
% This class implements a particle filter.
%

%% TODO:
%
% # Demystify beginning sample/tk = 0 problem (kinit maybe?)
% # Get rid of inv( ) warnings
% # Add continuous measurement model functionality?

%% PF Class Definition
classdef batch_PF < batchFilter
% Inherits batchFilter abstract class

%% PF Properties
% *Inputs:*
    properties
        
        nRK             % scalar:
                        %      
                        % The Runge Kutta iterations to perform for 
                        % coverting dynamics model from continuous-time 
                        % to discrete-time. Default value is 20 RK 
                        % iterations.
        
        Np              % scalar:
                        %
                        % The number of particles to generate and use in
                        % estimating the state. The default value is 100.
                    
        resampleFlag    % integer"
                        %
                        % Flag to determine which sampling algorithm to
                        % implement. Options are 0,1,2 which correspond to
                        % no resample, resample1, and resample2
                        % respectivley. Default value is 1.
    end
    
%% PF Methods
    methods
        % PF constructor
        function PFobj = batch_PF(fmodel,hmodel,modelFlag,kInit,xhatInit,PInit,uhist,zhist,thist,Q,R,varargin)
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
            PFobj@batchFilter(super_args{:});
            % Extra argument checker method
            PFobj = argumentsCheck(PFobj);
        end
        
        % This method checks the extra input arguments for PF class
        function PFobj = argumentsCheck(PFobj)
            % Switch on number of extra arguments.
            switch length(PFobj.optArgs)
                case 0
                    PFobj.nRK          = 20;
                    PFobj.Np           = 100;
                    PFobj.resampleFlag = 1;
                case 1
                    PFobj.nRK          = PFobj.optArgs{1};
                    PFobj.Np           = 100;
                    PFobj.resampleFlag = 1;
                case 2
                    PFobj.nRK          = PFobj.optArgs{1};
                    PFobj.Np           = PFobj.optArgs{2};
                    PFobj.resampleFlag = 1;
                case 3
                    PFobj.nRK          = PFobj.optArgs{1};
                    PFobj.Np           = PFobj.optArgs{2};
                    PFobj.resampleFlag = PFobj.optArgs{3};
                otherwise
                    error('Not enough input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if PFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            end
            if ~(PFobj.resampleFlag==0||PFobj.resampleFlag==1||PFobj.resampleFlag==2)
                error('Input resample flag does not correspond with a proper option')
            end
        end
        
        % This method initializes the PF class filter
        function [PFobj,tk,Xikp1,Wkp1] = initFilter(PFobj)
            % Setup the output arrays
            PFobj.xhathist     = zeros(PFobj.nx,PFobj.kmax+1);
            PFobj.Phist        = zeros(PFobj.nx,PFobj.nx,PFobj.kmax+1);
            PFobj.eta_nuhist   = zeros(size(PFobj.thist));
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            PFobj.xhathist(:,PFobj.kInit+1)     = PFobj.xhatInit;
            PFobj.Phist(:,:,PFobj.kInit+1)      = PFobj.PInit;
            % Make sure correct initial tk is used.
            if PFobj.kInit == 0
                tk = 0;
            else
                tk = PFobj.thist(PFobj.kInit);
            end
            % Generate Ns samples of Xi0 from N[x(k);xhatInit,PInit] and weights
            Xikp1 = PFobj.xhatInit*ones(1,PFobj.Np) + chol(PFobj.PInit)'*randn(PFobj.nx,PFobj.Np);
            Wkp1 = ones(1,PFobj.Np)*(1/PFobj.Np);
        end
        
        % This method performs PF class filter estimation
        function PFobj = doFilter(PFobj)
            % Filter initialization method
            [PFobj,tk,Xikp1,Wkp1] = initFilter(PFobj);
            
            % Main filter loop.
            for k = PFobj.kInit:(PFobj.kmax-1)
                Xik = Xikp1;
                Wk = Wkp1;
                logWk = log(Wk);
                % Sample vi(k) from N[v(k);0,Q(k)]
                vik = chol(PFobj.Q)'*randn(PFobj.nv,PFobj.Np);
                % Preallocation
                Xikp1 = zeros(PFobj.nx,PFobj.Np);
                logWbarkp1 = zeros(1,PFobj.nx);
                % Prepare loop
                kp1 = k+1;
                tkp1 = PFobj.thist(kp1);
                zkp1 = PFobj.zhist(kp1,:)';
                uk = PFobj.uhist(kp1,:)';
                
                for ii = 1:PFobj.Np
                    % Propagate particle to sample k+1
                    vk = vik(:,ii);
                    xk_ii = Xik(:,ii);
                    xkp1_ii = dynamicProp(PFobj,xk_ii,uk,vk,tk,tkp1,k);
%                     Xikp1(:,ii) = fmodel(Xik(:,ii),uhist(kp1,:)',vik(:,ii),k);
                    %  TODO: FIX!!!
                    [zkp1_ii,~] = feval(PFobj.hmodel,xkp1_ii,kp1,0);
                    % Dummy calc for debug
%                     zdum = zhist(kp1,:)'-hmodel(Xikp1(:,ii),k+1);
                    % Calculate the current particle's log-weight
                    logWbarkp1(ii) = -0.5*(zkp1-zkp1_ii)'*inv(PFobj.R)*(zkp1-zkp1_ii) + logWk(ii);
                    Xikp1(:,ii) = xkp1_ii;
                end
                % Find imax which has a greater log-weight than every other log-weight
                Wmax = max(logWbarkp1);
                % Normalize weights
                Wbarbarkp1 = exp(logWbarkp1-Wmax);
                Wkp1 = Wbarbarkp1/sum(Wbarbarkp1);
                
                [xhatkp1,Pkp1] = measUpdate(PFobj,Xikp1,Wkp1);
                
                % Resample
                if PFobj.resampleFlag == 1
                    [Xikp1,Wkp1] = resample1(Xikp1,Wkp1,PFobj.Np);
%                 elseif PFobj.resampleFlag == 2
                    
                end
                
                % Store results
                kp2 = kp1 + 1;
                PFobj.xhathist(:,kp2) = xhatkp1;
                PFobj.Phist(:,:,kp2) = Pkp1;
%                 PFobj.eta_nuhist(kp1) = eta_nukp1;
                % Prepare for next sample
%                 xhatk = xhatkp1;
%                 Pk = Pkp1;
                tk = tkp1;
            end
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function xbarkp1 = dynamicProp(PFobj,xhatk,uk,vk,tk,tkp1,k)
            % Check model types and get sample k a priori state estimate.
            if strcmp(PFobj.modelFlag,'CD')
                xbarkp1 = c2dnonlinear(xhatk,uk,vk,tk,tkp1,PFobj.nRK,PFobj.fmodel,0);
            elseif strcmp(PFobj.modelFlag,'DD')
                xbarkp1 = feval(PFobj.fmodel,xhatk,PFobj.uhist(k+1,:)',vk,k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
        end
        
        % Measurement update method at sample k+1.
        function [xhatkp1,Pkp1] = measUpdate(PFobj,Xikp1,Wkp1)
            % Compute xhat(k+1) and P(k+1) from weights and particles
            xhatkp1 = zeros(PFobj.nx,1);
            for ii = 1:PFobj.Np
                xhatkp1 = xhatkp1 + Wkp1(ii)*Xikp1(:,ii);
            end
            
            Psums = zeros(PFobj.nx,PFobj.nx,PFobj.Np);
            for ii = 1:PFobj.Np
                Psums(:,:,ii) = Wkp1(ii)*((Xikp1(:,ii)-xhatkp1)*(Xikp1(:,ii)-xhatkp1)');
            end
            Pkp1 = sum(Psums,3);
%             % Innovations, innovation covariance, and filter gain.
%             nukp1 = zkp1 - zbarkp1;
%             Skp1 = H*Pbarkp1*(H') + PFobj.R;
%             % Innovation statistics
%             eta_nukp1 = nukp1'*inv(Skp1)*nukp1;
        end
    end
end

%% Helper Function
function [Xi,Wi] = resample1(Xi,Wi,Np)
    
% Initialize re-sample algorithm
c = zeros(1,Np+1);
c(end) = 1.0000000001;
% Calculate c coefficients
for ii = 2:Np
    c(ii) = sum(Wi(1:ii-1));
end
    
% Preallocation
Xinew = zeros(size(Xi));

% Initialize
ll = 1;
while ll <= Np
    % Sample random, uniformly distributed variable
    eta = rand(1,1);
    
    % Find indices
    for ii = 1:Np
        % Check that c(i)<= eta <= c(i+1)
        if (c(ii) <= eta && c(ii+1) >= eta)
            % Save the index
            index = ii;
            break
        end
    end
    
    % Generate new particles with approriate indices
    Xinew(:,ll) = Xi(:,index);
    % iterate
    ll = ll + 1;
end
% Reset particle weights for sample k.
Wi = ones(1,Np)*(1/Np);
% Assign new, re-sampled particles to original particles variable
Xi = Xinew;
end