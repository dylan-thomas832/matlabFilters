%% Regularized Particle Filter (RegPF) Class
% *Author: Dylan Thomas*
%
% This class implements a regularized particle filter.
%
% *Note: the user must instantiate the class with no properties, as a
% structure with fieldnames matching all input properties, or all input
% properties separately.*

%% TODO:
%
% # Still weird answers for tricycle problem
% # Write resample2?
% # Clean up and get innovation statistics

%% PF Class Definition
classdef batch_RegPF < batchFilter
% Inherits batchFilter abstract class

%% RegPF Properties
% *Inputs:*
    properties
        
        nRK             % Scalar >= 5:
                        %      
                        % The Runge Kutta iterations to perform for 
                        % coverting dynamics model from continuous-time 
                        % to discrete-time. Default value is 10 RK 
                        % iterations.
        
        Np              % Scalar > 0:
                        %
                        % The number of particles to generate and use in
                        % estimating the state. The default value is 100.
                        
        NT              % Scalar > 1:
                        % 
                        % Lower bound on Neff which determines if
                        % regularization and resampling is performed.
                        % Default value is 50% of Np.
                    
        resampleFlag    % Integer either 0,1,2:
                        %
                        % Flag to determine which sampling algorithm to
                        % implement. Options are 0,1,2 which correspond to
                        % no resample, resample1, and resample2
                        % respectivley. Default value is 1.
    end
    
%% RegPF Methods
    methods
        % Input order:
        % fmodel,hmodel,modelFlag,kInit,xhatInit,PInit,uhist,zhist,thist,Q,R,nRK,Np,NT,resampleFlag
        function RegPFobj = batch_RegPF(varargin)
            % Prepare for superclass constructor
            if nargin == 0
                fprintf('Instantiating empty batch RegPF class\n\n')
                super_args = {};
            elseif nargin == 1
                fprintf('Instantiating batch RegPF class\n\n')
                super_args = varargin{1};
            elseif nargin < 11
                error('Not enough input arguments')
            else
                fprintf('Instantiating batch RegPF class\n\n')
                super_args = cell(1,12);
                for jj = 1:12
                    super_args{jj} = varargin{jj};
                end
            end
            % batchFilter superclass constructor
            RegPFobj@batchFilter(super_args{:});
            % Only do if intantiated class is not empty
            if nargin > 0
                % Extra argument checker method
                RegPFobj = argumentsCheck(RegPFobj);
            end
        end
        
        % This method checks the extra input arguments for RegPF class
        function RegPFobj = argumentsCheck(RegPFobj)
            % Switch on number of extra arguments.
            switch length(RegPFobj.optArgs)
                case 0
                    RegPFobj.nRK          = 10;
                    RegPFobj.Np           = 100;
                    RegPFobj.NT           = round(50*RegPFobj.Np);
                    RegPFobj.resampleFlag = 1;
                case 1
                    RegPFobj.nRK          = RegPFobj.optArgs{1};
                    RegPFobj.Np           = 100;
                    RegPFobj.NT           = round(50*RegPFobj.Np);
                    RegPFobj.resampleFlag = 1;
                case 2
                    RegPFobj.nRK          = RegPFobj.optArgs{1};
                    RegPFobj.Np           = RegPFobj.optArgs{2};
                    RegPFobj.NT           = round(50*RegPFobj.Np);
                    RegPFobj.resampleFlag = 1;
                case 3
                    RegPFobj.nRK          = RegPFobj.optArgs{1};
                    RegPFobj.Np           = RegPFobj.optArgs{2};
                    RegPFobj.NT           = RegPFobj.optArgs{3};
                    RegPFobj.resampleFlag = 1;
                case 4
                    RegPFobj.nRK          = RegPFobj.optArgs{1};
                    RegPFobj.Np           = RegPFobj.optArgs{2};
                    RegPFobj.NT           = RegPFobj.optArgs{3};
                    RegPFobj.resampleFlag = RegPFobj.optArgs{4};
                otherwise
                    error('Too many input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if RegPFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            end
            if ~(RegPFobj.resampleFlag==0||RegPFobj.resampleFlag==1||RegPFobj.resampleFlag==2)
                error('Input resample flag does not correspond with a proper option')
            end
        end
        
        % This method initializes the RegPF class filter
        function [RegPFobj,tk,Xikp1,Wkp1] = initFilter(RegPFobj)
            % Setup the output arrays
            RegPFobj.xhathist     = zeros(RegPFobj.nx,RegPFobj.kmax+1);
            RegPFobj.Phist        = zeros(RegPFobj.nx,RegPFobj.nx,RegPFobj.kmax+1);
            RegPFobj.eta_nuhist   = zeros(size(RegPFobj.thist));
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            RegPFobj.xhathist(:,RegPFobj.kInit+1)     = RegPFobj.xhatInit;
            RegPFobj.Phist(:,:,RegPFobj.kInit+1)      = RegPFobj.PInit;
            % Make sure correct initial tk is used.
            if RegPFobj.kInit == 0
                tk = 0;
            else
                tk = RegPFobj.thist(RegPFobj.kInit);
            end
            % Generate Ns samples of Xi0 from N[x(k);xhatInit,PInit] and weights
            Xikp1 = RegPFobj.xhatInit*ones(1,RegPFobj.Np) + chol(RegPFobj.PInit)'*randn(RegPFobj.nx,RegPFobj.Np);
            Wkp1 = ones(1,RegPFobj.Np)*(1/RegPFobj.Np);
        end
        
        % This method performs RegPF class filter estimation
        function RegPFobj = doFilter(RegPFobj)
            % Filter initialization method
            [RegPFobj,tk,Xikp1,Wkp1] = initFilter(RegPFobj);
            
            % Main filter loop.
            for k = RegPFobj.kInit:(RegPFobj.kmax-1)
                Xik = Xikp1;
                Wk = Wkp1;
                logWk = log(Wk);
                % Sample vi(k) from N[v(k);0,Q(k)]
                vik = chol(RegPFobj.Q)'*randn(RegPFobj.nv,RegPFobj.Np);
                % Preallocation
                Xikp1 = zeros(RegPFobj.nx,RegPFobj.Np);
                logWbarkp1 = zeros(1,RegPFobj.nx);
                % Prepare loop
                kp1 = k+1;
                tkp1 = RegPFobj.thist(kp1);
                zkp1 = RegPFobj.zhist(kp1,:)';
                uk = RegPFobj.uhist(kp1,:)';
                
                for ii = 1:RegPFobj.Np
                    % Propagate particle to sample k+1
                    vk = vik(:,ii);
                    xk_ii = Xik(:,ii);
                    xkp1_ii = dynamicProp(RegPFobj,xk_ii,uk,vk,tk,tkp1,k);
%                     Xikp1(:,ii) = fmodel(Xik(:,ii),uhist(kp1,:)',vik(:,ii),k);
                    %  TODO: FIX!!!
                    [zkp1_ii,~] = feval(RegPFobj.hmodel,xkp1_ii,kp1,0);
                    % Dummy calc for debug
%                     zdum = zhist(kp1,:)'-hmodel(Xikp1(:,ii),k+1);
                    % Calculate the current particle's log-weight
                    logWbarkp1(ii) = -0.5*(zkp1-zkp1_ii)'*(RegPFobj.R\(zkp1-zkp1_ii)) + logWk(ii);
                    Xikp1(:,ii) = xkp1_ii;
                end
                % Find imax which has a greater log-weight than every other log-weight
                Wmax = max(logWbarkp1);
                % Normalize weights
                Wbarbarkp1 = exp(logWbarkp1-Wmax);
                Wkp1 = Wbarbarkp1/sum(Wbarbarkp1);
                
                [xhatkp1,Pkp1] = measUpdate(RegPFobj,Xikp1,Wkp1);
                
                % Compute Neff and determine whether to re-sample or not
                Neff = 1/sum(Wkp1.^2);
                if (Neff>RegPFobj.NT)
                    return; % Neff is sufficiently large
                else
                    % Not enough effective particles 
                    
                    % Resample
                    if RegPFobj.resampleFlag == 1
                        [Xikp1,Wkp1] = resample1(Xikp1,Wkp1,RegPFobj.Np);
                        %                 elseif RegPFobj.resampleFlag == 2
                    end
                    
                    % Cholesky fctorize covariance
                    [Skp1,err] = chol(Pkp1);
                    %%%%% TODO: Cheap get around
                    if err > 0
                        Skp1 = zeros(size(Pkp1));
                    else
                        Skp1 = Skp1';
                    end
                    % Sample Beta from Kernel function. Need other outputs?
                    [Beta,~,~,~] = epanechnikovsample01(RegPFobj.nx,RegPFobj.Np);
                    % Hypersphere in nx space
                    Cnx = unithypervolume01(RegPFobj.nx);
                    % Coefficient calculations
                    A = ((8/Cnx)*(RegPFobj.nx+4)*(2*sqrt(pi))^RegPFobj.nx)^(1/(RegPFobj.nx+4));
                    hopt = A/(RegPFobj.Np^(1/(RegPFobj.nx+4)));
                    
                    % Recalculate the particles at sample k.
                    Xikp1 = Xikp1 + hopt*Skp1*Beta;
                end
                % Store results
                kp2 = kp1 + 1;
                RegPFobj.xhathist(:,kp2) = xhatkp1;
                RegPFobj.Phist(:,:,kp2) = Pkp1;
%                 RegPFobj.eta_nuhist(kp1) = eta_nukp1;
                % Prepare for next sample
%                 xhatk = xhatkp1;
%                 Pk = Pkp1;
                tk = tkp1;
            end
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function xbarkp1 = dynamicProp(RegPFobj,xhatk,uk,vk,tk,tkp1,k)
            % Check model types and get sample k a priori state estimate.
            if strcmp(RegPFobj.modelFlag,'CD')
                xbarkp1 = c2dnonlinear(xhatk,uk,vk,tk,tkp1,RegPFobj.nRK,RegPFobj.fmodel,0);
            elseif strcmp(RegPFobj.modelFlag,'DD')
                xbarkp1 = feval(RegPFobj.fmodel,xhatk,uk,vk,k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
        end
        
        % Measurement update method at sample k+1.
        function [xhatkp1,Pkp1] = measUpdate(RegPFobj,Xikp1,Wkp1)
            % Compute xhat(k+1) and P(k+1) from weights and particles
            xhatkp1 = zeros(RegPFobj.nx,1);
            for ii = 1:RegPFobj.Np
                xhatkp1 = xhatkp1 + Wkp1(ii)*Xikp1(:,ii);
            end
%             xhatkp1 = Xikp1*Wkp1';
            
            Psums = zeros(RegPFobj.nx,RegPFobj.nx,RegPFobj.Np);
            for ii = 1:RegPFobj.Np
                Psums(:,:,ii) = Wkp1(ii)*((Xikp1(:,ii)-xhatkp1)*(Xikp1(:,ii)-xhatkp1)');
            end
            Pkp1 = sum(Psums,3);
%             % Innovations, innovation covariance, and filter gain.
%             nukp1 = zkp1 - zbarkp1;
%             Skp1 = H*Pbarkp1*(H') + RegPFobj.R;
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