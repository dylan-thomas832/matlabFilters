%% Unscented Kalman Filter (UKF) Class
% *Author: Dylan Thomas*
%
% This class implements an unscented Kalman filter (Sigma Points filter).
%
% *Note: the user must instantiate the class with no properties, as a
% structure with fieldnames matching all input properties, or all input
% properties separately.*

%% TODO:
%
% # Get rid of inv( ) warnings
% # Add continuous measurement model functionality?
% # Finish input descriptions

%% UKF Class Definition
classdef batch_UKF < batchFilter
% Inherits batchFilter abstract class

%% UKF Properties
% *Inputs:*
    properties
        
        nRK         % scalar:
                    %
                    % The Runge Kutta iterations to perform for 
                    % coverting dynamics model from continuous-time 
                    % to discrete-time. Default value is 20 RK 
                    % iterations.
                    
        alpha
        
        beta
        
        kappa
        
        lambda
    end
    
%% UKF Methods
    methods
        % UKF constructor
        % Input order:
        % fmodel,hmodel,modelFlag,kInit,xhatInit,PInit,uhist,zhist,thist,Q,R,nRK,alpha,beta,kappa,lambda
        function UKFobj = batch_UKF(varargin)
            % Prepare for superclass constructor
            if nargin == 0
                fprintf('Instantiating empty batch UKF class\n\n')
                super_args = {};
            elseif nargin == 1
                fprintf('Instantiating batch UKF class\n\n')
                super_args = varargin{1};
            elseif nargin < 11
                error('Not enough input arguments')
            else
                fprintf('Instantiating batch UKF class\n\n')
                super_args = cell(1,12);
                super_args{1}   = varargin{1};
                super_args{2}   = varargin{2};
                super_args{3}   = varargin{3};
                super_args{4}   = varargin{4};
                super_args{5}   = varargin{5};
                super_args{6}   = varargin{6};
                super_args{7}   = varargin{7};
                super_args{8}   = varargin{8};
                super_args{9}   = varargin{9};
                super_args{10}  = varargin{10};
                super_args{11}  = varargin{11};
                super_args{12}  = varargin{12:end};
            end
            % batchFilter superclass constructor
            UKFobj@batchFilter(super_args{:});
            % Only do if intantiated class is not empty
            if nargin > 0
                % Extra argument checker method
                UKFobj = argumentsCheck(UKFobj);
            end
        end
        
        % This method checks the extra input arguments for UKF class
        function UKFobj = argumentsCheck(UKFobj)
            % Switch on number of extra arguments.
            switch length(UKFobj.optArgs)
                case 0
                    UKFobj.nRK    = 20;
                    % Scaling parameter
                    UKFobj.alpha  = 0.01;
                    % Optimal for Gaussian distribution samples
                    UKFobj.beta   = 2;
                    UKFobj.kappa  = 3-UKFobj.nx;
                    % UKF scaling parameter.
                    UKFobj.lambda = UKFobj.alpha^2*(UKFobj.nx+UKFobj.nv+UKFobj.kappa) - (UKFobj.nx+UKFobj.nv);
                case 1
                    UKFobj.nRK    = UKFobj.optArgs{1};
                    UKFobj.alpha  = 0.01;
                    UKFobj.beta   = 2;
                    UKFobj.kappa  = 3-UKFobj.nx;
                    UKFobj.lambda = UKFobj.alpha^2*(UKFobj.nx+UKFobj.nv+UKFobj.kappa) - (UKFobj.nx+UKFobj.nv);
                case 2
                    UKFobj.nRK    = UKFobj.optArgs{1};
                    UKFobj.alpha  = UKFobj.optArgs{2};
                    UKFobj.beta   = 2;
                    UKFobj.kappa  = 3-UKFobj.nx;
                    UKFobj.lambda = UKFobj.alpha^2*(UKFobj.nx+UKFobj.nv+UKFobj.kappa) - (UKFobj.nx+UKFobj.nv);
                case 3
                    UKFobj.nRK    = UKFobj.optArgs{1};
                    UKFobj.alpha  = UKFobj.optArgs{2};
                    UKFobj.beta   = UKFobj.optArgs{3};
                    UKFobj.kappa  = 3-UKFobj.nx;
                    UKFobj.lambda = UKFobj.alpha^2*(UKFobj.nx+UKFobj.nv+UKFobj.kappa) - (UKFobj.nx+UKFobj.nv);
                case 4
                    UKFobj.nRK    = UKFobj.optArgs{1};
                    UKFobj.alpha  = UKFobj.optArgs{2};
                    UKFobj.beta   = UKFobj.optArgs{3};
                    UKFobj.kappa  = UKFobj.optArgs{4};
                    UKFobj.lambda = UKFobj.alpha^2*(UKFobj.nx+UKFobj.nv+UKFobj.kappa) - (UKFobj.nx+UKFobj.nv);
                case 5
                    UKFobj.nRK    = UKFobj.optArgs{1};
                    UKFobj.alpha  = UKFobj.optArgs{2};
                    UKFobj.beta   = UKFobj.optArgs{3};
                    UKFobj.lambda = UKFobj.optArgs{5};
                otherwise
                    error('Too many input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if UKFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            elseif (UKFobj.alpha > 1 || UKFobj.alpha <= 0)
                error('alpha should be between 0 and 1')
            end
        end
        
        % This method initializes the UKF class filter
        function [UKFobj,xhatk,Pk,tk,vk,Svvk,Nsigma] = initFilter(UKFobj)
            % Setup the output arrays
            UKFobj.xhathist     = zeros(UKFobj.nx,UKFobj.kmax+1);
            UKFobj.Phist        = zeros(UKFobj.nx,UKFobj.nx,UKFobj.kmax+1);
            UKFobj.eta_nuhist   = zeros(size(UKFobj.thist));
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            xhatk                                = UKFobj.xhatInit;
            Pk                                   = UKFobj.PInit;
            UKFobj.xhathist(:,UKFobj.kInit+1)    = UKFobj.xhatInit;
            UKFobj.Phist(:,:,UKFobj.kInit+1)     = UKFobj.PInit;
            vk                                   = zeros(UKFobj.nv,1);
            Svvk                                 = chol(UKFobj.Q)';
            Nsigma                               = 1 + 2*(UKFobj.nx+UKFobj.nv);
            % Make sure correct initial tk is used.
            if UKFobj.kInit == 0
                tk = 0;
            else
                tk = UKFobj.thist(UKFobj.kInit);
            end
        end
        
        % This method performs UKF class filter estimation
        function UKFobj = doFilter(UKFobj)
            % Filter initialization method
            [UKFobj,xhatk,Pk,tk,vhatk,Svvk,Nsigma] = initFilter(UKFobj);

            % Main filter loop.
            for k = UKFobj.kInit:(UKFobj.kmax-1)
                % Prepare loop
                kp1     = k+1;
                tkp1    = UKFobj.thist(kp1);
                uk      = UKFobj.uhist(kp1,:)';
                zkp1    = UKFobj.zhist(kp1,:)';
                Sxxhatk = chol(Pk)';
                
                % Weight calculations
                Wfac = 1/(UKFobj.nx+UKFobj.nv+UKFobj.lambda);
                Wvec = [UKFobj.lambda ; (ones((Nsigma-1),1)*0.5)]*Wfac;
                
                % Pre-allocate new dummy variables for storage
                [xkp1sigmapt,zkp1sigmapt,xbar,zbar,Pbar,Pxz,Pzz] = initSigmaPoints(UKFobj,Nsigma);
                
                for jj = 1:Nsigma
                    % Generate sigma points
                    [xdum,vdum] = genSigmaPoint(UKFobj,xhatk,vhatk,Sxxhatk,Svvk,jj);
                    % Propagate sigma points
                    [xkp1_jj]   = dynamicProp(UKFobj,xdum,uk,vdum,tk,tkp1,k);
                    % Linearize measurements about sigma points
                    [zkp1_jj,~] = feval(UKFobj.hmodel,xkp1_jj,kp1,1);

                    % Calculate a priori state estimate and measurements
                    W_jj = Wvec(jj,1);
                    xbar = xbar + xkp1_jj*W_jj;
                    zbar = zbar + zkp1_jj*W_jj;
                    % Store sigma points for later calcs
                    xkp1sigmapt(:,jj) = xkp1_jj;
                    zkp1sigmapt(:,jj) = zkp1_jj;
                end
                
                % Modify Wvec(1,1) to be proper weighting for covariance calculations
                Wvec(1,1) = Wvec(1,1) + 1 + UKFobj.beta - UKFobj.alpha^2;

                % Compute components of covariance matrix of [xbarkp1;zbarkp1]
                for jj = 1:Nsigma
                    dxjj = xkp1sigmapt(:,jj) - xbar;
                    dzjj = zkp1sigmapt(:,jj) - zbar;
                    W_jj = Wvec(jj,1);
                    Pbar = Pbar + W_jj*(dxjj*(dxjj'));
                    Pxz  = Pxz + W_jj*(dxjj*(dzjj'));
                    Pzz  = Pzz + W_jj*(dzjj*(dzjj'));
                end
                % Finish z covariance matrix
                Pzz = Pzz + UKFobj.R;
                
                % Perform measurement update method to get a posteriori
                % state estimate, error covariance, and innovation stats.
                [xhatkp1,Pkp1,eta_nukp1] = measUpdate(UKFobj,xbar,zbar,zkp1,Pbar,Pxz,Pzz);
                
                % Store results
                kp2 = kp1 + 1;
                UKFobj.xhathist(:,kp2) = xhatkp1;
                UKFobj.Phist(:,:,kp2)  = Pkp1;
                UKFobj.eta_nuhist(kp1) = eta_nukp1;
                % Prepare for next sample
                xhatk = xhatkp1;
                Pk    = Pkp1;
                tk    = tkp1;
            end
        end
        
        function [xsigmapt,zsigmapt,xbar,zbar,Pbar,Pxz,Pzz] = initSigmaPoints(UKFobj,Nsigma)
            % Sigma point arrays
            xsigmapt = zeros(UKFobj.nx,Nsigma);
            zsigmapt = zeros(UKFobj.nz,Nsigma);
            % A priori state estimate and measurements
            xbar = zeros(UKFobj.nx,1);
            zbar = zeros(UKFobj.nz,1);
            % A priori covriance matrices
            Pbar = zeros(UKFobj.nx,UKFobj.nx);
            Pxz  = zeros(UKFobj.nx,UKFobj.nz);
            Pzz  = zeros(UKFobj.nz,UKFobj.nz);
        end
        
        % Method to generate sigma points 
        function [xdum,vdum] = genSigmaPoint(UKFobj,xhatk,vhatk,Sxxhatk,Svvk,jj)
            % Calculate variance factor
            sigmafac = sqrt(UKFobj.nx+UKFobj.nv+UKFobj.lambda);
            % Initialize sigma points
            xdum = xhatk;
            vdum = vhatk;
            % Determine sigma points based on current iteration
            if jj > 1
                if jj <= (1+UKFobj.nx)
                    xdum = xdum + sigmafac*Sxxhatk(:,(jj-1));
                elseif jj <= (1+2*UKFobj.nx)
                    xdum = xdum - sigmafac*Sxxhatk(:,(jj-(UKFobj.nx+1)));
                elseif jj <= (1+2*UKFobj.nx+UKFobj.nv)
                    vdum = vdum + sigmafac*Svvk(:,(jj-(2*UKFobj.nx+1)));
                else
                    vdum = vdum - sigmafac*Svvk(:,(jj-(2*UKFobj.nx+UKFobj.nv+1)));
                end
            end
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function [xbarkp1] = dynamicProp(UKFobj,xhatk,uk,vk,tk,tkp1,k)
            % Check model types and get sample k a priori state estimate.
            if strcmp(UKFobj.modelFlag,'CD')
                [xbarkp1,~,~] = c2dnonlinear(xhatk,uk,vk,tk,tkp1,UKFobj.nRK,UKFobj.fmodel,1);
            elseif strcmp(UKFobj.modelFlag,'DD')
                [xbarkp1,~,~] = feval(UKFobj.fmodel,xhatk,uk,vk,k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
        end
        
        % Measurement update method at sample k+1 using LMMSE equations.
        function [xhatkp1,Pkp1,eta_nukp1] = measUpdate(UKFobj,xbar,zbar,zkp1,Pbar,Pxz,Pzz)
            % Complete Kalman Filter Update by calculating the KF gain and innovations
            Wkp1      = Pxz/Pzz;
            nukp1     = zkp1 - zbar;
            eta_nukp1 = nukp1'*inv(Pzz)*nukp1;
            % Calculate the a posteriori state estimate & covariance at sample k+1.
            xhatkp1 = xbar + Wkp1*nukp1;
            Pkp1    = Pbar - Wkp1*(Pxz');
        end
    end
end