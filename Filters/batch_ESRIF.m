%% Extended Square-Root Information Filter (ESRIF) Class
% *Author: Dylan Thomas*
%
% This class implements an extended square-root information filter from
% sample k=0 to sample k=kmax.
%
% *Note: the user must instantiate the class with no properties, as a
% structure with fieldnames matching all input properties, or all input
% properties separately.*

%% TODO:
%
% # Figure out how to add in LTI $F$, $G$, $\Gamma$, $H$ matrices for cont & disc models

%% ESRIF Class Deinfition
classdef batch_ESRIF < batchFilter
% Inherits batchFilter abstract class

%% ESRIF Properties
% *Inputs:*
    properties 

        nRK         % Scalar >= 5:
                    %
                    % The Runge Kutta iterations to perform for 
                    % coverting dynamics model from continuous-time 
                    % to discrete-time. Default value is 10 RK iterations.
    end

%%%
% *Derived Properties:*
    properties
                
        Rvvk        % (nv)x(nv) Matrix:
                    % 
                    % The square-root information process noise 
                    % covariance.
                    
        Rxxk        % (nx)x(nx) Matrix:
                    %
                    % The square-root information state error a 
                    % posteriori covariance matrix at sample k.
        
        Ra          % (nz)x(nz) Matrix:
                    % 
                    % The transformation matrix to ensure that the
                    % measurement noise is zero mean with identity
                    % covariance.
        
        Rainvtr     % (nz)x(nz) Matrix:
                    % 
                    % Ra inversed and transposed - for convenience.
        
        zahist      % (kmax)x(nz) Array:
                    %
                    % The transformed measurement state time-history 
                    % which ensures zero mean, identity covariance 
                    % noise.
    end
    
%% ESRIF Methods
    methods
        % ESRIF constructor
        % Input order:
        % fmodel,hmodel,modelFlag,kInit,xhatInit,PInit,uhist,zhist,thist,Q,R,nRK
        function ESRIFobj = batch_ESRIF(varargin)
            % Prepare for superclass constructor
            if nargin == 0
                fprintf('Instantiating empty batch ESRIF class\n\n')
                super_args = {};
            elseif nargin == 1
                fprintf('Instantiating batch ESRIF class\n\n')
                super_args = varargin{1};
            elseif nargin < 11
                error('Not enough input arguments')
            else
                fprintf('Instantiating batch ESRIF class\n\n')
                super_args = cell(1,12);
                for jj = 1:12
                    super_args{jj} = varargin{jj};
                end
            end
            % batchFilter superclass constructor
            ESRIFobj@batchFilter(super_args{:});
            % Only do if intantiated class is not empty
            if nargin > 0
                % Extra argument checker method
                ESRIFobj = argumentsCheck(ESRIFobj);
            end
        end
        
        % This method checks the extra input arguments for ESRIF class
        function ESRIFobj = argumentsCheck(ESRIFobj)
            % Switch on number of extra arguments.
            switch length(ESRIFobj.optArgs)
                case 0
                    ESRIFobj.nRK = 10;
                case 1
                    ESRIFobj.nRK = ESRIFobj.optArgs{1};
                otherwise
                    error('Too many input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if ESRIFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            end
        end
        
        % This method initializes the ESRIF class filter
        function [ESRIFobj,xhatk,tk,vk] = initFilter(ESRIFobj)
            % Setup the output arrays
            ESRIFobj.xhathist     = zeros(ESRIFobj.nx,ESRIFobj.kmax+1);
            ESRIFobj.Phist        = zeros(ESRIFobj.nx,ESRIFobj.nx,ESRIFobj.kmax+1);
            ESRIFobj.eta_nuhist   = zeros(size(ESRIFobj.thist));
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            xhatk                                    = ESRIFobj.xhatInit;
            ESRIFobj.xhathist(:,ESRIFobj.kInit+1)    = ESRIFobj.xhatInit;
            ESRIFobj.Phist(:,:,ESRIFobj.kInit+1)     = ESRIFobj.PInit;
            vk                                       = zeros(ESRIFobj.nv,1);
            % Make sure correct initial tk is used.
            if ESRIFobj.kInit == 0
                tk = 0;
            else
                tk = ESRIFobj.thist(ESRIFobj.kInit);
            end
            
            % Determine the square-root information matrix for the process 
            % noise, and transform the measurements to have an error with 
            % an identity covariance.
            ESRIFobj.Rvvk = inv(chol(ESRIFobj.Q)');
            ESRIFobj.Ra = chol(ESRIFobj.R);
            ESRIFobj.Rainvtr = inv(ESRIFobj.Ra');
            ESRIFobj.zahist = ESRIFobj.zhist*(ESRIFobj.Rainvtr');
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            ESRIFobj.Rxxk = inv(chol(ESRIFobj.PInit)');
            
        end
        
        % This method performs ESRIF class filter estimation
        function ESRIFobj = doFilter(ESRIFobj)
            % Filter initialization method
            [ESRIFobj,xhatk,tk,vk] = initFilter(ESRIFobj);
            
            % Main filter loop.
            for k = ESRIFobj.kInit:(ESRIFobj.kmax-1)
                % Prepare loop
                kp1 = k+1;
                tkp1 = ESRIFobj.thist(kp1);
                uk = ESRIFobj.uhist(kp1,:)';
                
                % Perform dynamic propagation and measurement update
                [xbarkp1,zetabarxkp1,Rbarxxkp1] = dynamicProp(ESRIFobj,xhatk,uk,vk,tk,tkp1,k);
                [zetaxkp1,Rxxkp1,zetarkp1] = measUpdate(ESRIFobj,xbarkp1,zetabarxkp1,Rbarxxkp1,kp1);
                
                % Compute the state estimate and covariance at sample k + 1
                Rxxkp1inv = inv(Rxxkp1);
                xhatkp1 = Rxxkp1\zetaxkp1;
                Pkp1 = Rxxkp1inv*(Rxxkp1inv');
                % Store results
                kp2 = kp1 + 1;
                ESRIFobj.xhathist(:,kp2) = xhatkp1;
                ESRIFobj.Phist(:,:,kp2) = Pkp1;
                ESRIFobj.eta_nuhist(kp1) = zetarkp1'*zetarkp1;
                % Prepare for next sample
                ESRIFobj.Rxxk = Rxxkp1;
                xhatk = xhatkp1;
                tk = tkp1;

            end
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function [xbarkp1,zetabarxkp1,Rbarxxkp1] = dynamicProp(ESRIFobj,xhatk,uk,vk,tk,tkp1,k)
            % Check model types and get sample k a priori state estimate.
            if strcmp(ESRIFobj.modelFlag,'CD')
                [xbarkp1,F,Gamma] = c2dnonlinear(xhatk,uk,vk,tk,tkp1,ESRIFobj.nRK,ESRIFobj.fmodel,1);
            elseif strcmp(ESRIFobj.modelFlag,'DD')
                [xbarkp1,F,Gamma] = feval(ESRIFobj.fmodel,xhatk,uk,vk,k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
            % QR Factorize
            Rbig = [ESRIFobj.Rvvk,      zeros(ESRIFobj.nv,ESRIFobj.nx); ...
                  (-ESRIFobj.Rxxk*(F\Gamma)),         ESRIFobj.Rxxk/F];
            [Taktr,Rdum] = qr(Rbig);
            Tak = Taktr';
            zdum = Tak*[zeros(ESRIFobj.nv,1);ESRIFobj.Rxxk*(F\xbarkp1)];
            % Retrieve SRIF terms at k+1 sample
            idumxvec = ((ESRIFobj.nv+1):(ESRIFobj.nv+ESRIFobj.nx))';
            Rbarxxkp1 = Rdum(idumxvec,idumxvec);
            zetabarxkp1 = zdum(idumxvec,1);
        end
        
        % Measurement update method at sample k+1.
        function [zetaxkp1,Rxxkp1,zetarkp1] = measUpdate(ESRIFobj,xbarkp1,zetabarxkp1,Rbarxxkp1,kp1)
            % Linearized at sample k+1 a priori state estimate.
            [zbarkp1,H] = feval(ESRIFobj.hmodel,xbarkp1,kp1,1);
            % Transform ith H(k) matrix and non-homogeneous measurement terms
            Ha = ESRIFobj.Rainvtr*H;
            zEKF = ESRIFobj.zahist(kp1,:)' - ESRIFobj.Rainvtr*zbarkp1 + Ha*xbarkp1;
            % QR Factorize
            [Tbkp1tr,Rdum] = qr([Rbarxxkp1;Ha]);
            Tbkp1 = Tbkp1tr';
            zdum = Tbkp1*[zetabarxkp1;zEKF];
            % Retrieve k+1 SRIF terms
            idumxvec = (1:ESRIFobj.nx)';
            Rxxkp1 = Rdum(idumxvec,idumxvec);
            zetaxkp1 = zdum(idumxvec,1);
            zetarkp1 = zdum(ESRIFobj.nx+1:end);
        end
    end
end