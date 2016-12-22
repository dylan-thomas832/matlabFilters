%% Square-Root Information Filter (SRIF) Class
% *Author: Dylan Thomas*
%
% This class implements an square-root information filter.
%
% *Note: the user must instantiate the class with no properties, as a
% structure with fieldnames matching all input properties, or all input
% properties separately.*

%% TODO:
%
% # Figure out how to add in LTI $F$, $G$, $\Gamma$, $H$ matrices for cont & disc models
% # Demystify beginning sample/tk = 0 problem (kinit maybe?)
% # Get rid of inv( ) warnings
% # Add continuous measurement model functionality?

%% SRIF Class Deinfition
classdef batch_SRIF < batchFilter
% Inherits batchFilter abstract class

%% SRIF Properties
% *Inputs:*
    properties 

        nRK         % scalar:
                    %
                    % The Runge Kutta iterations to perform for 
                    % coverting dynamics model from continuous-time 
                    % to discrete-time. Default value is 20 RK 
                    % iterations.
    end

%%%
% *Derived Properties:*
    properties
                
        Rvvk        % (nv)x(nv) matrix:
                    % 
                    % The square-root information process noise 
                    % covariance.
                    
        Rxxk        % (nx)x(nx) matrix:
                    %
                    % The square-root information state error a 
                    % posteriori covariance matrix at sample k.
        
        Ra          % (nz)x(nz) matrix:
                    % 
                    % The transformation matrix to ensure that the
                    % measurement noise is zero mean with identity
                    % covariance.
        
        Rainvtr     % (nz)x(nz) matrix:
                    % 
                    % Ra inversed and transposed - for convenience.
        
        zahist      % (kmax)x(nz) array:
                    %
                    % The transformed measurement state time-history 
                    % which ensures zero mean, identity covariance 
                    % noise.
    end
    
%% SRIF Methods
    methods
        % SRIF constructor
        % Input order:
        % fmodel,hmodel,modelFlag,kInit,xhatInit,PInit,uhist,zhist,thist,Q,R,nRK
        function SRIFobj = batch_SRIF(varargin)
            % Prepare for superclass constructor
            if nargin == 0
                fprintf('Instantiating empty batch SRIF class\n\n')
                super_args = {};
            elseif nargin == 1
                fprintf('Instantiating batch SRIF class\n\n')
                super_args = varargin{1};
            elseif nargin < 11
                error('Not enough input arguments')
            else
                fprintf('Instantiating batch SRIF class\n\n')
                super_args = cell(1,12);
                for jj = 1:12
                    super_args{jj} = varargin{jj};
                end
            end
            % batchFilter superclass constructor
            SRIFobj@batchFilter(super_args{:});
            % Only do if intantiated class is not empty
            if nargin > 0
                % Extra argument checker method
                SRIFobj = argumentsCheck(SRIFobj);
            end
        end
        
        % This method checks the extra input arguments for SRIF class
        function SRIFobj = argumentsCheck(SRIFobj)
            % Switch on number of extra arguments.
            switch length(SRIFobj.optArgs)
                case 0
                    SRIFobj.nRK = 20;
                case 1
                    SRIFobj.nRK = SRIFobj.optArgs{1};
                otherwise
                    error('Too many input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if SRIFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            end
        end
        
        % This method initializes the SRIF class filter
        function [SRIFobj,xhatk,tk,vk] = initFilter(SRIFobj)
            % Setup the output arrays
            SRIFobj.xhathist     = zeros(SRIFobj.nx,SRIFobj.kmax+1);
            SRIFobj.Phist        = zeros(SRIFobj.nx,SRIFobj.nx,SRIFobj.kmax+1);
            SRIFobj.eta_nuhist   = zeros(size(SRIFobj.thist));
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            xhatk                                  = SRIFobj.xhatInit;
            SRIFobj.xhathist(:,SRIFobj.kInit+1)    = SRIFobj.xhatInit;
            SRIFobj.Phist(:,:,SRIFobj.kInit+1)     = SRIFobj.PInit;
            vk                                     = zeros(SRIFobj.nv,1);
            % Make sure correct initial tk is used.
            if SRIFobj.kInit == 0
                tk = 0;
            else
                tk = SRIFobj.thist(SRIFobj.kInit);
            end
        end
        
        % This method performs SRIF class filter estimation
        function SRIFobj = doFilter(SRIFobj)
            % Filter initialization method
            [SRIFobj,xhatk,tk,vk] = initFilter(SRIFobj);
            
            % Determine the square-root information matrix for the process 
            % noise, and transform the measurements to have an error with 
            % an identity covariance.
            SRIFobj.Rvvk = inv(chol(SRIFobj.Q)');
            SRIFobj.Ra = chol(SRIFobj.R);
            SRIFobj.Rainvtr = inv(SRIFobj.Ra');
            SRIFobj.zahist = SRIFobj.zhist*(SRIFobj.Rainvtr');
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            SRIFobj.Rxxk = inv(chol(SRIFobj.PInit)');
            
            % Main filter loop.
            for k = SRIFobj.kInit:(SRIFobj.kmax-1)
                % Prepare loop
                kp1 = k+1;
                tkp1 = SRIFobj.thist(kp1);
                uk = SRIFobj.uhist(kp1,:)';
                
                % Perform dynamic propagation and measurement update
                [xbarkp1,zetabarxkp1,Rbarxxkp1] = ...
                    dynamicProp(SRIFobj,xhatk,uk,vk,tk,tkp1,k);
                [zetaxkp1,Rxxkp1,zetarkp1] = ...
                    measUpdate(SRIFobj,xbarkp1,zetabarxkp1,Rbarxxkp1,kp1);
                
                % Compute the state estimate and covariance at sample k + 1
                Rxxkp1inv = inv(Rxxkp1);
                xhatkp1 = Rxxkp1\zetaxkp1;
                Pkp1 = Rxxkp1inv*(Rxxkp1inv');
                % Store results
                kp2 = kp1 + 1;
                SRIFobj.xhathist(:,kp2) = xhatkp1;
                SRIFobj.Phist(:,:,kp2) = Pkp1;
                SRIFobj.eta_nuhist(kp1) = zetarkp1'*zetarkp1;
                % Prepare for next sample
                SRIFobj.Rxxk = Rxxkp1;
                xhatk = xhatkp1;
                tk = tkp1;

            end
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function [xbarkp1,zetabarxkp1,Rbarxxkp1] = dynamicProp(SRIFobj,xhatk,uk,vk,tk,tkp1,k)
            % Check model types and get sample k a priori state estimate.
            if strcmp(SRIFobj.modelFlag,'CD')
                [xbarkp1,F,Gamma] = c2dnonlinear(xhatk,uk,vk,tk,tkp1,SRIFobj.nRK,SRIFobj.fmodel,1);
            elseif strcmp(SRIFobj.modelFlag,'DD')
                [xbarkp1,F,Gamma] = feval(SRIFobj.fmodel,xhatk,uk,vk,k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
            Finv = inv(F);
            FinvGamma = F\Gamma;
            % QR Factorize
            Rbig = [SRIFobj.Rvvk,      zeros(SRIFobj.nv,SRIFobj.nx); ...
                  (-SRIFobj.Rxxk*FinvGamma),         SRIFobj.Rxxk/F];
            [Taktr,Rdum] = qr(Rbig);
            Tak = Taktr';
            zdum = Tak*[zeros(SRIFobj.nv,1);SRIFobj.Rxxk*Finv*xbarkp1];
            % Retrieve SRIF terms at k+1 sample
            idumxvec = [(SRIFobj.nv+1):(SRIFobj.nv+SRIFobj.nx)]';
            Rbarxxkp1 = Rdum(idumxvec,idumxvec);
            zetabarxkp1 = zdum(idumxvec,1);
        end
        
        % Measurement update method at sample k+1.
        function [zetaxkp1,Rxxkp1,zetarkp1] = measUpdate(SRIFobj,xbarkp1,zetabarxkp1,Rbarxxkp1,kp1)
            % Linearized at sample k+1 a priori state estimate.
            [zbarkp1,H] = feval(SRIFobj.hmodel,xbarkp1,kp1,1);
            % Transform ith H(k) matrix and non-homogeneous measurement terms
            Ha = SRIFobj.Rainvtr*H;
            zEKF = SRIFobj.zahist(kp1,:)' - SRIFobj.Rainvtr*zbarkp1 + Ha*xbarkp1;
            % QR Factorize
            [Tbkp1tr,Rdum] = qr([Rbarxxkp1;Ha]);
            Tbkp1 = Tbkp1tr';
            zdum = Tbkp1*[zetabarxkp1;zEKF];
            % Retrieve k+1 SRIF terms
            idumxvec = [1:SRIFobj.nx]';
            Rxxkp1 = Rdum(idumxvec,idumxvec);
            zetaxkp1 = zdum(idumxvec,1);
            zetarkp1 = zdum(SRIFobj.nx+1:end);
        end
    end
end