%% Iterated, Extended Square-Root Information Filter (iESRIF) Class
% *Author: Dylan Thomas*
%
% This class implements an iterated, extended square-root information 
% filter.
%
% *Note: the user must instantiate the class with no properties, as a
% structure with fieldnames matching all input properties, or all input
% properties separately.*

%% iESRIF Class Deinfition
classdef batch_iESRIF < batchFilter
% Inherits batchFilter abstract class

%% iESRIF Properties
% *Inputs:*
    properties 

        nRK         % Scalar >= 5:
                    %
                    % The Runge Kutta iterations to perform for 
                    % coverting dynamics model from continuous-time 
                    % to discrete-time. Default value is 10 RK 
                    % iterations.
                    
        Niter       % Scalar >= 1:
                    %
                    % The number of measurement update iterations to
                    % run through. Default value is 5.
        
        alphalim    % Scalar between 1 and 0:
                    %
                    % The lower limit for the Gauss-Newton step
                    % change. This limit decreases until the cost
                    % function is decreased for the current state
                    % estimate. If the cost function is not
                    % decreased, then the alpha value is decreased
                    % to create a different state estimate. Default
                    % value is 0.01.
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
    
%% iESRIF Methods
    methods
        % iESRIF constructor
        % Input order:
        % fmodel,hmodel,modelFlag,kInit,xhatInit,PInit,uhist,zhist,thist,Q,R,nRK,Niter,alphalim
        function iESRIFobj = batch_iESRIF(varargin)
            % Prepare for superclass constructor
            if nargin == 0
                fprintf('Instantiating empty batch iESRIF class\n\n')
                super_args = {};
            elseif nargin == 1
                fprintf('Instantiating batch iESRIF class\n\n')
                super_args = varargin{1};
            elseif nargin < 11
                error('Not enough input arguments')
            else
                fprintf('Instantiating batch iESRIF class\n\n')
                super_args = cell(1,12);
                for jj = 1:12
                    super_args{jj} = varargin{jj};
                end
            end
            % batchFilter superclass constructor
            iESRIFobj@batchFilter(super_args{:});
            % Only do if intantiated class is not empty
            if nargin > 0
                % Extra argument checker method
                iESRIFobj = argumentsCheck(iESRIFobj);
            end
        end
        
        % This method checks the extra input arguments for iESRIF class
        function iESRIFobj = argumentsCheck(iESRIFobj)
            % Switch on number of extra arguments.
            switch length(iESRIFobj.optArgs)
                case 0
                    iESRIFobj.nRK = 10;
                    iESRIFobj.Niter = 5;
                    iESRIFobj.alphalim = 0.01;
                case 1
                    iESRIFobj.nRK = iESRIFobj.optArgs{1};
                    iESRIFobj.Niter = 5;
                    iESRIFobj.alphalim = 0.01;
                case 2
                    iESRIFobj.nRK = iESRIFobj.optArgs{1};
                    iESRIFobj.Niter = iESRIFobj.optArgs{2};
                    iESRIFobj.alphalim = 0.01;
                case 3
                    iESRIFobj.nRK = iESRIFobj.optArgs{1};
                    iESRIFobj.Niter = iESRIFobj.optArgs{2};
                    iESRIFobj.alphalim = iESRIFobj.optArgs{3};
                otherwise
                    error('Too many input arguments')
            end
            % Ensures extra input arguments have sensible values.
            if iESRIFobj.nRK < 5
                error('Number of Runge-Kutta iterations should be larger than 5')
            elseif (iESRIFobj.Niter > 1000 || iESRIFobj.Niter < 1)
                error('The number of measurement update iterations is too large')
            elseif (iESRIFobj.alphalim >= 1 || iESRIFobj.alphalim <= 0)
                error('Lower bound on alpha should be between 0 and 1')
            end
        end
        
        % This method initializes the iESRIF class filter
        function [iESRIFobj,xhatk,tk,vk] = initFilter(iESRIFobj)
            % Setup the output arrays
            iESRIFobj.xhathist     = zeros(iESRIFobj.nx,iESRIFobj.kmax+1);
            iESRIFobj.Phist        = zeros(iESRIFobj.nx,iESRIFobj.nx,iESRIFobj.kmax+1);
            iESRIFobj.eta_nuhist   = zeros(size(iESRIFobj.thist));
            
            % Initialize quantities for use in the main loop and store the 
            % first a posteriori estimate and its error covariance matrix.
            xhatk                                      = iESRIFobj.xhatInit;
            iESRIFobj.xhathist(:,iESRIFobj.kInit+1)    = iESRIFobj.xhatInit;
            iESRIFobj.Phist(:,:,iESRIFobj.kInit+1)     = iESRIFobj.PInit;
            vk                                         = zeros(iESRIFobj.nv,1);
            % Make sure correct initial tk is used.
            if iESRIFobj.kInit == 0
                tk = 0;
            else
                tk = iESRIFobj.thist(iESRIFobj.kInit);
            end
            
            % Determine the square-root information matrix for the process
            % noise, and transform the measurements to have an error with
            % an identity covariance.
            iESRIFobj.Rvvk      = inv(chol(iESRIFobj.Q)');
            iESRIFobj.Ra        = chol(iESRIFobj.R);
            iESRIFobj.Rainvtr   = inv(iESRIFobj.Ra');
            iESRIFobj.zahist    = iESRIFobj.zhist*(iESRIFobj.Rainvtr');
            
            % Initialize quantities for use in the main loop and store the
            % first a posteriori estimate and its error covariance matrix.
            iESRIFobj.Rxxk      = inv(chol(iESRIFobj.PInit)');
        end
        
        % This method performs iESRIF class filter estimation
        function iESRIFobj = doFilter(iESRIFobj)
            % Filter initialization method
            [iESRIFobj,xhatk,tk,vk] = initFilter(iESRIFobj);
            
            % Main filter loop.
            for k = iESRIFobj.kInit:(iESRIFobj.kmax-1)
                % Prepare loop
                kp1 = k+1;
                tkp1 = iESRIFobj.thist(kp1);
                uk = iESRIFobj.uhist(kp1,:)';
                
                % Perform dynamic propagation and measurement update
                [xbarkp1,zetabarxkp1,Rbarxxkp1] = dynamicProp(iESRIFobj,xhatk,uk,vk,tk,tkp1,k);
                [zetaxkp1,Rxxkp1,zetarkp1] = measUpdate(iESRIFobj,xbarkp1,zetabarxkp1,Rbarxxkp1,kp1);
                
                % Compute the state estimate and covariance at sample k + 1
                xhatkp1 = Rxxkp1\zetaxkp1;
                Pkp1 = Rxxkp1\inv(Rxxkp1);
                % Store results
                kp2 = kp1 + 1;
                iESRIFobj.xhathist(:,kp2) = xhatkp1;
                iESRIFobj.Phist(:,:,kp2) = Pkp1;
                iESRIFobj.eta_nuhist(kp1) = zetarkp1'*zetarkp1;
                % Prepare for next sample
                iESRIFobj.Rxxk = Rxxkp1;
                xhatk = xhatkp1;
                tk = tkp1;

            end
        end
        
        % Dynamic propagation method, from sample k to sample k+1.
        function [xbarkp1,zetabarxkp1,Rbarxxkp1] = dynamicProp(iESRIFobj,xhatk,uk,vk,tk,tkp1,k)
            % Check model types and get sample k a priori state estimate.
            if strcmp(iESRIFobj.modelFlag,'CD')
                [xbarkp1,F,Gamma] = c2dnonlinear(xhatk,uk,vk,tk,tkp1,iESRIFobj.nRK,iESRIFobj.fmodel,1);
            elseif strcmp(iESRIFobj.modelFlag,'DD')
                [xbarkp1,F,Gamma] = feval(iESRIFobj.fmodel,xhatk,uk,vk,k);
            else
                error('Incorrect flag for the dynamics-measurement models')
            end
            % QR Factorize
            Rbig = [iESRIFobj.Rvvk, zeros(iESRIFobj.nv,iESRIFobj.nx); ...
                  (-iESRIFobj.Rxxk*(F\Gamma)), iESRIFobj.Rxxk/F];
            [Taktr,Rdum] = qr(Rbig);
            Tak = Taktr';
            zdum = Tak*[zeros(iESRIFobj.nv,1);iESRIFobj.Rxxk*(F\xbarkp1)];
            % Retrieve SRIF terms at k+1 sample
            idumxvec = ((iESRIFobj.nv+1):(iESRIFobj.nv+iESRIFobj.nx))';
            Rbarxxkp1 = Rdum(idumxvec,idumxvec);
            zetabarxkp1 = zdum(idumxvec,1);
        end
        
        % Measurement update method at sample k+1.
        function [zetaxkp1,Rxxkp1,zetarkp1] = measUpdate(iESRIFobj,xbarkp1,zetabarxkp1,Rbarxxkp1,kp1)
            % Regular ESRIF measurement update
            
            % Get current transformed measurement
            zakp1 = iESRIFobj.zahist(kp1,:)';
            % Linearized at sample k+1 a priori state estimate.
            [zbarkp1,H] = feval(iESRIFobj.hmodel,xbarkp1,kp1,1);
            % Transform ith H(k) matrix and non-homogeneous measurement terms
            Ha = iESRIFobj.Rainvtr*H;
            zEKF = zakp1 - iESRIFobj.Rainvtr*zbarkp1 + Ha*xbarkp1;
            % QR Factorize
            [Tbkp1tr,Rdum] = qr([Rbarxxkp1;Ha]);
            Tbkp1 = Tbkp1tr';
            zdum = Tbkp1*[zetabarxkp1;zEKF];
            % Retrieve k+1 SRIF terms
            idumxvec = (1:iESRIFobj.nx)';
            Rxxkp1 = Rdum(idumxvec,idumxvec);
            zetaxkp1 = zdum(idumxvec,1);
            zetarkp1 = zdum(iESRIFobj.nx+1:end);
            
            % Check if MAP iterations are to be performed
            if iESRIFobj.Niter > 1
                % Initialize the measurement update iteration-loop
                xhati = xbarkp1;
                % Define cost function to minimize
                Jc = @(xkp1,zetar) (Rbarxxkp1*xkp1-zetabarxkp1)'*(Rbarxxkp1*xkp1-zetabarxkp1)+zetar'*zetar;
                % Loop through Niter iterations
                for ii = 2:iESRIFobj.Niter
                    % Linearize at ith MAP state estimate.
                    [zbari,Hi] = feval(iESRIFobj.hmodel,xhati,kp1,1);
                    % Transform ith H(k) matrix and non-homogeneous measurement terms
                    Hai = iESRIFobj.Rainvtr*Hi;
                    zEKFi = zakp1 - iESRIFobj.Rainvtr*zbari + Hai*xhati;
                    % QR Factorize
                    [Tbitr,Rdum] = qr([Rbarxxkp1;Hai]);
                    Tbi = Tbitr';
                    zdumi = Tbi*[zetabarxkp1;zEKFi];
                    % Retrieve i+1 SRIF terms
                    Rxxi = Rdum(idumxvec,idumxvec);
                    zetaxi = zdumi(idumxvec,1);
                    zetari = zdumi(iESRIFobj.nx+1:end);
                    % Get 
                    xhatip1 = Rxxi\zetaxi;
                    % Calculate cost function at current state estimate
                    Jcold = Jc(xhati,zetari);
                    
                    % NOTE: The calculated (i+1)th MAP state estimate may
                    % not necessarily decrease the cost function, so we
                    % perform a loop with Gauss-Newton steps to ensure
                    % either that the cost function is decreased, or that
                    % the lower limit of our step is reached.
                    
                    
                    % Initialize Gauss-Newton loop
                    alpha = 1; Jcnew = Jcold;
                    while (alpha >= iESRIFobj.alphalim && Jcnew >= Jcold)
                        % Gauss-Newton step to find new MAP state estimate
                        xhatip1new = xhati + alpha*(xhatip1-xhati);
                        % Re-linearize about new state estimate @ (i+1)th step
                        [zbarip1,Hip1] = feval(iESRIFobj.hmodel,xhatip1new,kp1,1);
                        % Transform ith H(k) matrix and non-homogeneous measurement terms
                        Haip1 = iESRIFobj.Rainvtr*Hip1;
                        zEKFip1 = zakp1 - iESRIFobj.Rainvtr*zbarip1 + Haip1*xhatip1new;
                        % QR Factorize
                        [Tbip1tr,Rdum] = qr([Rbarxxkp1;Haip1]);
                        Tbip1 = Tbip1tr';
                        zdum = Tbip1*[zetabarxkp1;zEKFip1];
                        % Retrieve k+1 SRIF terms
                        Rxxip1new = Rdum(idumxvec,idumxvec);
                        zetaxip1new = zdum(idumxvec,1);
                        zetarip1new = zdum(iESRIFobj.nx+1:end);
                        % Calculate new cost function at MAP state estimate
                        Jcnew = Jc(xhatip1new,zetarip1new);
                        % Decrease alpha
                        alpha = alpha/2;
                    end
                    % Debug print
%                     fprintf('alpha at term: %4.5f\n',alpha)
                    % Check norm condition so that iterations aren't wasteful
                    if (norm(xhati-xhatip1new)/norm(xhati) < 1e-12)
                        % Debug print
%                         fprintf('norm condition reached at iter: %i\n',ii)
                        break
                    end
                    % Update for next loop
                    xhati = xhatip1new;
                end
                
                % The a posteriori error covariance, state estimate, &
                % residual
                Rxxkp1 = Rxxip1new;
                zetaxkp1 = zetaxip1new;
                zetarkp1 = zetarip1new;
            end
        end
    end
end