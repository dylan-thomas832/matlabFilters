%% Batch Filter Superclass
% *Author: Dylan Thomas*
%
% This class serves as an abstract superclass for implementing various 
% batch filters. It has a set of 10 default input arguments, 3 output 
% arguments, and an extra default property to store optional arguments. The
% abstract class instantiates the filters and checks to make sure the input
% arguments are sensible. Each filter subclass instance is require to 
% define the five abstract classes listed.
%

%% TODO:
%
% # Demystify beginning sample/tk = 0 problem (kinit maybe?)
% # Get rid of inv( ) warnings
% # Add continuous measurement model functionality?

%% Abstract Filter Class Deinfition
classdef (Abstract) batchFilter

%% Filter Object Properties
% *Default Inputs:*
   properties

       fmodel       % String: 
                    % 
                    % Name of the user-defined dynamics function. It 
                    % takes the time, state estimate, control vector,
                    % and process noise at sample k as an argument.
                    
       hmodel       % String:
                    %
                    % Name of the user-defined measurement function. 
                    % It takes the state estimate at sample k as an 
                    % argument.
                    
       modelFlag    % String:
                    %
                    % Model type descriptor - two options:
                    % Option 1) "CD" for a continuous-time dynamics 
                    % model andd a discrette-time measurement model
                    % Option 2) "DD" for discrete-time dynamics and 
                    % measurement models. 
                    %
                    % Continuous-time models are converted to 
                    % discrete-time models via the c2dnonlinear 
                    % function.
                    
       xhat0        % (nx)x1 vector:
                    %
                    % Initial a posteriori state estimate.
       
       P0           % (nx)x(nx) matrix:
                    %
                    % Initial a posteriori error covariance matrix.
       
       uhist        % (kmax)x(nu) array:
                    % 
                    % Control state time-history from sample k=1 to 
                    % sample k=kmax. This is assumed to be known and
                    % deterministic. This input can be an empty 
                    % array if there are no control inputs.
       
       zhist        % (kmax)x(nz) array:
                    %
                    % Measurement state time-history from sample k=1 
                    % to sample k=kmax. These are the actual 
                    % measurements taken by the system, supplied by 
                    % the user.
       
       thist        % (kmax)x1 vector:
                    %
                    % Time vector defining the discrete times for each
                    % sample from sample k=1 to sample k=kmax. This is
                    % mainly for problems where the dynamics models 
                    % are defined in continuous-time. This can be an 
                    % empty array if there is no need for defining the 
                    % times of each sample.
       
       Q            % (nv)x(nv) matrix:
                    % 
                    % Symmetric, positive definite process noise 
                    % covariance matrix.
       
       R            % (nz)x(nz) matrix:
                    % 
                    % Symmetric, positive definite measurement noise 
                    % covariance matrix.
                    
       optArgs      % cell array:
                    %
                    % Extra/optional arguments for each filter class.
   end
   
%%%
% *Default Outputs:*
   properties

       xhathist     % (nx)x(kmax+1) array:
                    % 
                    % A posteriori state estimate time-history.
       
       Phist        % (nx)x(nx)x(kmax+1) array:
                    %
                    % A posteriori error covariance matrix 
                    % time-history.
       
       eta_nuhist   % (kmax)x1 vector:
                    % 
                    % Innovation statistics time-history.
   end

%%% 
% *Dependent Properties:*
   properties (Dependent, Access = protected)
       nx           % Number of states
       nu           % Number of control inputs
       nv           % Number of process noise states
       nz           % Number of measurement states
       kmax         % Number of discrete samples
   end
   
%% Filter Object Methods  
% *Default Constructor:*
   methods (Access = protected)
       % Batch Filter constructor
       function Filterobj = batchFilter(fmodel,hmodel,modelFlag,xhat0,P0,uhist,zhist,thist,Q,R,optArgs)
           % Check inputs
           inputsCheck(modelFlag,xhat0,P0,uhist,zhist,thist,R);
           % Assign properties to Filter Object
           Filterobj.fmodel = fmodel;
           Filterobj.hmodel = hmodel;
           Filterobj.modelFlag = modelFlag;
           Filterobj.xhat0 = xhat0;
           Filterobj.P0 = P0;
           Filterobj.uhist = uhist;
           Filterobj.zhist = zhist;
           Filterobj.thist = thist;
           Filterobj.Q = Q;
           Filterobj.R = R;
           Filterobj.optArgs = optArgs;
       end
   end
%%%  
% *Dependent 'Get' Methods:*
   methods
       function val = get.nx(Filterobj)
           val = size(Filterobj.xhat0,1);
       end
       function val = get.nu(Filterobj)
           val = size(Filterobj.uhist,2);
       end
       function val = get.nv(Filterobj)
           val = size(Filterobj.Q,1);
       end
       function val = get.nz(Filterobj)
           val = size(Filterobj.zhist,2);
       end
       function val = get.kmax(Filterobj)
           val = size(Filterobj.zhist,1);
       end
   end
%%%  
% *Abstract Methods:*
   methods (Abstract)
       argumentsCheck(Filterobj)
       initFilter(Filterobj)
       doFilter(Filterobj)
       dynamicProp(Filterobj)
       measUpdate(Filterobj)
   end
   
end

%% Helper Function
%
function inputsCheck(modelFlag,xhat0,P0,uhist,zhist,thist,R)
% This helper function checks that the inputs are the right type and size.

% Get problem dimensions for easier checking
Nx = size(xhat0,1);
Nz = size(R,1);
Kmax = size(zhist,1);

% Assertions to ensure correct input arguments to the filter.

% Assert that the model types are correct
assert((strcmp(modelFlag,'CD')||strcmp(modelFlag,'DD')),...
    'batchFilter:instantiation',...
    'The model-type flag is incorrect')
% Assert that the error covariance matrix is sized correctly
assert(((size(P0,1)==Nx)&&(size(P0,2)==Nx)),...
    'batchFilter:instantiation',...
    'Initial error covariance does not have the correct dimensions')
% Assert that the measurement vector and covariance are correct.
assert((size(zhist,2)==Nz),...
    'batchFilter:instantiation',...
    'Measurement vector and measurement noise covariance do not match in size')
% Assert that the control history and time vectors are correct
if ~isempty(uhist)
    assert((size(uhist,1)==Kmax),...
        'batchFilter:instantiation',...
        'Measurement and control time-histories should be the same length')
end
if ~isempty(thist)
    assert((size(thist,1)==Kmax),...
        'batchFilter:instantiation',...
        'Measurement time-histories and time vector should be the same length')
end

end