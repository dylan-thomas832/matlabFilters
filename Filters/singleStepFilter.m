%% Single Step Filter Superclass
% *Author: Dylan Thomas*
%
% This class serves as an abstract superclass for implementing various 
% single step filters. It has a set of 12 default input arguments, 3 output 
% arguments, and an extra default property to store optional arguments. The
% abstract class instantiates the filters and checks to make sure the input
% arguments are sensible. Each filter subclass instance is required to 
% define the five abstract classes listed.
%
% *Note: the user must instantiate the class with no properties, as a
% structure with fieldnames matching all input properties, or all input
% properties separately.*

%% TODO:
%
% # Add option for variable Q and R?
% # Add continuous measurement model functionality?? (maybe not?)

%% Abstract Filter Class Definition
classdef (Abstract) singleStepFilter

%% Default Inputs
% These are the properties that every batchFilter subclass takes in from
% the class instantiation.
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
                    
       kInit        % Scalar Integer:
                    %
                    % Initial sample to begin filtering. This is depends on
                    % the initial xhat and P.
                    
       xhatInit     % (nx)x1 Vector:
                    %
                    % Initial a posteriori state estimate.
       
       PInit        % (nx)x(nx) Matrix:
                    %
                    % Initial a posteriori error covariance matrix.
       
       uhist        % (kmax)x(nu) Array:
                    % 
                    % Control state time-history from sample k=1 to 
                    % sample k=kmax. This is assumed to be known and
                    % deterministic. This input can be an empty 
                    % array if there are no control inputs.
       
       zhist        % (kmax)x(nz) Array:
                    %
                    % Measurement state time-history from sample k=1 
                    % to sample k=kmax. These are the actual 
                    % measurements taken by the system, supplied by 
                    % the user.
       
       thist        % (kmax)x1 Vector:
                    %
                    % Time vector defining the discrete times for each
                    % sample from sample k=1 to sample k=kmax. This is
                    % mainly for problems where the dynamics models 
                    % are defined in continuous-time. This can be an 
                    % empty array if there is no need for defining the 
                    % times of each sample.
       
       Q            % (nv)x(nv) Matrix:
                    % 
                    % Symmetric, positive definite process noise 
                    % covariance matrix. Assumed constant for all k.
       
       R            % (nz)x(nz) Matrix:
                    % 
                    % Symmetric, positive definite measurement noise 
                    % covariance matrix. Assumed constant for all k.
                    
       optArgs      % Cell Array:
                    %
                    % Extra/optional arguments for each filter class.
   end
   
%% Default Outputs
% These are the default properties which all batchFilter subclasses
% calculate and store as outputs of their algorithm. More may be added.
   properties

       xhathist     % (nx)x(kmax+1) Array:
                    % 
                    % A posteriori state estimate time-history.
       
       Phist        % (nx)x(nx)x(kmax+1) Array:
                    %
                    % A posteriori error covariance matrix 
                    % time-history.
       
       eta_nuhist   % (kmax)x1 Vector:
                    % 
                    % Innovation statistics time-history.
   end

%% Dependent Properties
% These are general properties which every batchfilter subclass may use,
% but are dependent on the Input Properties.
   properties (Dependent, Access = protected)
       nx           % Number of states
       nu           % Number of control inputs
       nv           % Number of process noise states
       nz           % Number of measurement states
       kmax         % Number of discrete samples
   end

%% Default Constructor
% This method is the default constructor for all batchFilter subclasses
   methods (Access = protected)
       % Batch Filter constructor
       function Filterobj = batchFilter(varargin)
           % If user inputs properties inside a structure
           if nargin == 1
               % Get input structure containing inputs
               in = varargin{1};
               % Check inputs
               inputsCheck(in.fmodel,in.hmodel,in.modelFlag,in.kInit,in.xhatInit,in.PInit,in.uhist,in.zhist,in.thist,in.Q,in.R);
               % Assign inputs to Filter Object properties
               Filterobj.fmodel    = in.fmodel;
               Filterobj.hmodel    = in.hmodel;
               Filterobj.modelFlag = in.modelFlag;
               Filterobj.kInit     = in.kInit;
               Filterobj.xhatInit  = in.xhatInit;
               Filterobj.PInit     = in.PInit;
               % Assigns zero vector if uhist is empty
               if ~isempty(in.uhist)
                   Filterobj.uhist = in.uhist;
               else
                   Filterobj.uhist = zeros(size(in.zhist,1),1);
               end
               % Assigns zero vector if thist is empty
               if ~isempty(in.thist)
                   Filterobj.thist = in.thist;
               else
                   Filterobj.thist = zeros(size(in.zhist,1),1);
               end
               Filterobj.zhist     = in.zhist;
               Filterobj.Q         = in.Q;
               Filterobj.R         = in.R;
               Filterobj.optArgs   = in.optArgs;
           
           % If user inputs properties separately
           elseif nargin == 12
               % Check inputs
               inputsCheck(varargin{1:11});
               % Assign inputs to Filter Object properties
               Filterobj.fmodel    = varargin{1};
               Filterobj.hmodel    = varargin{2};
               Filterobj.modelFlag = varargin{3};
               Filterobj.kInit     = varargin{4};
               Filterobj.xhatInit  = varargin{5};
               Filterobj.PInit     = varargin{6};
               % Assigns zero vector if uhist is empty
               if ~isempty(varargin{7})
                   Filterobj.uhist = varargin{7};
               else
                   Filterobj.uhist = zeros(size(varargin{8},1),1);
               end
               % Assigns zero vector if thist is empty
               if ~isempty(varargin{9})
                   Filterobj.thist = varargin{9};
               else
                   Filterobj.thist = zeros(size(varargin{8},1),1);
               end
               Filterobj.zhist     = varargin{8};
               Filterobj.Q         = varargin{10};
               Filterobj.R         = varargin{11};
               Filterobj.optArgs   = {varargin{12}};
               
           % If user inputs properties incorrectly
           elseif nargin ~= 0
               error('Invalid input arguments. See filter class.')
           end
       end
   end
%% Dependent 'Get' Methods
% These methods are used to retrieve the dependent properties defined above
   methods
       % Size of state
       function val = get.nx(Filterobj)
           val = size(Filterobj.xhatInit,1);
       end
       % Size of control input
       function val = get.nu(Filterobj)
           val = size(Filterobj.uhist,2);
       end
       % Size of process noise
       function val = get.nv(Filterobj)
           val = size(Filterobj.Q,1);
       end
       % Size of measurements
       function val = get.nz(Filterobj)
           val = size(Filterobj.R,1);
       end
       % Number of samples to filter
       function val = get.kmax(Filterobj)
           val = size(Filterobj.zhist,1);
%%%% Save this for implementation in smoothers?
%            if ~isempty(Filterobj.zhist)
%                val = size(Filterobj.zhist,1);
%            elseif ~isempty(Filterobj.uhist)
%                val = size(Filterobj.uhist,1);
%            elseif ~isempty(Filterobj.thist)
%                val = size(Filterobj.thist,1);
%            else
%            end
       end
%%%% Save this for implementation in smoothers?
%        % Check if simulation is needed.
%        function val = get.simFlag(Filterobj)
%            val = isempty(Filterobj.zhist);
%        end
   end
%% Abstract Methods
% These methods are required to be defined by any batchFilter subclass

   methods (Abstract)
       argumentsCheck(Filterobj)
       initFilter(Filterobj)
       doFilter(Filterobj)
       dynamicProp(Filterobj)
       measUpdate(Filterobj)
   end
   
end

%% Helper Function
% This helper function checks that the inputs are the right type and size
function inputsCheck(fmodel,hmodel,modelFlag,kInit,xhatInit,PInit,uhist,zhist,thist,Q,R)

% Get problem dimensions for easier checking
Nx = size(xhatInit,1);
Nz = size(R,1);
Kmax = size(zhist,1);

% Assert that the necessary inputs have been given
assert((~isempty(fmodel) && isa(fmodel,'char')),...
    'batchfilter:instantiation',...
    'User must supply filter with a function name for the dynamics model')
assert((~isempty(hmodel) && isa(hmodel,'char')),...
    'batchfilter:instantiation',...
    'User must supply filter with a function name for the measurement model')
assert((~isempty(kInit) && kInit >=0),...
    'batchfilter:instantiation',...
    'User must supply filter with an initial sample integer larger than 1')
assert((~isempty(xhatInit) && ~isempty(PInit)),...
    'batchfilter:instantiation',...
    'User must supply filter with an initial a posteriori state estimate and error covariance')
assert(~isempty(zhist),...
    'batchfilter:instantiation',...
    'User must supply filter with a measurement history')
assert((~isempty(Q) && ~isempty(R)),...
    'batchfilter:instantiation',...
    'User must supply filter with process and measurement noise covariances')

% Assert that the model types are correct
assert((strcmp(modelFlag,'CD')||strcmp(modelFlag,'DD')),...
    'batchFilter:instantiation',...
    'The model-type flag is incorrect')
% Assert that the error covariance matrix is sized correctly
assert(((size(PInit,1)==Nx)&&(size(PInit,2)==Nx)),...
    'batchFilter:instantiation',...
    'Initial error covariance does not have the correct dimensions')

% Assert that the measurement vector and covariance are correctly sized.
assert((size(zhist,2)==Nz),...
    'batchFilter:instantiation',...
    'Measurement vector and measurement noise covariance do not match in size')
    
% Assert that the control history and time vectors are correctly sized.
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