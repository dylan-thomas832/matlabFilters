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
classdef (Abstract) stepFilter

%% Default Inputs
% These are the properties that every stepFilter subclass takes in from
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
                    
       k            % Scalar Integer:
                    %
                    % Current sample at which the filter step is to be 
                    % implemented.
                    
       xhatk        % (nx)x1 Vector:
                    %
                    % a posteriori state estimate at sample k.
       
       Pk           % (nx)x(nx) Matrix:
                    %
                    % a posteriori error covariance matrix at sample k.
       
       uk           % (nu)x1 Vector:
                    % 
                    % Control state at sample k. This is assumed to be 
                    % known and deterministic. This input can be an empty 
                    % vector if there are no control inputs.
       
       zkp1         % (nz)x1 Vector:
                    %
                    % Measurement state at sample k. These are the actual 
                    % measurements taken by the system, supplied by 
                    % the user.
                    
       tk           % Scalar:
                    %
                    % Defines the discrete time at sample k. This is
                    % mainly for problems where the dynamics models 
                    % are defined in continuous-time. This can be an 
                    % empty vector if there is no need for defining the 
                    % times of each sample.
       
       dt           % Scalar:
                    % 
                    % Change in time between sample k and sample k+1. Leave
                    % as empty scalar if not needed (discrete models).
                    
       Qk           % (nv)x(nv) Matrix:
                    % 
                    % Symmetric, positive definite process noise 
                    % covariance matrix at sample k.
       
       Rkp1         % (nz)x(nz) Matrix:
                    % 
                    % Symmetric, positive definite measurement noise 
                    % covariance matrix at sample k+1.
                    
       optArgs      % Cell Array:
                    %
                    % Extra/optional arguments for each filter class.
   end
   
%% Default Outputs
% These are the default properties which all stepFilter subclasses
% calculate and store as outputs of their algorithm. More may be added.
   properties

       xhatkp1      % (nx)x1 Vector:
                    % 
                    % a posteriori state estimate at sample k+1.
       
       Pkp1         % (nx)x(nx) Matrix:
                    %
                    % a posteriori error covariance matrix at sample k+1.
       
       eta_nukp1    % Scalar:
                    % 
                    % Innovation statistic at sample k+1.
   end

%% Dependent Properties
% These are general properties which every stepfilter subclass may use,
% but are dependent on the Input Properties.
   properties (Dependent, Access = protected)
       nx           % Number of states
       nu           % Number of control inputs
       nv           % Number of process noise states
       nz           % Number of measurement states
       tkp1         % Time at sample k+1
   end

%% Default Constructor
% This method is the default constructor for all stepFilter subclasses
   methods (Access = protected)
       % step Filter constructor
       function Filterobj = stepFilter(varargin)
           % If user inputs properties inside a structure
           if nargin == 1
               % Get input structure containing inputs
               in = varargin{1};
               % Check inputs
               inputsCheck(in.fmodel,in.hmodel,in.modelFlag,in.k,in.xhatk,in.Pk,in.zkp1,in.Qk,in.Rkp1);
               % Assign inputs to Filter Object properties
               Filterobj.fmodel    = in.fmodel;
               Filterobj.hmodel    = in.hmodel;
               Filterobj.modelFlag = in.modelFlag;
               Filterobj.k         = in.k;
               Filterobj.xhatk     = in.xhatk;
               Filterobj.Pk        = in.Pk;
               % Assigns zero if uk is empty
               if ~isempty(in.uk)
                   Filterobj.uk = in.uk;
               else
                   Filterobj.uk = 0;
               end
               Filterobj.zkp1      = in.zkp1;
               % Assigns zero if tk is empty
               if ~isempty(in.tk)
                   Filterobj.tk = in.tk;
               else
                   Filterobj.tk = 0;
               end
               % Assigns zero if dt is empty
               if ~isempty(in.dt)
                   Filterobj.dt = in.dt;
               else
                   Filterobj.dt = 0;
               end
               Filterobj.Qk        = in.Qk;
               Filterobj.Rkp1      = in.Rkp1;
               Filterobj.optArgs   = in.optArgs;
           
           % If user inputs properties separately
           elseif nargin == 13
               % Check inputs
               inputsCheck(varargin{1:11});
               % Assign inputs to Filter Object properties
               Filterobj.fmodel    = varargin{1};
               Filterobj.hmodel    = varargin{2};
               Filterobj.modelFlag = varargin{3};
               Filterobj.k         = varargin{4};
               Filterobj.xhatk     = varargin{5};
               Filterobj.Pk        = varargin{6};
               % Assigns zero if uk is empty
               if ~isempty(varargin{7})
                   Filterobj.uk    = varargin{7};
               else
                   Filterobj.uk    = 0;
               end
               Filterobj.zkp1      = varargin{8};
               % Assigns zero if tk is empty
               if ~isempty(varargin{9})
                   Filterobj.tk    = varargin{9};
               else
                   Filterobj.tk    = 0;
               end
               % Assigns zero if dt is empty
               if ~isempty(varargin{10})
                   Filterobj.dt    = varargin{10};
               else
                   Filterobj.dt    = 0;
               end
               Filterobj.Qk        = varargin{11};
               Filterobj.Rkp1      = varargin{12};
               Filterobj.optArgs   = {varargin{13}};
               
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
           val = size(Filterobj.xhatk,1);
       end
       % Size of control input
       function val = get.nu(Filterobj)
           val = size(Filterobj.uk,2);
       end
       % Size of process noise
       function val = get.nv(Filterobj)
           val = size(Filterobj.Qk,1);
       end
       % Size of measurements
       function val = get.nz(Filterobj)
           val = size(Filterobj.Rkp1,1);
       end
       % Number of samples to filter
       function val = get.tkp1(Filterobj)
           val = Filterobj.tk + Filterobj.dt;
       end
   end
%% Abstract Methods
% These methods are required to be defined by any stepFilter subclass

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
function inputsCheck(fmodel,hmodel,modelFlag,k,xhatk,Pk,zkp1,Qk,Rkp1)

% Get problem dimensions for easier checking
Nx = size(xhatk,1);
Nz = size(Rkp1,1);

% Assert that the necessary inputs have been given
assert((~isempty(fmodel) && isa(fmodel,'char')),...
    'stepfilter:instantiation',...
    'User must supply filter with a function name for the dynamics model')
assert((~isempty(hmodel) && isa(hmodel,'char')),...
    'stepfilter:instantiation',...
    'User must supply filter with a function name for the measurement model')
assert((~isempty(k) && k >=0),...
    'stepfilter:instantiation',...
    'User must supply filter with a sample number')
assert((~isempty(xhatk) && ~isempty(Pk)),...
    'stepfilter:instantiation',...
    'User must supply filter with an a posteriori state estimate and error covariance at sample k')
assert(~isempty(zkp1),...
    'stepfilter:instantiation',...
    'User must supply filter with a measurement at sample k+1')
assert((~isempty(Qk) && ~isempty(Rkp1)),...
    'stepfilter:instantiation',...
    'User must supply filter with process and measurement noise covariances at sample k and k+1 respectively')

% Assert that the model types are correct
assert((strcmp(modelFlag,'CD')||strcmp(modelFlag,'DD')),...
    'stepFilter:instantiation',...
    'The model-type flag is incorrect')
% Assert that the error covariance matrix is sized correctly
assert(((size(Pk,1)==Nx)&&(size(Pk,2)==Nx)),...
    'stepFilter:instantiation',...
    'Initial error covariance does not have the correct dimensions')

% Assert that the measurement vector and covariance are correctly sized.
assert((size(zkp1,1)==Nz),...
    'stepFilter:instantiation',...
    'Measurement vector and measurement noise covariance do not match in size')

end