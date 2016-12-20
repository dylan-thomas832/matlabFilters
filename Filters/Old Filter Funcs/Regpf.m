function [xhatkhist,Pkhist] = Regpf(xhat0,P0,uhist,zhist,Qk,Rkp1,fmodel,hmodel,Ns,NT)


% Get problem dimensions and setup the output arrays
nx = size(xhat0,1);
nv = size(Qk,1);
kmax = size(zhist,1);

% Preallocation
xhatkhist = zeros(nx,kmax+1);
Pkhist = zeros(nx,nx,kmax+1);

% Store initial estimate and its error covariance
xhatkhist(:,1) = xhat0;
Pkhist(:,:,1) = P0;

% Generate Ns samples of Xi0 from N[x(k);xhat0,P0] and weights
Xikp1 = xhat0 + chol(P0)'*randn(nx,Ns);
Wikp1 = ones(1,Ns)*(1/Ns);


for k = 0:(kmax-1)
    Xik = Xikp1;
    wik = Wikp1;
    % precalculations to save time
    % invRkp1 = inv(Rkp1);
    logwk = log(wik);
    
    % Sample vi(k) from N[v(k);0,Q(k)]
    vik = chol(Qk)'*randn(nv,Ns);
    
    % Preallocation
    Xikp1 = zeros(nx,Ns);
    logWbarkp1 = zeros(1,nx);
    
    kp1 = k + 1;
    for ii = 1:Ns
        % Propagate particle to sample k+1
        Xikp1(:,ii) = fmodel(Xik(:,ii),uhist(kp1,:)',vik(:,ii),k);
        % Dummy calc for debug
        zdum = zhist(kp1,:)'-hmodel(Xikp1(:,ii),k+1);
        % Calculate the current particle's log-weight
        logWbarkp1(ii) = -0.5*(zdum)'*inv(Rkp1)*(zdum) + logwk(ii);
    end
    
    % Find imax which has a greater log-weight than every other log-weight
    Wmax = max(logWbarkp1);
    
    % Normalize weights
    Wbarbarkp1 = exp(logWbarkp1-Wmax);
    Wikp1 = Wbarbarkp1./sum(Wbarbarkp1);
    
    % Compute xhat(k+1) and P(k+1) from weights and particles
    xhatkp1 = Xikp1*Wikp1';
    Psums = zeros(nx,nx,Ns);
    for ii = 1:Ns
        Psums(:,:,ii) = Wikp1(ii)*((Xikp1(:,ii)-xhatkp1)*(Xikp1(:,ii)-xhatkp1)');
    end
    Pkp1 = sum(Psums,3);
    
    % Compute Neff and determine whether to re-sample or not
    Neff = 1/sum(Wikp1.^2);
    if (Neff>NT)
        return; % Neff is suddificiently large
    else
        % Not enough effective particles
        
        % Resample
        [Xkp1new,Wikp1] = resample1(Xikp1,Wikp1,Ns);
        
        % Cholesky fctorize covariance
        [Skp1,err] = chol(Pkp1);
        %%%%% TODO: Cheap get around
        if err > 0
            Skp1 = zeros(size(Pkp1));
        else
            Skp1 = Skp1';
        end
        % Sample Beta from Kernel function. Need other outputs?
        [Beta,~,~,~] = epanechnikovsample01(nx,Ns);
        % Hypersphere in nx space
        Cnx = unithypervolume01(nx);
        % Coefficient calculations
        A = ((8/Cnx)*(nx+4)*(2*sqrt(pi))^nx)^(1/(nx+4));
        hopt = A/(Ns^(1/(nx+4)));
        
        % Recalculate the particles at sample k.
        Xikp1 = Xkp1new + hopt*Skp1*Beta;
    end
    
    
%     zkp1 = zhist(kp1,:)';
    kp2 = kp1 + 1;
    xhatkhist(:,kp2) = xhatkp1;
    Pkhist(:,:,kp2) = Pkp1;
end


end

function [Xi,Wi] = resample1(Xi,Wi,Ns)
    
% Initialize re-sample algorithm
c = zeros(1,Ns+1);
c(end) = 1.0000000001;
% Calculate c coefficients
for ii = 2:Ns
    c(ii) = sum(Wi(1:ii-1));
end
    
% Preallocation
Xinew = zeros(size(Xi));

% Initialize
ll = 1;
while ll <= Ns
    % Sample random, uniformly distributed variable
    eta = rand(1,1);
    
    % Find indices
    for ii = 1:Ns
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
Wi = ones(1,Ns)*(1/Ns);
% Assign new, re-sampled particles to original particles variable
Xi = Xinew;
end

