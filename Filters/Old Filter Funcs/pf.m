function [xhatkhist,Pkhist] = pf(xhat0,P0,uhist,thist,zhist,Qk,Rkp1,fmodel,hmodel,modelFlag,nRK,Np)
% Particle Filter with Resampling

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
Xikp1 = xhat0*ones(1,Np) + chol(P0)'*randn(nx,Np);
Wikp1 = ones(1,Np)*(1/Np);

tk = 0;
for k = 0:(kmax-1)
    Xik = Xikp1;
    wik = Wikp1;
    logwk = log(wik);
    
    % Sample vi(k) from N[v(k);0,Q(k)]
    vik = chol(Qk)'*randn(nv,Np);
    
    % Preallocation
    Xikp1 = zeros(nx,Np);
    logWbarkp1 = zeros(1,nx);
    
    kp1 = k + 1;
    tkp1 = thist(kp1);
    for ii = 1:Np
        % Propagate particle to sample k+1
        if strcmp(modelFlag,'CD')
            [Xikp1(:,ii),~,~] = c2dnonlinear(Xik(:,ii),uhist(kp1,:)',vik(:,ii),tk,tkp1,nRK,fmodel,0);
        elseif strcmp(modelFlag,'DD')
            [Xikp1(:,ii),~,~] = feval(fmodel,Xik(:,ii),uhist(kp1,:)',vik(:,ii),k);
        else
            error('Incorrect flag for the dynamics-measurement models')
        end
%         Xikp1(:,ii) = fmodel(Xik(:,ii),uhist(kp1,:)',vik(:,ii),k);
        % Dummy calc for debug
        zdum = zhist(kp1,:)'-feval(hmodel,Xikp1(:,ii),0);
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
    Psums = zeros(nx,nx,Np);
    for ii = 1:Np
        Psums(:,:,ii) = Wikp1(ii)*((Xikp1(:,ii)-xhatkp1)*(Xikp1(:,ii)-xhatkp1)');
    end
    Pkp1 = sum(Psums,3);

    % Resample
    [Xikp1,Wikp1] = resample1(Xikp1,Wikp1,Np);
    
    % Prepare for next loop
    kp2 = kp1 + 1;
    tk = tkp1;
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

