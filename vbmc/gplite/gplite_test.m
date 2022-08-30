function gp = gplite_test(test,hyp,X,y,covfun,meanfun,noisefun,s2,outwarpfun)
%GPLITE_TEST Test computations for lite GP functions.

if nargin < 1; test = []; end
if nargin < 2; hyp = []; end
if nargin < 3; X = []; end
if nargin < 4; y = []; end
if nargin < 5 || isempty(covfun); covfun = 'seard'; end
if nargin < 6 || isempty(meanfun); meanfun = 'zero'; end
if nargin < 7; noisefun = []; end
if nargin < 8; s2 = []; end
if nargin < 9; outwarpfun = []; end

if isempty(test); test = 'all'; end
all_flag = any(strcmpi(test,'all'));

%% Set up basic GP
if isempty(noisefun)
    if isempty(s2); noisefun = [1 0 0]; else; noisefun = [1 1 0]; end
end

if isempty(X)
    D = 2; 
    N = 20;
    X = randn(N,D);
else
    D = size(X,2);
end

Ncov = gplite_covfun('info',X,covfun);
Nnoise = gplite_noisefun('info',X,noisefun);
Nmean = gplite_meanfun('info',X,meanfun);
if ~isempty(outwarpfun)
    Noutwarp = outwarpfun('info',y);
else
    Noutwarp = 0;
end

if isempty(hyp)
    Ns = randi(3);    % Test multiple hyperparameters
    hyp = [randn(D,Ns); 0.2*randn(1,Ns); 0.3*randn(Nnoise,Ns); randn(Nmean,Ns); randn(Noutwarp,Ns)];
end

if isempty(s2)
    if noisefun(2) > 0; s2 = 0.1*exp(randn(N,1)); end
end

if isempty(y)   % Generate from the GP prior
    gp = gplite_post(hyp(:,1),X,[],covfun,meanfun,noisefun,s2,0,outwarpfun);    
    [~,y] = gplite_rnd(gp,X);
end

[N,D] = size(X);            % Number of training points and dimension
[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples

hyp0 = hyp(:,1);

gp = gplite_post(hyp0,X,y,covfun,meanfun,noisefun,s2,0,outwarpfun);

fprintf('Ncov = %d; Nnoise = %d; Nmean = %d; Noutwarp = %d. # samples: %d.\n\n',Ncov,Nnoise,Nmean,Noutwarp,Ns);

hyp

fprintf('---------------------------------------------------------------------------------\n');
fprintf('Print noise variance for every GP point...\n\n');

sn2 = gplite_noisefun(hyp0(Ncov+1:Ncov+Nnoise),gp.X,gp.noisefun,gp.y,gp.s2)

% Test GP marginal likelihood gradient computation
if any(strcmpi(test,'nlzgrad')) || all_flag
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check GP marginal likelihood gradient computation...\n\n');
    f = @(x) gplite_nlZ(x,gp);
    derivcheck(f,hyp0.*exp(0.1*rand(size(hyp0))));
end

if any(strcmpi(test,'priorgrad')) || all_flag
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check GP hyperparameters log prior gradient computation...\n\n');
    hprior.mu = randn(size(hyp0));
    hprior.sigma = exp(randn(size(hyp0)));
    hprior.df = exp(randn(size(hyp0))).*(randi(2,size(hyp0))-1);
    f = @(x) gplite_hypprior(x,hprior);
    derivcheck(f,hyp0.*exp(0.1*rand(size(hyp0))));
end

if any(strcmpi(test,'rank1')) || all_flag
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check rank-1 GP updates...\n\n');

    idx = ceil(size(X,1)/2);
    gp1 = gplite_post(hyp0,X(1:idx,:),y(1:idx),covfun,meanfun,noisefun,s2(1:min(idx,size(s2,1))),0,outwarpfun);
    for ii = idx+1:size(gp.X,1)
        xstar = X(ii,:);
        ystar = y(ii);
        if ~isempty(s2); s2star = s2(ii); else; s2star = []; end    
        gp1 = gplite_post(gp1,xstar,ystar,[],[],[],s2star,1);
    end

    ff = fields(gp.post)';
    for i = 1:numel(ff)
        update1_errs(i) = sum(abs(gp.post.(ff{i})(:) - gp1.post.(ff{i})(:)));
    end
    update1_errs
end

if any(strcmpi(test,'meangrad')) || all_flag
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check mean function gradient computation...\n\n');
    f = @(x) gplite_meanfun(x,X(1,:),meanfun,y(1));
    hyp0_mean = hyp0(Ncov+Nnoise+1:Ncov+Nnoise+Nmean);    
    derivcheck(f,hyp0_mean.*exp(0.1*rand(size(hyp0_mean))));
end

if any(strcmpi(test,'intmean')) || all_flag
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check integrated mean function...\n\n');
    gp = gplite_intmeantest(X,y,covfun,noisefun,s2);
end

% Test plotting
gplite_plot(gp);

% fprintf('---------------------------------------------------------------------------------\n');
% fprintf('Check GP training with warped data...\n\n');
% 
% D = 1;
% s2 = 2;
% Xt = linspace(-3,3,51)';
% yt = -0.5*Xt.^2/s2;
% LB = 0; UB = 5;
% warp.LB = LB; warp.UB = UB;
% trinfo = warpvars(D,LB,UB);
% trinfo.type = 9;
% trinfo.alpha = 3;
% trinfo.beta = 0.1;
% X = warpvars(Xt,'inv',trinfo);
% y = yt - warpvars(Xt,'logp',trinfo);
% Ncov = D+1;
% Nmean = gplite_meanfun([],X,meanfun);
% Nhyp0 = Ncov+1+Nmean+2*D;
% 
% warp.LB = LB; warp.UB = UB; warp.logpdf_flag = 1;
% hyp0 = zeros(Nhyp0,1);
% [gp,hyp] = gplite_train(hyp0,0,X,y,meanfun,[],warp);
% plot(Xt,yt); hold on;
% plot(X,y); plot(gp.X,gp.y);


end

%--------------------------------------------------------------------------
function gp = gplite_intmeantest(X,y,covfun,noisefun,s2)

[N,D] = size(X);
Nb = 2*D+1;

% Add true basis functions to observations
b_true = randn(1,Nb)
y = y + b_true(1) + X*b_true(2:D+1)' + (X.^2)*b_true(D+2:2*D+1)';

Ns = 0;
bb = zeros(1,Nb);
BB = Inf*ones(1,Nb);

meanfun = {0,3,bb,BB};
options = [];

gp = gplite_train([],Ns,X,y,covfun,meanfun,noisefun,s2,[],options);

fprintf('Inferred basis function coefficients:\n')
gp.post.intmean.betabar

end