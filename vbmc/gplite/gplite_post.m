function gp = gplite_post(hyp,X,y,covfun,meanfun,noisefun,s2,update1,outwarpfun)
%GPLITE_POST Compute posterior GP for a given training set.
%   GP = GPLITE_POST(HYP,X,Y,S2,MEANFUN) computes the posterior GP for a vector
%   of hyperparameters HYP and a given training set. HYP is a column vector 
%   of hyperparameters (see below). X is a N-by-D matrix of training inputs 
%   and Y a N-by-1 vector of training targets (function values at X).
%   MEANFUN is the GP mean function (see GPLITE_MEANFUN.M for a list).
%
%   GP = GPLITE_POST(HYP,X,Y,S2,MEANFUN) computes the posterior GP for a vector
%   of hyperparameters HYP and a given training set. HYP is a column vector 
%   of hyperparameters (see below). X is a N-by-D matrix of training inputs 
%   and Y a N-by-1 vector of training targets (function values at X).
%   MEANFUN is the GP mean function (see GPLITE_MEANFUN.M for a list).
%
%   GP = GPLITE_POST(GP,XSTAR,YSTAR,[],1) performs a fast rank-1 update for 
%   a GPLITE structure, given a single new observation at XSTAR with observed
%   value YSTAR.
%
%   Note that the returned GP contains auxiliary structs for faster
%   computations. To save memory, call GPLITE_CLEAN.
%
%   See also GPLITE_CLEAN, GPLITE_MEANFUN.

if nargin < 4; covfun = []; end
if nargin < 5; meanfun = []; end
if nargin < 6; noisefun = []; end
if nargin < 7; s2 = []; end
if nargin < 8 || isempty(update1); update1 = false; end
if nargin < 9; outwarpfun = []; end

gp = [];
if isstruct(hyp)
    gp = hyp;
    if nargin < 2; X = gp.X; end
    if nargin < 3; y = gp.y; end
    if nargin < 4; covfun = gp.covfun; end
    if nargin < 5
        if isfield(gp,'intmeanfun') && gp.intmeanfun > 0
            meanfun = {gp.meanfun,gp.intmeanfun,gp.intmeanfun_mean,gp.intmeanfun_var};
        else
            meanfun = gp.meanfun;
        end
    end
    if nargin < 6; noisefun = gp.noisefun; end
    if nargin < 7; s2 = gp.s2; end
    if nargin < 9
        if isfield(gp,'outwarpfun'); outwarpfun = gp.outwarpfun; else; outwarpfun = []; end
    end
end

if update1
    if size(X,1) > 1
        error('gplite_post:NotRankOne', ...
          'GPLITE_POST with this input format only supports rank-one updates.');
    end
    if isempty(gp)
        error('gplite_post:NoGP', ...
          'GPLITE_POST can perform rank-one update only with an existing GP struct.');        
    end
    if ~isempty(covfun)
        warning('gplite_post:RankOneCovFunction', ...
            'No need to specify a GP covariance function when performing a rank-one update.');
    end
    if ~isempty(meanfun)
        warning('gplite_post:RankOneMeanFunction', ...
            'No need to specify a GP mean function when performing a rank-one update.');
    end
    if ~isempty(noisefun)
        warning('gplite_post:RankOneNoiseFunction', ...
            'No need to specify a GP noise function when performing a rank-one update.');
    end
    if nargin > 8 && ~isempty(outwarpfun)
        warning('gplite_post:RankOneNoiseFunction', ...
            'No need to specify a GP output warping function when performing a rank-one update.');
    end
    if ~isempty(s2)
        % Rank-1 update is not supported with heteroskedastic noise
        update1 = false;
    end
    if gp.intmeanfun > 0
        % Rank-1 update is not supported with integrated mean functions
        update1 = false;
    end
    
    if ~update1
        % Perform standard update
        gp.X = [gp.X; X];
        gp.y = [gp.y; y];
        if ~isempty(s2); gp.s2 = [gp.s2; s2]; end
    end
end

% Create GP struct
if isempty(gp)
    gp.X = X;
    gp.y = y;
    gp.s2 = s2;
    [N,D] = size(X);            % Number of training points and dimension
    [Nhyp,Ns] = size(hyp);      % Number of hyperparameters and samples
    if isempty(covfun); covfun = 1; end
    if isempty(meanfun); meanfun = 1; end
    if isempty(noisefun)
        if isempty(s2); noisefun = [1 0 0]; else; noisefun = [1 1 0]; end
    end
    if iscell(meanfun)
        % Integrated mean function
        intmeanfun = meanfun{2};
        intmeanfun_priormean = meanfun{3};
        intmeanfun_priorvar = meanfun{4};
        meanfun = meanfun{1};
    else
        intmeanfun = 0;
    end
    
    % Get number and field of covariance / noise / mean function
    [gp.Ncov,info] = gplite_covfun('info',X,covfun);
    gp.covfun = info.covfun;
    [gp.Nnoise,info] = gplite_noisefun('info',X,noisefun);
    gp.noisefun = info.noisefun;
    [gp.Nmean,info] = gplite_meanfun('info',X,meanfun,y);
    gp.meanfun = info.meanfun;
    gp.meanfun_extras = info.extras;
    gp.intmeanfun = intmeanfun;
    if gp.intmeanfun > 0
        gp.intmeanfun_mean(1,:) = intmeanfun_priormean;
        gp.intmeanfun_var(1,:) = intmeanfun_priorvar;
        inf_idx = isinf(gp.intmeanfun_var);
        gp.intmeanfun_mean(inf_idx) = 0;
    else
        gp.intmeanfun_mean = [];
        gp.intmeanfun_var = [];
    end
    
    % Output warping function (optional)
    if ~isempty(outwarpfun)
        [Noutwarp,info] = outwarpfun('info',y);
        gp.Noutwarp = Noutwarp;
        gp.outwarpfun = info.outwarpfun;
    else
        Noutwarp = 0;
        gp.outwarpfun = [];
    end
        
    % Create posterior structure
    postfields = {'hyp','alpha','sW','L','sn2_mult','Lchol'};
    for i = 1:numel(postfields); post.(postfields{i}) = []; end
    if gp.intmeanfun > 0; post.intmean = []; end
    for s = 1:size(hyp,2)
        gp.post(s) = post;
        gp.post(s).hyp = hyp(:,s);
    end
        
    if isempty(hyp) || isempty(y); return; end
    if Nhyp ~= gp.Ncov+gp.Nnoise+gp.Nmean+Noutwarp
        error('gplite_post:dimmismatch', ...
            'Number of hyperparameters mismatched with GP model specification.');
    end
else
    [N,D] = size(gp.X);         % Number of training points and dimension
    Ns = numel(gp.post);        % Hyperparameter samples    
    if ~isfield(gp,'meanfun_extras')
        [~,info] = gplite_meanfun('info',gp.X,gp.meanfun,gp.y);
        gp.meanfun_extras = info.extras;        
    end
end

if ~update1    
    % Loop over hyperparameter samples
    for s = 1:Ns
        hyp = gp.post(s).hyp;
        [~,~,gp.post(s)] = gplite_core(hyp,gp,0,0);        
    end    
else
    % Perform rank-1 update of the GP posterior        
    Ncov = gp.Ncov;
    Nnoise = gp.Nnoise;
    Nmean = gp.Nmean;
    
    outwarp_flag = isfield(gp,'outwarpfun') && ~isempty(gp.outwarpfun);
    
    if outwarp_flag
        Noutwarp = gp.Noutwarp;    
        if ~isempty(s2); error('Heteroskedastic noise not supported with output warping yet.'); end
    end
    
    % Added training input
    xstar = X;
    
    % Compute prediction for all samples
    [mstar,vstar] = gplite_pred(gp,xstar,y,s2,1,1);
        
    % Loop over hyperparameter samples
    for s = 1:Ns
        
        hyp = gp.post(s).hyp;
        
        if outwarp_flag
            hyp_outwarp = hyp(Ncov+Nnoise+Nmean+1:Ncov+Nnoise+Nmean+Noutwarp);
            ystar = gp.outwarpfun(hyp_outwarp,y);
            s2star = s2;
        else
            ystar = y;
            s2star = s2;
        end
                
        hyp_noise = hyp(Ncov+1:Ncov+Nnoise); % Get noise hyperparameters
        sn2 = gplite_noisefun(hyp_noise,xstar,gp.noisefun,ystar,s2star);
        sn2_eff = sn2*gp.post(s).sn2_mult;            
        
        % Compute covariance and cross-covariance
        if gp.covfun(1) == 1    % Hard-coded SE-ard for speed
            ell = exp(hyp(1:D));
            sf2 = exp(2*hyp(D+1));        
            K = sf2;
            Ks_mat = sq_dist(diag(1./ell)*gp.X',diag(1./ell)*xstar');
            Ks_mat = sf2 * exp(-Ks_mat/2);
        else
            hyp_cov = hyp(1:Ncov);
            K = gplite_covfun(hyp_cov,xstar,gp.covfun,'diag');
            Ks_mat = gplite_covfun(hyp_cov,gp.X,gp.covfun,xstar);            
        end            
        
        L = gp.post(s).L;
        Lchol = gp.post(s).Lchol;
        
        if Lchol        % high-noise parameterization
            alpha_update = (L\(L'\Ks_mat)) / sn2_eff;
            new_L_column = linsolve(L, Ks_mat, ...
                                struct('UT', true, 'TRANSA', true)) / sn2_eff;
            gp.post(s).L = [L, new_L_column; ...
                               zeros(1, size(L, 1)), ...
                               sqrt(1 + K / sn2_eff - new_L_column' * new_L_column)];
        else            % low-noise parameterization
            alpha_update = -L * Ks_mat;
            v = -alpha_update / vstar(:,s);
            gp.post(s).L = [L + v * alpha_update', -v; -v', -1 / vstar(:,s)];
        end
        
        gp.post(s).sW = [gp.post(s).sW; 1/sqrt(sn2_eff)];
        
        % alpha_update now contains (K + \sigma^2 I) \ k*
        gp.post(s).alpha = ...
            [gp.post(s).alpha; 0] + ...
            (mstar(:,s) - ystar) / vstar(:,s) * [alpha_update; -1];
    end
    
    % Add single input to training set
    gp.X = [gp.X; xstar];
    gp.y = [gp.y; y];
    if ~isempty(s2); gp.s2 = [gp.s2; s2]; end
end