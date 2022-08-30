function [Xs,gp] = gplite_sample(gp,Ns,x0,method,logprior,beta,VarThresh,proppdf,proprnd,bounds)
%GPLITE_SAMPLE Draw random samples from log pdf represented by GP.

if nargin < 3; x0 = []; end
if nargin < 4 || isempty(method); method = 'slicesample'; end
if nargin < 5 || isempty(logprior); logprior = []; end
if nargin < 6 || isempty(beta); beta = 0; end
if nargin < 7 || isempty(VarThresh); VarThresh = Inf; end
if nargin < 8 || isempty(proppdf); proppdf = []; end
if nargin < 9 || isempty(proprnd); proprnd = []; end
if nargin < 10; bounds = []; end

D = size(gp.X,2);

widths = std(gp.X,[],1);
if isempty(bounds)
    MaxBnd = 10;
    diam = max(gp.X) - min(gp.X);
    LB = min(gp.X) - MaxBnd*diam;
    UB = max(gp.X) + MaxBnd*diam;
else
    LB = bounds(1,:);
    UB = bounds(2,:);
end

% First, train GP
if ~isfield(gp,'post') || isempty(gp.post)
    % How many samples for the GP?
    if isfield(gp,'Ns') && ~isempty(gp.Ns)
        Ns_gp = gp.Ns;
    else
        Ns_gp = 0;
    end
    if isfield(gp,'Nopts') && ~isempty(gp.Nopts)
        options.Nopts = gp.Nopts;
    else
        options.Nopts = 1;  % Do only one optimization
    end
    if isfield(gp,'s2'); s2 = gp.s2; else; s2 = []; end
    gp = gplite_train(...
        [],Ns_gp,gp.X,gp.y,gp.covfun,gp.meanfun,gp.noisefun,s2,[],options);
end

% Recompute posterior auxiliary info if needed
if ~isfield(gp.post(1),'alpha') || isempty(gp.post(1).alpha)
    gp = gplite_post(gp);
end

logpfun = @(x) log_gpfun(gp,x,beta,VarThresh);

switch method
    case {'slicesample','slicesamplebnd'}
        sampleopts.Burnin = ceil(Ns/10);
        sampleopts.Thin = 1;
        sampleopts.Display = 'off';
        sampleopts.Diagnostics = false;
        sampleopts.LogPrior = logprior;
        sampleopts.MetropolisPdf = proppdf;
        sampleopts.MetropolisRnd = proprnd;

        if isempty(x0)
            [~,idx0] = max(gp.y);
            x0 = gp.X(idx0,:);
        else
            x0 = x0(1,:);
        end
        Xs = slicesamplebnd(logpfun, ...
            x0,Ns,widths,LB,UB,sampleopts);
        
    case 'parallel'
        sampleopts.Burnin = ceil(Ns/5);
        sampleopts.Thin = 1;
        sampleopts.Display = 'off';
        sampleopts.Diagnostics = false;
        sampleopts.VarTransform = false;
        sampleopts.InversionSample = false;
        sampleopts.FitGMM = false;
        
        if ~isempty(logprior)
            logPfuns = {logprior,logpfun};
        else
            logPfuns = logpfun;
        end
                
        % sampleopts.TransitionOperators = {'transSliceSampleRD'};
        
        W = 2*(D+1);
        if isempty(x0)
            % Take starting points from high posterior density region
            hpd_frac = 0.25;
            N = numel(gp.y);
            N_hpd = min(N,max(W,round(hpd_frac*N)));
            if isempty(logprior)
                [~,ord] = sort(gp.y,'descend');
            else
                dy = logprior(gp.X);
                [~,ord] = sort(gp.y + dy,'descend');                
            end
            X_hpd = gp.X(ord(1:N_hpd),:);
            x0 = X_hpd(randperm(N_hpd,min(W,N_hpd)),:);
        end
        x0 = bsxfun(@min,bsxfun(@max,x0,LB),UB);
        Xs = eissample_lite(logPfuns,x0,Ns,W,widths,LB,UB,sampleopts);
end

end

%--------------------------------------------------------------------------
function y = log_gpfun(gp,x,beta,VarThresh)

if (VarThresh == 0 || ~isfinite(VarThresh)) && beta == 0
    y = gplite_pred(gp,x);
else
    [y,s2] = gplite_pred(gp,x);
    y(s2 >= VarThresh) = y(s2 >= VarThresh) - (s2(s2 >= VarThresh) - VarThresh);
    y = y - beta*sqrt(s2);
end

end