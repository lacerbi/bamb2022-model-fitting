%% Tutorial for "Bayesian Inference for Statistical model fitting"
% BAMB! Summer School - Day 2 (September 2022)
% by Luigi Acerbi (2022)

% In this tutorial, we will estimate model parameters (aka fit our models) 
% in a fully Bayesian manner, which will yield a posterior distribution.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load mouse behavioral data

% Like the optimization tutorial, we will use data from the International 
% Brain Laboratory (IBL; https://www.internationalbrainlab.com/) publicly 
% released behavioral mouse dataset (training data of mouse `KS014`). See 
% The IBL et al. (2021) https://elifesciences.org/articles/63711 for info.

% Add utility folder to MATLAB path
baseFolder = fileparts(which('bamb2022_bayes_tutorial.m'));
addpath([baseFolder,filesep(),'bads']);
addpath([baseFolder,filesep(),'vbmc']);
addpath([baseFolder,filesep(),'utils']);

filename = './data/KS014_train.csv';
data = csvread(filename,1);
% We add a last column to represent "signed contrasts"
data = [data, data(:,4).*data(:,5)];

% The columns of 'data' are now (each row is a trial):
% 1. trial_num
% 2. session_num
% 3. stim_probability   (unused in training sessions)
% 4. contrast           (from 0 to 100)
% 5. position           (-1 left, 1 right)
% 6. response_choice    (-1 left, 1 right)
% 7. trial_correct      (1 yes, 0 no)
% 8. reaction_time      (seconds)
% 9. signed contrasts   (from -100 to 100)

fprintf('Loaded file %s.\n', filename);
fprintf('Total # of trials: %d\n', size(data,1));
fprintf('Sessions: %s\n', mat2str(unique(data(:,2))'));
data(1:5,:)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% I. Bayesian inference "by hand"

% As a first exercise, we fix the parameters of our psychometric function 
% model except for one, and derive a (one-dimensional) posterior 
% distribution for the free parameter. 
%
% Here we take sigma as the free parameter and fix the others.
% Our goal then is to infer: p(sigma|mu,lapse_rate,lapse_bias,data).

% Fixed parameter values (e.g., maximum-likelihood values)
mu_star = -10;          % bias
lambda_star = 0.26;     % lapse_rate
gamma_star = 0.54;      % lapse_bias

% We define hard lower/upper bounds for sigma
lb = 1;
ub = 100;

% Define a grid for sigma
sigma_range = linspace(0, ub + 1, 1001);
dsigma = sigma_range(2) - sigma_range(1);   % Grid spacing

% Define the prior over sigma as a uniform box between lb and ub (0 outside)
prior_pdf = 1/(ub - lb) * ((sigma_range >= lb) & (sigma_range <= ub));

% Just to see what happens: change the prior to a (truncated) Gaussian
%prior_pdf = normpdf(sigma_range,5,6) .* ((sigma_range >= lb) & (sigma_range <= ub));
%prior_pdf = prior_pdf ./ (sum(prior_pdf)*dsigma);   % Normalize

% Plot prior pdf
figure;
subplot(1,2,1);
plot(sigma_range,prior_pdf,'-k','LineWidth',1);
xlabel('\sigma')
ylabel('p(\sigma)');
title('Prior pdf');
set(gca,'TickDir','out');
box off;
set(gcf,'Color','w');
xlim([0,110]);
ylim([0,0.15]);

% Define the log-likelihood function
session_num = 12;   % Pick session data
session_data = data(data(:,2) == session_num,:);
loglike = @(sigma_) psychofun_loglike([mu_star,sigma_,lambda_star,gamma_star],session_data);

% Now we directly compute the log posterior using Bayes rule (in log space)

% Define a grid to store the log posterior
logpost = zeros(1, numel(sigma_range));

% Compute log of unnormalized posterior, log (likelihood * prior), for each
% value in the grid
for i = 1:numel(sigma_range)
    sigma = sigma_range(i);
    logpost(i) = loglike(sigma) + log(prior_pdf(i));    % Unnormalized
end

% Normalize log posterior
post = exp(logpost - max(logpost)); % Subtract maximum for numerical stability
post = post ./ sum(post * dsigma);  % This is a density

fprintf('Posterior integral: %.3f.\n', sum(post * dsigma)); % Integrates to 1

% Plot posterior pdf
subplot(1,2,2);
plot(sigma_range,post,'-k','LineWidth',1);
xlabel('\sigma')
ylabel('p(\sigma | \mu_*, \lambda_*, \gamma_*, data)');
title('Posterior pdf');
set(gca,'TickDir','out');
box off;
set(gcf,'Color','w');
xlim([0,110]);
ylim([0,0.15]);

% Try this again with a different prior!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% II. Setting up for Bayesian inference

% Similar to the optimizaation tutorial, we define the "hard" lower and 
% upper bounds for the parameters. We also define "plausible" bounds which
% we associate with a higher prior probability of the parameters.

% Define hard parameter bounds
lb = [-100,1,0,0];
ub = [100,100,1,1];

% Define plausible range
plb = [-25,5,0.05,0.2];
pub = [25,25,0.40,0.8];

% Number of variables
D = numel(lb);

% We need to define a prior for the parameters. This choice is up to you!
% A common choice is to define the prior separately for each parameter.
% Here, for each parameter we define the prior to be flat within the 
% plausible range, and then it smoothly falls to zero towards the hard bounds. 
% The provided `msplinetrapezpdf` function implements this kind of prior.
% Alternatively, you could have a non-informative uniformly flat prior 
% between the hard bounds (implemented by `unifboxpdf`).

% Let's plot the "smoothed trapezoidal" prior for each dimension
parameter_names{1} = 'bias (\mu)';
parameter_names{2} = 'slope/noise (\sigma)';
parameter_names{3} = 'lapse rate (\lambda)';
parameter_names{4} = 'lapse bias (\gamma)';

for d = 1:D
    x = linspace(lb(d),ub(d),1e3)';
    subplot(2,ceil(D/2),d);
    plot(x, msplinetrapezpdf(x,lb(d),plb(d),pub(d),ub(d)), 'k-', 'LineWidth', 1);
    set(gca,'TickDir','out');
    xlabel(parameter_names{d})
    ylabel('prior pdf')
    box off;
end
set(gcf,'Color','w');

% Define the smoothed trapezoidal prior
prior_fun = @(theta_) msplinetrapezpdf(theta_,lb,plb,pub,ub);

% Alternatively, this implements a uniform prior
% prior_fun = @(theta_) unifboxpdf(theta_,lb,ub);

% Pick session data
session_num = 12;
session_data = data(data(:,2) == session_num,:);

% Define objective function: log-joint
fun = @(theta_) psychofun_loglike(theta_,session_data) + log(prior_fun(theta_));

% Generate random starting point inside the plausible box
theta0 = rand(1,D).*(pub-plb) + plb;

% Get the maximum-a-posteriori (MAP) estimate using BADS
options = bads('defaults');
options.Display = 'iter';
theta_MAP = bads(@(x) -fun(x),theta0,lb,ub,plb,pub,options);

% Run inference using Variational Bayesian Monte Carlo (VBMC):
% https://github.com/lacerbi/vbmc

% We start the algorithm from the MAP (not necessary but it helps)
options = vbmc('defaults');
options.Display = 'iter';
options.Plot = 'on';
[vp,elbo,elbo_sd,exitflag,output] = vbmc(fun,theta_MAP,lb,ub,plb,pub,options);

% VBMC returns:
% - vp: the (approximate) posterior distribution
% - elbo: the lower bound to the log marginal likelihood (log evidence)
% - elbo_sd: standard deviation of the estimate of the elbo
% - exitflag: a success flag
% - output: a struct with information about the run

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% III. Plot prediction

close all;

% First, let's visualize the posterior
Nsamples = 1e5;
thetas = vbmc_rnd(vp,Nsamples);
cornerplot(thetas,parameter_names);

% Plot data first
figure(2);
plot_psychometric_data(data,session_num);

% Get random samples from the posterior
Nsamples = 1e4;
thetas = vbmc_rnd(vp,Nsamples);
% Compute psychometric function values for each sample
stim = linspace(-100,100,201);
p_right = zeros(Nsamples,201); 
for i = 1:Nsamples; p_right(i,:) = psychofun(thetas(i,:),stim); end
median_pred = quantile(p_right,0.5,1);  % Median prediction
bottom_pred = quantile(p_right,0.025,1);  % Bottom 2.5% prediction
top_pred = quantile(p_right,0.975,1);     % Top 97.5% prediction

hold on;
patch([stim,fliplr(stim)],[bottom_pred,fliplr(top_pred)],'k','EdgeColor','none','FaceAlpha',0.2);
plot(stim,median_pred,'LineWidth',1,'Color','k','DisplayName','model');