%% Tutorial for "Introduction to Optimization for Statistical model fitting"
% BAMB! Summer School - Day 2 (September 2022)
% by Luigi Acerbi (2022)

% Add utility folder to MATLAB path
baseFolder = fileparts(which('bamb2022_optimization_tutorial.m'));
addpath([baseFolder,filesep(),'bads']);
addpath([baseFolder,filesep(),'utils']);

% During this tutorial, we are going to use data from the International 
% Brain Laboratory (IBL; https://www.internationalbrainlab.com/) publicly 
% released behavioral mouse dataset, from exemplar mouse `KS014`. 
% See The IBL et al. (2021) https://elifesciences.org/articles/63711 for 
% more information about the task and datasets. 
%
% These data can also be inspected via the IBL DataJoint public interface: 
% https://data.internationalbrainlab.org/mouse/18a54f60-534b-4ed5-8bda-b434079b8ab8
%
% For convenience, the data of all behavioral sessions from examplar mouse 
% `KS014` have been already downloaded in the `data` folder and slightly 
% preprocessed into two `.csv` files, one for the training sessions 
% (`KS014_train.csv`) and one with the *biased* sessions (`KS014_biased.csv`). 
% 
% We begin our tutorial by examining the training sessions. 

filename = './data/KS014_train.csv';
data = csvread(filename,1);
% We add a last column to represent "signed contrasts"
data = [data, data(:,4).*data(:,5)];

% The columns of 'data' are now:
% 1. trial_num
% 2. session_num
% 3. stim_probability   (unused in training sessions)
% 4. contrast           (from 0 to 100)
% 5. position           (-1 left, 1 right)
% 6. response_choice    (-1 left, 1 right)
% 7. trial_correct      (1 yes, 0 no)
% 8. reaction_time      (seconds)
% 9. signed contrasts

fprintf('Loaded file %s.\n', filename);
fprintf('Total # of trials: %d\n', size(data,1));
fprintf('Sessions: %s\n', mat2str(unique(data(:,2))'));
data(1:5,:)

%% Inspecting the data

% The first thing to do with any dataset is to get familiar with it by 
% running simple visualizations. Just plot stuff!
% Here we plot data from individual sessions. What can we see from here?

figure(1);
subplot(1,2,1);
plot_psychometric_data(data, 2);
subplot(1,2,2);
plot_psychometric_data(data, 15);

%% Psychometric function model

% Now let's try plotting the psychometric function model for different 
% values of the parameters (use both the symmetric and asymmetric 
% psychometric function).
% The parameters are: [bias, slope/noise, lapse rate, lapse bias]
% Try and match the data from one of the sessions.

theta0 = [-20,40,0.2,0.5]; % Arbitrary parameter values - try different ones
session_num = 15;

close all;
figure(1);
plot_psychometric_data(data,session_num);
hold on;
psychofun_plot(theta0);
legend('Location','NorthWest','Box','off','FontSize',12);

%% Psychometric function log-likelihood

% The psychofun_loglike.m file defines the log likelihood function of the 
% psychometric function model for a given dataset and model parameter 
% vector, log p(data|theta).

% Now try to get the best fit for this session, as we did before, but by 
% finding better and better values of the log-likelihood. 
% Higher values of the log-likelihood are better.

session_num = 14; % Let's use a different session
theta0 = [-20,40,0.2,0.5];
session_data = data(data(:,2) == session_num,:);

ll = psychofun_loglike(theta0,session_data);
% print('Log-likelihood value: ' + "{:.3f}".format(ll))

close all;
figure(1);
plot_psychometric_data(data,session_num);
hold on;
psychofun_plot(theta0);
legend('Location','NorthWest','Box','off','FontSize',12);
text(-100,0.7,['Log-likelihood: ' num2str(ll)],'FontSize',12);

%% Maximum-likelihood estimation

% In this section, we are going to estimate model parameters (aka fit our 
% models) by maximizing the log-likelihood. By convention in optimization, 
% we are going to minimize the negative log-likelihood.

% Before running the optimization, we define the hard lower and upper 
% bounds for the parameters. If the optimization algorithm supports 
% constrained (bound) optimization, it will never go outside the hard bounds. 
% We also define informally the "plausible" bounds as the range of 
% parameters that we would expect to see. We are going to use the plausible 
% range to initialize the problem later.

% Define hard parameter bounds
lb = [-100,1,0,0];
ub = [100,100,1,1];

% Define plausible range
plb = [-25,5,0.05,0.2];
pub = [25,25,0.40,0.8];

% Number of variables
D = numel(lb);

% Pick session data
session_num = 14;
session_data = data(data(:,2) == session_num,:);

% Define objective function: negative log-likelihood
opt_fun = @(theta_) -psychofun_loglike(theta_,session_data);

% We are now going to run a black-box optimization algorithm called BADS. 
% For now we are going to run the optimization only once, but in general you 
% should always run the optimization from multiple distinct starting points.

% Generate random starting point inside the plausible box
theta0 = rand(1,D).*(pub-plb) + plb;

% Set BADS options
options = bads('defaults');
options.Display = 'iter';

% Run optimization
[theta,fval] = bads(opt_fun,theta0,lb,ub,plb,pub,options);

fprintf('Returned parameter vector: %s\n', mat2str(theta,3));
fprintf('Negative log-likelihood at solution: %s', num2str(fval));

close all;
figure(1);
plot_psychometric_data(data,session_num);
hold on;
psychofun_plot(theta);
legend('Location','NorthWest','Box','off','FontSize',12);
text(-100,0.7,['Log-likelihood: ' num2str(ll)],'FontSize',12);