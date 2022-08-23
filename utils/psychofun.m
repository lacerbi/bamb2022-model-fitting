function p_right = psychofun(theta,stim)
%PSYCHOFUN Psychometric function based on normal CDF and lapses.

mu = theta(1);              % bias
sigma = theta(2);           % slope/noise
lapse = theta(3);           % lapse rate
if numel(theta) == 4
    lapse_bias = theta(4);  % lapse bias
else
    % if theta has only three elements, assume symmetric lapses
    lapse_bias = 0.5;
end

% Probability of responding "rightwards", without lapses
p_right = normcdf(stim,mu,sigma);

% Add lapses
p_right = lapse*lapse_bias + (1-lapse)*p_right;

end