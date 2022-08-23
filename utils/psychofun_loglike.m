function loglike = psychofun_loglike(theta,data)
%PSYCHOFUN_LOGLIKE Log-likelihood for psychometric function model.

s_vec = data(:,9); % Stimulus values (signed contrasts)
r_vec = data(:,6); % Responses

% Compute probability of rightward response for each trial
p_right = psychofun(theta,s_vec);

% Compute summed log likelihood for all rightwards and leftwards responses
loglike = sum(log(p_right(r_vec == 1))) + sum(log(1 - p_right(r_vec == -1)));

end