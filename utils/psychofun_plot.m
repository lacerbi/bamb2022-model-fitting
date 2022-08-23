function psychofun_plot(theta)
%PSYCHOFUN_PLOT Plot psychometric function model.

stim = linspace(-100,100,201);      % Create stimulus grid for plotting    
p_right = psychofun(theta,stim);    % Compute psychometric function values
plot(stim,p_right,'LineWidth',1,'Color','k','DisplayName','model');

end
