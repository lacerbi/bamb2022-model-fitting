function psychofun_plot(theta)
%PSYCHOFUN_PLOT Plot psychometric function model.

N = size(theta,1);  % Number of samples (rows) of parameters
Nx = 201;           % Number of grid points for plotting

stim = linspace(-100,100,Nx);      % Create stimulus grid for plotting    

if N == 1   % Single sample (point estimate)
    p_right = psychofun(theta,stim);    % Compute psychometric function values
    plot(stim,p_right,'LineWidth',1,'Color','k','DisplayName','model');
else
    p_right = zero(N,Nx);        
    % Compute psychometric function values for each parameter setting
    for i = 1:N
        p_right(i,:) = psychofun(theta(i,:),stim);
    end
    
    median_pred = quantile(p_right,0.5,1);    % Median prediction
    
    
    
end
    
end
