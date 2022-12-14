function plot_psychometric_data(data,session_num)
%PLOT_PSYCHOMETRIC_DATA Plot psychometric data as a line plot.

if nargin < 2; session_num = []; end

fontsize = 16;
axisfontsize = 12;

if isempty(session_num)
    trial_mask = true(size(data,1),1);      % Select all trials
else
    trial_mask = data(:,2) == session_num;  % Select trials from session
end

% Extract trials we are going to plot
plot_data = data(trial_mask,:);
Ntrials = size(plot_data,1);

% Count "left" and "right" responses for each signed contrast level
signed_contrasts_levels = unique(plot_data(:,9))';

response_choice = plot_data(:,6);
signed_constrasts = plot_data(:,9);

% Count left/right responses for each signed contrast level
for idx = 1:numel(signed_contrasts_levels)
    left_resp(idx) = sum((signed_constrasts == signed_contrasts_levels(idx)) & (response_choice == -1));
    right_resp(idx) = sum((signed_constrasts == signed_contrasts_levels(idx)) & (response_choice == 1));
end
frac_resp = right_resp ./ (left_resp + right_resp);
err_bar = 1.96*sqrt(frac_resp.*(1-frac_resp)./(left_resp + right_resp)); % Why this formula for error bars?

% Make error bar plot
errorbar(signed_contrasts_levels,frac_resp,err_bar,'LineWidth',1,'DisplayName','data');
xlabel('Left   \leftarrow   Signed contrast (%)   \rightarrow   Right','FontSize',fontsize)
ylabel('Rightward response probability','FontSize',fontsize)
if isempty(session_num)
    title(['Psychometric data (# trials = ' num2str(Ntrials) ')'],'FontSize',fontsize);
else
    title(['Psychometric data (session ' num2str(session_num) ', # trials = ' num2str(Ntrials) ')'],'FontSize',fontsize);
end
xlim([-105,105]);
ylim([0,1]);

Nc = numel(signed_contrasts_levels);
for i = 1:Nc
    xticklabel{i} = num2str(signed_contrasts_levels(i));
end

% Thin x ticks labels around 0 if too dense (ad hoc)
if Nc >= 9
    xticklabel{(Nc+1)/2-1} = [];
    xticklabel{(Nc+1)/2+1} = [];
end

set(gca,'XTick', signed_contrasts_levels);
set(gca,'XTickLabel',xticklabel);
set(gca,'TickDir','out','FontSize',axisfontsize);
box off;
set(gcf,'Color','w');

end