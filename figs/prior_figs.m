%% Plot priors

fontsize = 16;
pos = [489,343,560,420];

lb = 1;
ub = 6;
plb = 2;
pub = 4;
x = linspace(0,8,1001)';

for i = 1:3
    figure(1);
    
    switch i
        case 1; y = munifboxpdf(x,lb,ub);
        case 2; y = mtrapezpdf(x,lb,plb,pub,ub);
        case 3; y = msplinetrapezpdf(x,lb,plb,pub,ub);
    end

    plot(x, y, 'k-', 'LineWidth', 2);
    xlim([0,8]);
    ylim([0,0.5]);

    set(gca,'XTick',0);
    set(gca,'YTick',0);
    set(gca,'TickDir','out');
    xlabel('\theta','FontSize',fontsize);
    ylabel('p(\theta)','FontSize',fontsize);
    box off;
    set(gcf,'Color','w');

    set(gcf,'Position',pos);
    filename = ['prior' num2str(i) '.png'];
    saveas(gcf, filename, 'png');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
