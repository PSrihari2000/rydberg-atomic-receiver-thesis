% Re-draws fig4_strongref.png from fig4_strongref.mat (no Monte Carlo re-run)
clear; close all;
load('fig4_strongref.mat');

%% ---- Plot ----
figure('Color','w','Position',[100 100 1100 650]);
hold on; grid on; box on;

cGD=[0.93 0.69 0.13]; cPGD=[0 0.45 0.74]; cCR=[0.85 0.33 0.10];
for pp = 1:2
    plot(SNR_dB,10*log10(NMSE_GD(pp,:)),'-.s','Color',cGD,'LineWidth',1.8,'MarkerSize',7,'HandleVisibility',ternary(pp==1,'on','off'));
    plot(SNR_dB,10*log10(NMSE_PGD(pp,:)),'-^','Color',cPGD,'LineWidth',1.8,'MarkerSize',7,'MarkerFaceColor',cPGD,'HandleVisibility',ternary(pp==1,'on','off'));
    plot(SNR_dB,10*log10(NMSE_CRLB(pp,:)),'-o','Color',cCR,'LineWidth',1.8,'MarkerSize',7,'HandleVisibility',ternary(pp==1,'on','off'));
end
legend({'GD','PGD','CRLB'},'Location','southwest');
i10 = find(SNR_dB==10); i5 = find(SNR_dB==5);
pilotEllipse(10,10*log10([NMSE_GD(1,i10) NMSE_PGD(1,i10) NMSE_CRLB(1,i10)]),'P = 10','right');
pilotEllipse(5,10*log10([NMSE_GD(2,i5) NMSE_PGD(2,i5) NMSE_CRLB(2,i5)]),'P = 30','left');

xlabel('SNR [dB]','FontWeight','bold');
ylabel('NMSE [dB]','FontWeight','bold');
set(gca,'FontSize',12,'LineWidth',1);

hold off;

exportgraphics(gcf,'fig4_strongref.png','Resolution',300);

function out = ternary(c,a,b)
if c, out = a; else, out = b; end
end

function pilotEllipse(x0,yvals,lbl,side)
% dashed ellipse around the curves of one pilot length at SNR = x0
yc = mean([min(yvals) max(yvals)]);
ry = (max(yvals)-min(yvals))/2 + 1.2;
rx = 0.7;
t = linspace(0,2*pi,200);
plot(x0+rx*cos(t),yc+ry*sin(t),'k--','LineWidth',1.2,'HandleVisibility','off');
if strcmp(side,'right')
    text(x0+rx+0.4,yc+ry,lbl,'FontWeight','bold','FontSize',12,'HorizontalAlignment','left');
else
    text(x0-rx-0.4,yc-ry,lbl,'FontWeight','bold','FontSize',12,'HorizontalAlignment','right');
end
end
