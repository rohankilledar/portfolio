  % WRITE YOU CODE HERE

function myplot(xVec,yVec)
p = load('plot_properties.mat');
%disp(plotProp)
 plotType = { p.plot_properties.plottype};
 plotProperties = { p.plot_properties.plotproperties};
 
 
%using the ASCII value to get single quotes in q so as to avoid the
%confusion while make strings to evaluate.
q = char(39);

% making up strings to evaluate for each subplot
subplot(3,1,1)
plotString1 = strcat(plotType{1},'(xVec,yVec,',q,plotProperties{1}.Color,plotProperties{1}.LineStyle,q,',',q,'LineWidth',q,',',num2str(plotProperties{1}.LineWidth),')');
eval(plotString1)
grid on

subplot(3,1,2)

plotString2 = strcat(plotType{2},'(xVec,yVec,',q,'FaceColor',q,',',q,plotProperties{2}.FaceColor,q,',',q,'EdgeColor',q,',',q,plotProperties{2}.EdgeColor,q,',',q,'BarWidth',q,',',num2str(plotProperties{2}.BarWidth),')');
eval(plotString2)
axis([-8 8 -1 1])

grid on

subplot(3,1,3)

plotString3 = strcat(plotType{3},'(xVec,yVec,',q,'FaceColor',q,',',q,plotProperties{3}.FaceColor,q,',',q,'EdgeColor',q,',',q,plotProperties{3}.EdgeColor,q,',',q,'BarWidth',q,',',num2str(plotProperties{3}.BarWidth),')');
eval(plotString3)
axis([-1 1 -10 10])
grid on

end
