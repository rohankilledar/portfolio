% WRITE YOU CODE HERE
function plottrigs(varargin)
Defaults = {'sin','k',1,'.'};
Defaults(1:nargin) = varargin;
trigFun = Defaults{1};
color = Defaults{2};
thickness = Defaults{3};
shape = Defaults{4};

%using the ASCII value to get single quotes in q so as to avoid the
%confusion while make strings to evaluate.
q = char(39);

style = strcat(color,shape,'-');
plotString = strcat('plot(-2*pi:0.1:2*pi,', trigFun,'(-2*pi:0.1:2*pi),',q,style,q,',',q,'LineWidth',q,',',num2str(thickness),')');

eval(plotString);
title([num2str(nargin), ' input arguments']);
xlabel('x')
yl=strcat(trigFun,'(x)');
ylabel(yl);
grid on

end