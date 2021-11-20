% WRITE YOU CODE HERE
function myplotarea(filename,n)
formatSpec = '%s %f %s %f';
fileID = fopen(filename,'r');
op=textscan(fileID,formatSpec,'Delimiter',' ');
%taking the x and y values from the struct op
xValue = op{2};
yValue = op{4};

%condition to check if the number of points are less than or equal to n
if n <= length(xValue)
  
   xPlot = xValue(1:n);
   yPlot = yValue(1:n);
   area(xPlot,yPlot);
   title([num2str(n), ' data points']);
   grid on
else
    disp('The input n is greater than the number of points in the file provided! Please try with other value.')
end
%closing all the files that were open
fclose('all');

end
