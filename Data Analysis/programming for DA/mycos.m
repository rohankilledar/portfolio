% WRITE HERE YOUR FUNCTION FOR EXERCISE 3
function mycos(np1,np2)
color = menu('Choose colour','blue','green','red');
  switch color
        case 1
            pColor = 'bo-';
        case 2
            pColor = 'go';
        case 3
            pColor = 'ro-';
        otherwise
            fprintf('wrong selection');
  end
    x1 = linspace(-pi,pi,np1);
    x2 = linspace(-pi,pi,np2);
    subplot(2,1,1);
    plot(x1,cos(x1),pColor);
    title(strcat(string(np1),' points'));
    grid on; 
	axis equal;
    subplot(2,1,2);
    plot(x2,cos(x2),pColor);
    title(strcat(string(np2),' points'));
    grid on;
	axis equal;
end