% CS5810J  -- Programming for data analysis (January)
%
%  Assignment 1 | Dr. Matteo Sammartino
%  Deadline: February 18, 2021, at 10:00 am
%
% 
% Insert BELOW your code for exercise 1, 5, 6
% The templates for functions for exercises 2, 3 and 4 are provided in separate files.
%


%% ================== Exercise 1 ==================


% INSERT YOUR CODE HERE!
sales = readmatrix('salesfigs.dat');
A = sales(1,:);
B = sales(2,:);
Quarter =(1:0.5:4);
plot(A, 'ko');
hold on
plot(B, 'k*');
title('ABC Corporation Sales: 2013')
xlabel('Quarter')
ylabel('Sales(billions)')
legend('Division A','Division B')


%% ================== Exercise 5 ==================


% INSERT YOUR CODE HERE!

%X = randi([10 35],5000,1);
%Y = randi([10 35],5000,1);
min = 10;
max =35;

X = min + (max -min).*rand(5000,1);
Y = min + (max -min).*rand(5000,1);

plot(X,Y, 'g*')
axis([-10 40 -5 40])
hold on
Xn = 1.*randn(5000,1) +2;
Yn = 1.*randn(5000,1) +7;

%Xn = mean(2,7).*randn(5000,1) +1 ;
%Yn = mean(2,7).*randn(5000,1) +1 ;
plot(Xn,Yn , 'r*')

%% ================== Exercise 6 ==================


% INSERT YOUR CODE HERE!
mini =-3;
maxi =5;

X = mini + (maxi -mini).*rand(10000,1);
Y = mini + (maxi -mini).*rand(10000,1);
Z = mini + (maxi -mini).*rand(10000,1);

xlog = X>=0;
ylog = Y>=0;
zlog = Z>=0;
    
ind=and(and(xlog,ylog),zlog) ;

xpos = X(ind==1);
ypos = Y(ind==1);   
zpos = Z(ind==1);

xnorm=(1:length(xpos));
ynorm=(1:length(ypos));
znorm=(1:length(zpos));
% xnorm = normalize(xpos,1);
% ynorm = normalize(ypos,1);
% znorm = normalize(zpos,1);
% scatter3(xpos,ypos,zpos, 'r*')
% 
% xnorm = xpos./norm([xpos,ypos,zpos],2);
% ynorm = ypos./norm([xpos,ypos,zpos],2);
% znorm = zpos./norm([xpos,ypos,zpos],2);
for i = 1:length(xpos)
xnorm(i) = xpos(i)./norm([xpos(i),ypos(i),zpos(i)],2);
ynorm(i) = ypos(i)./norm([xpos(i),ypos(i),zpos(i)],2);
znorm(i) = zpos(i)./norm([xpos(i),ypos(i),zpos(i)],2);
end

scatter3(xnorm,ynorm,znorm, 'r*')
