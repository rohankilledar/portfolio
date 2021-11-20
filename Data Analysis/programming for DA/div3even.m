% CS5810J  -- Programming for data analysis (January)
%
%  Assignment 1 | Dr. Matteo Sammartino
%  Deadline: February 18, 2021, at 10:00 am
%
% 
% Insert BELOW your function for exercise 4


% INSERT YOUR CODE HERE

function div = div3even(n)
vec = randi([0,50],1,n);
evenVec = vec(2:2:length(vec));
index = ~mod(evenVec,3)==1;
div = evenVec(index);
end
