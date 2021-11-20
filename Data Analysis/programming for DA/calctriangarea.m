% CS5810J  -- Programming for data analysis (January)
%
%  Assignment 1 | Dr. Matteo Sammartino
%  Deadline: February 18, 2021, at 10:00 am
%
% 
% Insert BELOW your function for exercise 2


% INSERT YOUR CODE HERE
function areas = calctriangarea(bases,heights)
 size = length(bases);
 if size == length(heights)
  areas = (bases.*heights)/2;
 else
     disp('Please enter vectors with same lengths')
end

