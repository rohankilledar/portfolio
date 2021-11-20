% CS5810J  -- Programming for data analysis (January)
%
%  Assignment 1 | Dr. Matteo Sammartino
%  Deadline: February 18, 2021, at 10:00 am
%
% 
% Insert BELOW your function for exercise 3


% INSERT YOUR CODE HERE

function conv = conversion(measure, vector)

if measure == 'o' 
   conv = vector/0.035274;
   
 elseif measure == 'g' 
   conv = vector*0.035274;
     
else
    disp("Please provide measurement string as either 'o' or 'g' "); 
end
end

