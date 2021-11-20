% WRITE HERE YOUR FUNCTION FOR EXERCISE 2
function answer = countdown()
persistent x 
if isempty(x)
    x=6;
end
x=x-1;
if x>0
    answer = strcat(string(x)," seconds left");
else
    answer = "countdown has expired";

end