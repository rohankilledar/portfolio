% WRITE HERE YOUR FUNCTION FOR EXERCISE 4
% WRITE HERE YOUR FUNCTION FOR EXERCISE 4
function answer = buildrandomstrings(n)
if n ~=0
    rstring = "";
    answer = cell(n,1);
    for i = 1:abs(n)
        %find the index of ASCII a-z using random integer function. here
        %a=97 and z=122
        randomLetter = randi([97 122]);
        rstring = strcat(rstring,char(randomLetter));
        if n>0
            answer(i) = cellstr(rstring);
        else
            %reverse indexing the random string generated
            answer((abs(n)+1)-i) = cellstr(rstring);
        end
    end
else
    disp("No values for n=0");
end