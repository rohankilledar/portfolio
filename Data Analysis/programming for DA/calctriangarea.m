
function areas = calctriangarea(bases,heights)
 size = length(bases);
 if size == length(heights)
  areas = (bases.*heights)/2;
 else
     disp('Please enter vectors with same lengths')
end

