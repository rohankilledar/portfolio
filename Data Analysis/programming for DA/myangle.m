% WRITE HERE YOUR FUNCTION FOR EXERCISE 6
function answer =  myangle(val1,val2)
mini = min(val1,val2);
maxi = max(val1,val2);
answer = (mini:maxi)';
radian = (answer * pi /180);
answer = [answer radian];
print2col(answer)
end

function print2col(mat)
fprintf("Degree \t Radian \n")
tMat = mat';
fprintf("%d \t %f \n", tMat(:,:));
end