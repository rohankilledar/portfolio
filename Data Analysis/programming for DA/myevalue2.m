% WRITE HERE YOUR SCRIPT FOR EXERCISE 7
n=0;
approxE=1;
while(abs(exp(1)- approxE) >10^-5)
    n=n+1;
    approxE=approxE+(1/factorial(n));
    
end

fprintf("the build in value of e is %f, \n approximation of e to 5 decimal places : %.5f \n value of n to find this accuracy: %d \n ", exp(1),approxE,n);
