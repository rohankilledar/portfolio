% WRITE HERE YOUR SCRIPT FOR EXERCISE 7
n=1;
approxE=0;
while(abs(exp(1)- approxE) >10^-5)
    n=n+1;
    approxE=(1-1/n)^-n;
    
end
fprintf("the build in value of e is %f, \n approximation of e to 5 decimal places : %.5f \n value of n to find this accuracy: %d \n ", exp(1),approxE,n);
