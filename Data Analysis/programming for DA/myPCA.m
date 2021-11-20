% WRITE YOU CODE HERE
%part1 task4

function [eVec, eVal] = myPCA(X)
%finding the covariance matrix using cov
covMat = cov(X);

%finding the eigenVector and eigenValue using eig
[eigVec,eigVal] = eig(covMat);

%sorting the eigenValues in descending order and using the index of those
%sorted values to sort the eigenVectors.
[eVal, ind] = sort(diag(eigVal),'descend');
eVec = eigVec(:, ind);
end