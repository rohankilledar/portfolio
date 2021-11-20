% WRITE YOU CODE HERE

% I have done the two parts of this question in two sections in MATLAB so we can run the
% two part seperately in the same file. Also, using figures to output each
% figure in its own individual window.

k=1;
%part1 task1
pcaData = load("pcadata.mat");
X = pcaData.X;

%part1 task2
%subplot(2,1,1)
figure
plot(X(:,1),X(:,2),'bo')
hold on
xlabel('Figure A');
title('Datapoints and their 2 principal components');
axis([0 7 2 8])

%part1 task3
[Xmu,mu] = subtractMean(X);

%part1 task4
[U,S] = myPCA(Xmu);

%part1 task5

%adding the mean to each col to plot the PCA on graph.
eVecPlot = U + transpose(mu);

%as mu is the centrold, drawing a line from centroid to the PCA for each
%PCA.

%line for PCA1 with highest eigenValue
x = [eVecPlot(1,1), mu(1)];
y =[eVecPlot(2,1), mu(2)];
line(x,y,'Color','r','LineWidth',2);
%line for PCA2
x = [eVecPlot(1,2), mu(1)];
y =[eVecPlot(2,2), mu(2)];
line(x,y,'Color','g','LineWidth',2);
hold off

%%part1 task6
Z = projectData(Xmu,U,k);

%%part1 task7 

%displaying the first 3 projected values
disp(Z(1:3, :));

%%part1 task8
Xrec = recoverData(Z, U, k, mu);

%%subplot(2,1,2)
figure
plot(X(:,1),X(:,2),'bo')
hold on 
xlabel('Figure B');
title('Datapoints and their reconstruction');
plot(Xrec(:,1),Xrec(:,2),'r*');
axis([0 7 2 8])
hold off

%% part 2
k=200;
%%task1
X = load("pcafaces.mat").X;

%%task2
figure
displayData(X(1:100, :))

title('Figure C');


%%task3
[Xmu,mu] = subtractMean(X);

%%task4
[U,S] = myPCA(Xmu);

Z = projectData(Xmu,U,k);

%%task 5
Xrec = recoverData(Z, U, k, mu);

%%task 6
figure
subplot(1,2,1)
displayData(X(1:100, :))
title('Original faces');

subplot(1,2,2)
displayData(Xrec(1:100, :))
title('Recovered faces');

