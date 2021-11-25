# Loading the library ----------------------------
LoadLibraries <- function(){
  library(MASS)
  print("The libraries are now loaded")
}
LoadLibraries()
attach(Boston)
##

# Splitting the data ------------------------------------------------------

# Boston.data <- subset(Boston,select= c("lstat", "rm"))  
Boston.label <- subset(Boston, select = c("medv"))  
Boston.data <- subset(Boston,select= c("lstat", "rm","medv"))  

set.seed(2501)
indexTrain <- sample(1:nrow(Boston.data), size = nrow(Boston.data)/2)
X <- Boston.data[indexTrain,]
X_test <- Boston.data[-indexTrain,]
y <- Boston.label[indexTrain,]
# y_test <- Boston.label[-indexTrain,1]
# DS function -------------------------------------------------------------
DS <- function(data){
  best_RSS_l = Inf
  
  #calculate the best RSS for rm attribute
  for ( s in data[,1]){
    
    
    yhatlt <- subset(data, lstat<s)
    yhatltMean = mean(unlist(yhatlt["medv"]))
    
    yhatge <- subset(data, lstat>=s)
    yhatgeMean = mean(unlist(yhatge["medv"]))
    
    #for values where we get nothing less that the current S the subset returns NaN values, 
    #in such cases as there are no numbers we take the mean to be 0
    #for example, this happens for s=1.73 in case of lstat
    
    if(is.na(yhatltMean)){sumofsquareofdiffLT <- 0}else sumofsquareofdiffLT <- sum((yhatlt["medv"]-yhatltMean)^2)
    
    #this is not needed though as we will have 1 value atleast for Greater equal S
    if(is.na(yhatgeMean)){sumofsquareofdiffGE<-0}else sumofsquareofdiffGE <- sum((yhatge["medv"] - yhatgeMean)^2) 
    
    rss = sumofsquareofdiffLT + sumofsquareofdiffGE 
    
    if(best_RSS_l > rss){
      best_RSS_l = rss
      best_S_l = s
    }
  }
  
  #calculate the best RSS for rm attribute
  best_RSS_r = Inf
  
  for ( s in data[,2]){
    
    
    yhatlt <- subset(data, rm<s)
    yhatltMean = mean(unlist(yhatlt["medv"]))
    
    yhatge <- subset(data, rm>=s)
    yhatgeMean = mean(unlist(yhatge["medv"]))  
    
    #for values where we get nothing less that the current S the subset or mean returns NaN values, 
    #in such cases as there are no numbers we take the mean to be 0
    #for example, this happens for s=1.73 in case of lstat
    
    if(is.na(yhatltMean)){sumofsquareofdiffLT <- 0}else sumofsquareofdiffLT <- sum((yhatlt["medv"]-yhatltMean)^2)
    
    #this is not needed though as we will have 1 value atleast for Greater equal S
    if(is.na(yhatgeMean)){sumofsquareofdiffGE<-0}else sumofsquareofdiffGE <- sum((yhatge["medv"] - yhatgeMean)^2) 
    
    rss = sumofsquareofdiffLT + sumofsquareofdiffGE 
    
    if(best_RSS_r > rss){
      best_RSS_r = rss
      best_S_r = s
    }
  }
  #cat(best_RSS_l,best_RSS_r,best_S_l,best_S_r)
  output = list()
  if(best_RSS_l < best_RSS_r){
    output[1] = 1
    output[2] = best_S_l
  }else{
    output[1] = 2
    output[2] = best_S_r
  }
  return(output)
}


# Getting the values for training set -------------------------------------

out <- DS(X)
colNo <- out[1]
best_S <- out[2]
colNo <-as.integer(colNo)


# Calculating Test MSE ----------------------------------------------------

MSE <- function(data,test_data, s, col){
  
  yhatlt <- subset(data, data[,col]<s)
  yhatge <- subset(data, data[,col]>=s)
  yhatltMean = mean(unlist(yhatlt["medv"]))
  yhatgeMean = mean(unlist(yhatge["medv"]))
  
  yhatltTest <- subset(test_data, test_data[,col]<s)
  yhatgeTest <- subset(test_data, test_data[,col]>=s)
  
  
  
  rss = sum((yhatltTest["medv"]-yhatltMean)^2) + sum((yhatltTest["medv"] - yhatgeMean)^2) 
  
  return(rss/nrow(test_data))
}

# calling the test MSE function -------------------------------------------

testMSE <- MSE(X,X_test,best_S,colNo)
testMSE

# BDS function ------------------------------------------------------------

BDS <- function(data,B=1000, learning_rate = 0.01){
  fx = 0
  row.names(data) <- 1:nrow(data)
  r <- data["medv"]
  decision_stump <- 1:B
  col <- 1:B
  for (i in 1:B){
    
    out <- DS(data)
    decision_stump[i] <- as.numeric(out[2])
    col[i] <- as.numeric(out[1])
    
    yhatlt <- subset(data, data[,col[i]]<decision_stump[i])
    yhatge <- subset(data, data[,col[i]]>=decision_stump[i])
    
    yhatltMean = mean(unlist(yhatlt["medv"]))
    yhatgeMean = mean(unlist(yhatge["medv"]))
    
    yhatlt["medv"] <- yhatltMean
    yhatge["medv"] <- yhatgeMean
    
    indxLT = as.list(as.numeric(rownames(yhatlt)))
    indxGE = as.list(as.numeric(rownames(yhatge)))
    
    for( i in indxLT ){
      data[i,3]= yhatltMean
    }
    
    for( i in indxGE ){
      data[i,3]= yhatgeMean
    }
    
    
    r <- r - learning_rate*data[,3]
    data["medv"] <- r
    
    fx <- fx + learning_rate*data[,3]
    
    
  }
  return(fx)
}


# calling BDS function ----------------------------------------------------



# calculating the Test MSE for BDS ----------------------------------------

testMSE <- function(train_data,test_label,B=1000)
{
  tMSE <- sum((test_label - BDS(train_data,B))^2)/length(test_label)
  return(tMSE)
}


# calling TestMSE function ------------------------------------------------

testM <- testMSE(X,y,1000)
testM


# Finding the TestMSE for the corresponding values of B -----------------------------------------
B<-1:100

testM<-0
for( i in B)
{
  testM[i] <- testMSE(X,y,i)
  cat(i,"->",testM[i],"\n")
}

# plotting the points on graph as a line ----------------------------------



plot(B,testM,"l",xlab = "Number of Trees",ylab = "Test MSE",col=149)


