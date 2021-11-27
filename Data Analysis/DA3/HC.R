# Importing the dataset ---------------------------------------------------
nci.data <- read.table("nci.data.txt", quote="\"", comment.char="")

dim(nci.data)
nci.label <- read.table("label.txt")

k<- length(table(nci.label))
k
# data pre-processing ------------------------------------------------------
#transposing the dataset as its repesented in other way.
nci.data <- t(nci.data)
dim(nci.data)

# distance function -------------------------------------------------------
euclideanDist <- function(x,y){
  return (sqrt(sum((x-y)^2)))
}


# Calculating distance matrix ---------------------------------------------

distanceMatrix <- function(dataset){
  dataset <- as.matrix(dataset)
  n <- nrow(dataset)
  distMatrix <- matrix(NA,nrow = n,ncol =n)
  for(i in 1:n){
    for(j in i:n){
      if(i != j){
        distMatrix[i,j] = euclideanDist(dataset[i,],dataset[j,])
        distMatrix[j,i] = euclideanDist(dataset[i,],dataset[j,])
      }
      
    }
  }
  # rownames(distMatrix)<- c(1:n)
  # colnames(distMatrix) <- c(1:n)
  
  #distMatrix<- distMatrix[-1,]
  return(distMatrix)
}

# hac ---------------------------------------------------------------------

data <- nci.data
n <- nrow(data)

hc <- function(data, k, linkage){

    
  distMat <- distanceMatrix(nci.data)
  distMat
  
  cList <- list()
  for (i in 1:nrow(data)) {cList[[i]] <- i}
  
  cList
  
  status <- c(rep(1,nrow(data)))
  status
  
  clustDistMat <- matrix(NA,2*n-k,2*n-k)
  
  clustDistMat[1:nrow(distMat),1:nrow(distMat)] <- distMat
  
  LoopcDistMat <- clustDistMat
  
  for (i in 1:(nrow(data)-k)) {
    
    print(paste("level:",i-1,sep=' '))
    
    # find the two cluster with minimum cluster distance
    minClust <- which(LoopcDistMat==min(LoopcDistMat, na.rm=TRUE), arr.ind=T)
    
    
    print(paste("height:",min(LoopcDistMat, na.rm = TRUE),sep=' '))
    
    # update cluster list with the new cluster found minClust
    merge <- c(cList[[minClust[nrow(minClust),1]]],cList[[minClust[nrow(minClust),2]]])  
    
    print(paste('merge:', merge[1],merge[2], sep= ' '))
    cList[[nrow(data)+i]] <- merge
    
    # update the status as used 
    status[minClust[nrow(minClust),]] <- 0
    status[nrow(data)+i] <- 1
    # to merge clusters and update ClusterDistanceMatrix
    
    clustDistMat <- clusterDistanceFunction(data, clustDistMat, cList, linkage)  
    # remove the new clust
    LoopcDistMat[minClust[nrow(minClust),],] <- NA
    LoopcDistMat[,minClust[nrow(minClust),]] <- NA
    LoopcDistMat[(nrow(data)+i),1:(nrow(data)+i)] <- clustDistMat[(nrow(data)+i),1:(nrow(data)+i)]
    LoopcDistMat[1:(nrow(data)+i),(nrow(data)+i)] <- clustDistMat[1:(nrow(data)+i),(nrow(data)+i)]
  }
  
  finClu <- cList[status==1]
  
  finClu1 <- c()
  for (i in 1:length(finClu)) finClu1[finClu[[i]]] <- i
  cat('The Cluster Number of each observation:\n',finClu1,'\n')
  return(list(finClu1, finClu))
}


# creating distance/proximity matrix for given linkage ------------------------------

clusterDistanceFunction <- function(data, clustDistMat, cList, linkage){
  
  CluNum <- length(cList)
  # finding the value of the last cluster added / new cluster found
  newClu <- cList[[length(cList)]]
  
  if(linkage == 'single'){
    # loop over all the clusters except the new one
    for (i in c(1:(CluNum-1))[-newClu]) {
      tmp <- c()
      tmp.c <- 1
      # loop over the clusters
      for (j in cList[[i]]) {
        # loop update the value of clust distance matrix only for the new clust
        for (k in newClu) {
          tmp[tmp.c] <- clustDistMat[j,k]
          tmp.c <- tmp.c+1
        }
      }
      # find the minimum distance between two cluster i.e. single linkage
      clustDistMat[CluNum,i] <- min(tmp)
      clustDistMat[i,CluNum] <- min(tmp)
    }
    return(clustDistMat)
  }
  if(linkage == 'complete'){
    for (i in c(1:(CluNum-1))[-newClu]) {
      tmp <- c()
      tmp.c <- 1
      
      for (j in cList[[i]]) {
        for (k in newClu) {
          tmp[tmp.c] <- clustDistMat[j,k]
          tmp.c <- tmp.c+1
        }
      }
      
      clustDistMat[CluNum,i] <- max(tmp)
      clustDistMat[i,CluNum] <- max(tmp)
    }
    return(clustDistMat)
  }
  if(linkage == 'average'){
    for (i in c(1:(CluNum-1))[-newClu]) {
      tmp <- c()
      tmp.c <- 1
      
      for (j in cList[[i]]) {
        for (k in newClu) {
          tmp[tmp.c] <- clustDistMat[j,k]
          tmp.c <- tmp.c+1
        }
      }
      
      clustDistMat[CluNum,i] <- sum(tmp)/(length(cList[[i]])*length(newClu))
      clustDistMat[i,CluNum] <- sum(tmp)/(length(cList[[i]])*length(newClu))
    }
    return(clustDistMat)
  }
  if(linkage == 'centroid'){
    
    for (i in c(1:(CluNum-1))[-newClu]) {
      
      clustDistMat[CluNum,i] <- abs(mean(data[cList[[i]],])-mean(data[newClu,]))
      clustDistMat[i,CluNum] <- abs(mean(data[cList[[i]],])-mean(data[newClu,]))
    }
    return(clustDistMat)
  }
  
}




# hierarchical agglomerative clustering -----------------------------------

clust.single <- hc(data, k, 'average')
# 
# 
# ogIndx<- which(nci.label[[1]] ==  "CNS")
# ogIndx
# 
# predIndx <- which(clust.single[[1]] %in% c(clust.single[[1]][1]))
# predIndx
# 
# #sum(!(ogIndx %in% predIndx))/nrow(data)

#correct <- intersect(ogIndx,predIndx)

# indx = which(clust.single[[1]] %in% c(14))
# nci.label[[1]][indx]
# 
# 
# pred<-clust.single[[1]] %in% c(14)
# 
# og<-nci.label[[1]] %in% "CNS"
# 
# og
# pred
# pred!=og

clust.complete <- hc(data, k, 'complete')

clust.average <- hc(data, k, 'average')

#taking too long to compute after level 19
clust.centroid <- hc(data, k, 'centroid')



# apply k-means 
for( i in 2:nrow(data)-1 ){
  kmeanOutput <- kmeans(data,i)
  as.vector(kmeanOutput$cluster)
  print(paste("for K=",i,sep = ' '))
  cat(kmeanOutput$cluster)
  cat("\n")
}

#when k =14 to compare the Kmean with hierarchical agglomerative clustering

kmeanOutput <- kmeans(data,k)
as.vector(kmeanOutput$cluster)

