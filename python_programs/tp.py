#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'minTime' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY files
#  2. INTEGER numCores
#  3. INTEGER limit
#

def minTime(files, numCores, limit):
    # Write your code here
    
    maxi = 0
    divis = []
    temp = list(files)
    indxl=[]
    for indx,f in enumerate(files):
        if f % numCores ==0:
            divis.append(indx)
    divisi = len(divis)
    while(limit>0 and divisi>0):
        for indx in divis:
            if temp[indx]>maxi:
                maxi = temp[indx]
                maxind = indx
        indxl.append(maxind)
        temp[maxind] = 0        
        limit -=1
        divisi -=1
            # if f > maxi:
            #     maxi = f
            #     maxind = indx
                
        
    
    for i in indxl:
        files[i] = files[i]/numCores
    
    return int(sum(files))
        
    

if __name__ == '__main__':