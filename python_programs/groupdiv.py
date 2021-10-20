#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'groupDivision' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY levels
#  2. INTEGER maxSpread
#

#write a function to divide a class of student into groups based on their skill level given as an array and max spread that is allowed in each group

def groupDivision(levels, maxSpread):
    
    levels.sort()
    classes = []
    i=0
    group =[levels[i]]
    # Write your code here
    while i < len(levels)-1:
        
        if levels[i+1] in range(group[0],group[0]+maxSpread+1):
            group.append(levels[i+1])
            i+=1
        else:
            classes.append(group)
            group =[levels[i+1]]
            i+=1
            
          
    return len(classes)+1
        
    
if __name__ == '__main__':
    level = [1,3,4,4,7]
    maxs=2
    print(groupDivision(level,maxs))