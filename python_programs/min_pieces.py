def min_pieces(original, desired):
    count=0
    len_of_strip=len(original)
    for i in range(len(desired)):
        #print("i:"+ i)
        indx = original.index(desired[i])
        if indx == len_of_strip-1:
            count+=1
        else:
            for j in range(len_of_strip-indx-1):
                #print("j:"+j)
                if desired[i+j] != original[indx+j]:
                    count+=1
                    
    return count
                
            
# original = [1, 4, 3, 2]
# desired = [1, 2, 4, 3]

original = [1, 4, 3, 2,5,6,9,8]
desired = [1, 2, 4, 3,6,9,5,8]
print(min_pieces(original, desired))