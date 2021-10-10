def min_pieces(original, desired):
    count=1
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
                    break
    return count
                
            

original = [1, 4, 3, 2]
desired = [1, 2, 4, 3]
print(min_pieces(original, desired))