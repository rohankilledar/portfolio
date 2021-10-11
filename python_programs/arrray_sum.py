# Problem Statement:"Determine if an integer array has an index where the sum of the values before the index equal the sum of the values after the index"

def arraySum(arr):
    for indx in range(len(arr)):
        if(sum(arr[:indx])==sum(arr[indx+1:])):
            return indx
    return False

if __name__ == "__main__":
    sample_arr = [4,3,6,5,2]
    print(arraySum(sample_arr))