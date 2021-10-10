
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        for indx,num in enumerate(nums):
            if num == target:
                return indx
        
        return -1

if __name__ == "__main__":
    num_list = [-1,0,3,5,9,12]
    target= 9
    print(Solution.search(Solution,num_list,target))