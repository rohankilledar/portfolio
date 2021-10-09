from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        output=[]
        for indx,i in enumerate(nums):
            indxNums = nums[:indx] + nums[indx+1:]
            if (target-i) in indxNums:
                output.append(indx)
        return output

if __name__ == "__main__":
    num_list = [2,7,11,15]
    target = 9
    print(Solution.twoSum(Solution,num_list,target))