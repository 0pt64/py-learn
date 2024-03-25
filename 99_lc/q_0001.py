from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mp = dict()
        for i, num in enumerate(nums):
            if target - num in mp:
                return [mp[target - num], i]
            mp[num] = i
        return []


nums, target = [1, 2, 3, 4, 5, 6], 8
s = Solution()
print(s.twoSum(nums, target))
