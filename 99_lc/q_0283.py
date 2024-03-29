from typing import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        n = len(nums)
        l = r = 0
        while r < n:
            if nums[r] != 0:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
            r += 1


s = Solution()
nums = [0, 1, 2, 0, 5, 0, 7,]
s.moveZeroes(nums)
print(nums)
