package dynamic;

import javafx.scene.effect.SepiaTone;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * @Auther: Think
 * @Date: 2019/1/15 22:29
 * @Description:
 * 无序数组中的最长子序列长度
 *
 * 如果不考虑时间复杂度，先将数组排序之后再查找
 * 考虑时间复杂度可以使用hashMap
 *
 */
public class LongestConsecutive {
    public static int solution(int[] nums){
        Set<Integer> set = new HashSet<>();
        int max = 0;
        for (int num:nums){
            set.add(num);
        }
        for (int num : nums)
            if (!set.contains(num - 1)) {
            int val = num;
            while (set.remove(val++)){
                max = Math.max(max,val-num-1);
            }

            }
            return max;
    }

    public static void main(String[] args) {
       int nums[] = {1,2,4,9,4,3};
        System.out.println(solution(nums));


    }
}
