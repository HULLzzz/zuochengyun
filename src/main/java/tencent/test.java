package tencent;
import com.sun.org.apache.bcel.internal.generic.IF_ACMPEQ;

import java.util.*;

/**
 * @Auther: Think
 * @Date: 2019/2/25 10:45
 * @Description:
 *
 * https://leetcode-cn.com/explore/featured/card/tencent/221/array-and-strings/894/
 * leetcode上的tencent探索内容
 */
public class test {

    //-----------------------------数组与字符串-------------------------------------------
    //num1:两数之和
    /*
    * 题目：
            给定一个整数数组和一个目标值，找出数组中和为目标值的两个数。
            你可以假设每个输入只对应一种答案，且同样的元素不能被重复利用。
            示例:
            给定 nums = [2, 7, 11, 15], target = 9
            因为 nums[0] + nums[1] = 2 + 7 = 9
            所以返回 [0, 1]
    * */

    public static int[] twoSum(int[] nums,int target){
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0;i<nums.length;i++){
            map.put(nums[i],i);
        }
        for (int i = 0;i<nums.length;i++){
            int another = target - nums[i];
            if (map.containsKey(another)&&map.get(another)!=i){
                return new int[] {i,map.get(another)};
            }
        }

        throw new IllegalArgumentException("No two sum solution");
    }

    /*
    * 拓展：三数之和
        给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
        注意：答案中不可以包含重复的三元组。
        例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，
        满足要求的三元组集合为：
        [
        [-1, 0, 1],
        [-1, -1, 2]
        ]

        思路：
        跟两数之和不同的是，三数之后要输出的是不同三元组的集合。
        因此，我们考虑先将nums进行排序，将nums[i]作为第一个加数，从i+1到nums.length-1之间初始化两个指针left，right，为了避免有重复的情况，当nums[i]==nums[i-1],说明有重复的情况，开始下一个循环。
        如果num[i]+num[left]+num[right]>0,说明加多了，让right–，如果num[i]+num[left]+num[right]<0,说明加少了，让left++，如果等于0，说明符合条件，将这一组解加到集合中，这是也应该避免第二个加数和第三个加数重复的情况。

    * */

    public static List<List<Integer>> solution(int[] nums ){
        List<List<Integer>> list = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        for (int i = 0;i<nums.length - 1;i++){
            if (i>0&&nums[i] == nums[i-1]){
                continue; //避免重复
            }
            int left = i+1;
            int right = nums.length - 1;
            while (left<right){
                if (nums[left]+nums[right]+nums[i]>0){
                    right -- ;
                }else if (nums[left]+nums[right]+nums[i]<0){
                    left ++;
                }else {
                    list.add(Arrays.asList(nums[left],nums[right],nums[i]));
                    left++;
                    right--;

                    while (left<right&&nums[left]== nums[left-1]){
                        left++;
                    }

                    while (left<right&&nums[right] == nums[right + 1]){
                        right--;
                    }
                }

            }
        }
        return list;
    }


    /*
      最接近的三个数
      给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。
        例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.
        与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
     */
    // 经典双指针问题
    // 将其重新排列，利用 left,right指针在中数为i,(起始left=0;right=nums.length -1; 1<= i <= nums.length -1;)中进行移动操作
    //           sum = nums[left] + nums[right] + nums[i]; 如果sum - target大于0，right--； 反之则left++；

    public int threeSumclose( int[] nums,int target ){
        Arrays.sort(nums);
        int clost = Integer.MAX_VALUE,sub = 0,abssub = 0,sum = 0;
        for (int i = 1;i<nums.length - 1;i++){
            int left = 0,right = nums.length - 1;
            while (left < i && right > i) {
                sub = nums[left] + nums[right] + nums[i] - target;
                abssub = Math.abs(sub); //记录和target最小的差值
                if (clost > abssub) {
                    clost = abssub;
                    sum = nums[left] + nums[right] + nums[i] ;
                }
                if (sub > 0) {
                    right--;
                } else if (sub < 0) {
                    left ++ ;
                }
                else {
                    sum = nums[left] + nums[right] + nums[i];
                    break;
                }
            }
        }
        return sum;
    }




    //num02:最长回文字串
    /*
     给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
     输入: "babad"
     输出: "bab"
     注意: "aba" 也是一个有效答案。

    O(n^2)时间复杂度方法——从中心向外扩散
    1.思想：
        1）将子串分为单核和双核的情况，单核即指子串长度为奇数，双核则为偶数；
        2）遍历每个除最后一个位置的字符index(字符位置)，单核：初始low = 初始high = index，low和high均不超过原字符串的下限和上限；判断low和high处的字符是否相等，相等则low++、high++（双核：初始high = 初始low+1 = index + 1）；
        3）每次low与high处的字符相等时，都将当前最长的回文子串长度与high-low+1比较。后者大时，将最长的回文子串改为low与high之间的；
        4）重复执行2）、3），直至high-low+1 等于原字符串长度或者遍历到最后一个字符，取当前截取到的回文子串，该子串即为最长的回文子串。

    一层循环：O(n-1) 嵌套两个独立循环： O(2n*(n-1))
     */
    private static int maxLen = 0;
    private static String sub = "";
    public static String longestPalindrome(String s){

        if (s.length()<=1){
            return s;
        }
        for (int i = 0;i<s.length()-1;i++){  //一层循环
            //以下为嵌套两个独立循环
            findLongestPalindrome(s,i,i); //回文子串为单数
            findLongestPalindrome(s,i,i+1); //回文子串为双数
        }
        return sub;
    }

    private static void findLongestPalindrome(String s, int low, int high) {
        while (low>=0&&high<=s.length()-1){
            if (s.charAt(low)==s.charAt(high)){
                if (high - low + 1 > maxLen) {
                    maxLen = high - low + 1;
                    sub = s.substring(low,high + 1);
                }
                low--; //向两边扩散找当前字符为中心的最大回文字串
                high++;
            }
            else
                break;
        }
    }

    //manacher算法：https://blog.csdn.net/weixin_42373330/article/details/82118694
    //添加特殊字符解决单双回文串问题，存放回文半径的数组
    //此算法有两点重要的内容：首先为了避免上个办法中的单双核子串的问题，将原来的字符串字符间距添加特殊字符
    //其次，使用辅助数组p[i]，记录以s[i]为中心的回文半径，如何计算p[i]呢？
    //由于我们之前使用特殊字符添加在字符串的字符间隙中，这个半径也就是我们所求的回文子串的长度
    //从左到右依次计算p[i]，但是p[j](j<i)已经计算过了，为了避免重复计算，引入id和max
    //id表示最大的回文子串的中心的位置，mx=id+p[id]。即回文子串的边界，mx指向的位置并不在最长回文子串中
    //就可以分为两种情况：i<=mx，i>mx

    public int getPalindromeLength(String str){
        //1. 构造新的字符串
        StringBuilder newStr = new StringBuilder();
        newStr.append('#');
        for (int i = 0;i<str.length();i++){
            newStr.append(str.charAt(i));
            newStr.append('#');
        }

        //定义最大回文半径
        int[] p = new int[newStr.length()];
        //定义已知回文中最大回文子串的边界
        int mx = -1;
        //又有mx边界的回文子串中心id
        int id = -1;
        //2.计算p数组中的所有回文半径
        //算法是o(n)的
        for (int i = 0;i<newStr.length();i++){
            //2.1 确定一个最小的半径 （若i<=mx有最小的回文半径）
            int r = 1;
            if (i<=mx){
                r = Math.min(p[id] - i + id,p[2*id - i]);
            }
            //2.2 尝试更大的半径
            while (i - r >= 0 && i + r < newStr.length() && newStr.charAt(i - r) == newStr.charAt(i + r)) {
                r++;
            }
            //2.3 更新边界和回文坐标
            if (i + r - 1 > mx) {
                mx = i + r - 1;
                id = i;
            }
            p[i] = r;
        }

        //3. 扫描p数组找出最大的半径
        int maxLength = 0;
        for (int r : p){
            if (r > maxLength) {
                maxLength = r;
            }
        }
        return maxLength - 1;
    }


    /*
    * 合并两个有序数组
    * 给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。
        说明:
        初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
        你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
        示例:
        输入:
        nums1 = [1,2,3,0,0,0], m = 3
        nums2 = [2,5,6],       n = 3
        输出: [1,2,2,3,5,6]
    *
    *
    * */

    public void merger(int[] a,int m,int[] b,int n){
        //这里的m和n表示数组中的元素，并不是数组的长度
        int i = m-1,j = n-1,k = m+n-1;
        while (i>-1&&j>-1){
            a[k--] = (a[i]>b[j])?a[i--]:b[j--];
        }
        while (j>-1){
            a[k--] = b[j--];
        }
    }


    /*
    * 螺旋矩阵
    * 给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。
        输入:
        [
         [ 1, 2, 3 ],
         [ 4, 5, 6 ],
         [ 7, 8, 9 ]
        ]
        输出: [1,2,3,6,9,8,7,4,5]
    * */

    public void printMartrix(int[][] array){

        if (array.length == 0) {
            return ;
        }


        int x = 0; //记录一圈开始的行
        int y = 0; //记录一圈开始的列

        while (x*2<array.length&&y*2<array[0].length){
            helper(array,x,y);
        }
    }

    private void helper(int[][] numbers, int x, int y) {
        int row = numbers.length; //行数
        int col = numbers[0].length; //列数

        //从左到右
        for (int i = y;i<col-y;i++){
            System.out.println(numbers[x][i]);
        }

        //从上到下
        if (row - 1 - x > x) {
            for (int i = x+1;i<=row-x-1;i++){
                System.out.println(numbers[i][col-1-y]);

            }
        }

        //从右到左
        if ((row - x - 1 > x) && (col - 1 - y > y)) {
            for (int i = col - 2 - y ;i>=y;i--){
                System.out.println(numbers[row-1-x][i]);
            }
        }

        //从左下到左上
        if ((row - 2 - x > x) && (col - 1 - y > y)) {
            for (int i = row - 2 - x;i>=x+1;i--){
                System.out.println(numbers[i][y]);
            }
        }
    }


    /*
    * 螺旋矩阵2：给出一个数n，从1~n^2按照顺时针生成矩阵
    * */

    public int[][] solution(int n){
        int[][] res = new int[n][n];
        //生成的矩阵是n*n矩阵
        int left = 0;
        int right = n-1;
        int up = 0;
        int down = n-1;
        int count = 1;
        while (up <= down && left <= right) {
            //右
            for (int i = left;i<=right;i++){
                res[up][i] = count++;
            }
            //下
            up++;
            for (int i = up;i<=down;i++){
                res[i][right] = count ++;
            }
            //左
            right -- ;
            for (int i = right;i>=left;i--){
                res[down][i] = count++;
            }
            //上
            down--;
            for (int i = down;i>=up;i--){
                res[i][left] = count++;
            }
            left++;
        }
        return res;
    }


    /*
    * 判断是否存在重复元素，是则返回true，否则返回false
    * */

    public static boolean solution02(int[] nums){
        //算法遍历一遍数组，遍历的过程中将元素按照升序的放到他相应的位置，此元素一定》=他之前的元素，如果不成立则说明有重复
        for (int i = 1;i<nums.length;i++){
            int j = i - 1;
            int temp = nums[j+1];
            //每次遍历将比temp大的元素依次往前挪，直到找到temp的位置
            while (j >= 0 && nums[j] > temp) {
                nums[j+1] = nums[j];
                j--;
            }
            nums[j+1] = temp;
            if (j>0&&nums[j]==nums[j+1]){
                return true;
            }
        }
        return false;
    }

    /*
    给定长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
    输入: [1,2,3,4]
    输出: [24,12,8,6]
    说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
    进阶：
    你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
    *
     */

    //对于每个nums[i]来说，对应的乘积是它左边的那部分乘以右边的那部分，然后用两次遍历
    // 分别用res和res2来存储某个元素左边和右边的乘积，比如对于nums[i],res[i]表示这个元素左边所有数字的乘积
    // res2[i]表示这个元素右边所有元素的乘积，最后乘起来就可以了，注意一下边界。

    //空间复杂度为N
    public int[] productExceptSelf(int[] nums){
        int len = nums.length;
        int[] res1 = new int[len];  //存储i左边的乘积
        int[] res2 = new int[len];  //存储i右边的乘积

        res1[0] = 1;
        for (int i = 1;i<len;i++){
            res1[i] = res1[i-1]*nums[i-1];
        }

        res2[len-1] = 1;
        for (int i = len-2;i>=0;i--){
            res2[i] = res2[i+1]*nums[i+1];
        }
        for (int i = 0;i<len;i++){
            res1[i]*=res2[i];
        }
        return res1;
    }

    //空间复杂度为常数，使用常数p保存每次计算的结果值
    public int[] productExceptSelf02(int[] nums){
        int len = nums.length,p;
        int[] arr = new int[nums.length];
        arr[0] = p = 1;
        for (int i  = 1; i<len;i++){
            p = p*nums[i-1];
            arr[i] = p;
        }
        p = 1;
        for (int i = len - 2;i>=0;i--){
            p = p*nums[i+1];
            arr[i] *= p;
        }
        return arr;
    }


    /*
        反转字符串
        示例 1：
        输入：["h","e","l","l","o"]
        输出：["o","l","l","e","h"]
     */

    public String reverseString(String s){
        if (s==null||s.length() == 0){
            return s;
        }
        StringBuilder sb = new StringBuilder(s);
        return sb.reverse().toString();
    }

    //进阶：Input: s = "abcde", k = 3
    //Output: "deabc"
    //先把3之前的部分逆序 cbade，再把3之后的部分逆序 cbaed ，整体逆序 deabc

    public void reverseString02(char[] chars,int k){
        if (chars == null || chars.length == 0 || k >= chars.length) {
            return ;
        }
        reverse(chars,0,k-1);
        reverse(chars,k,chars.length-1);
        reverse(chars,0,chars.length-1);

    }

    private void reverse(char[] chars, int start, int end) {
        char tmp = 0;
        while (start < end) {
            tmp = chars[start];
            chars[start] = chars[end];
            chars[end] = tmp;
            start++;
            end--;
        }
    }


    /*
        字符串相乘
        给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
        输入: num1 = "2", num2 = "3"
        输出: "6"
        示例 2:
        输入: num1 = "123", num2 = "456"
        输出: "56088"

     */

    public static String mutiply(String num1,String num2){
        //先将string反转
        String n1 = new StringBuilder(num1).reverse().toString();
        String n2 = new StringBuilder(num2).reverse().toString();

        int[] d = new int[n1.length()+n2.length()]; //构建数组存储乘积
        for (int i = 0;i<n1.length();i++){
            for (int j = 0;j<n2.length();j++){
                d[i+j] += (n1.charAt(i) - '0')*(n2.charAt(j)-'0');
            }
        }
       StringBuilder sb = new StringBuilder();
        for (int i = 0;i<d.length;i++){
            int digit = d[i]%10; //当前位
            int carry = d[i]/10; //进位
            sb.insert(0,digit); //insert方法是在0位置插入，若继续插入，则原来插入的元素向后移位
            if (i+1<d.length){
                d[i+1]+=carry;
            }
        }
        while (sb.length()>0&&sb.charAt(0)=='0'){
            sb.deleteCharAt(0);
        }
        return sb.length()==0?"0":sb.toString();
    }

    /*
    *
    * 给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。画 n 条垂直线，使得垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
    * */
    public int maxArea(int[] height){
        int maxArea = 0;
        int left = 0; //首尾指针，谁小谁往里面走
        int right = height.length-1;


        while (left<right){
            maxArea = Math.max(maxArea,(right - left)*Math.min(height[left],height[right]));

            if (height[left]<height[right]){
                left++;
            }else {
                right--;
            }
        }
        return maxArea;
    }

    /*
    * 删除排序数组中的重复项
    *
    * 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
    * 给定数组 nums = [1,1,2], 函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。
    * */

    //经典的双指针问题
    public int removeDuplicates(int[] arr){
        if (arr.length == 0|| arr == null){
            return 0;
        }
        int len = 1;  //使用len记录没有重复元素的位置
        for (int index = 1;index<arr.length;index++){
            if (arr[index]!=arr[index-1]){
                if (arr[index]!=arr[len])
                    arr[len] = arr[index];
                len ++;
            }
        }
        return len;
    }

    /*
    *
        给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
        有效字符串需满足：
        左括号必须用相同类型的右括号闭合。
        左括号必须以正确的顺序闭合。
        注意空字符串可被认为是有效字符串。
    *
    * */

    public boolean isValid(String s){
        Stack<Character> stack = new Stack<>();
        for (int i = 0;i<s.length();i++){
            char ch = s.charAt(i);  //前括号入栈
            if (ch == '(' || ch == '[' || ch == '{'){
                stack.push(ch);
            }else {
                //后括号判断
                if (stack.isEmpty()) {
                    return false;
                }

                char topChar = stack.pop(); //pop 弹出栈顶元素，peek 寻找栈顶元素但不弹出
                if (ch == ')' && topChar != '(') {
                    return false;
                } else if (ch == ']' && topChar != '[') {
                    return false;
                } else if (ch == '}' && topChar != '{') {
                    return false;
                }
            }
        }

        return stack.isEmpty();
    }

    /**
     * 最长公共前缀：
     * 编写一个函数来查找字符串数组中的最长公共前缀。
     * 如果不存在公共前缀，返回空字符串 ""。
     * 示例 1:
     * 输入: ["flower","flow","flight"]
     * 输出: "fl"
     */

    public String longestCommonPrefix(String[] strings){
        int count = strings.length;
        String prefix = "";
        if (count != 0) {
            prefix = strings[0];
        }
        for (int i = 0;i<count;i++){
            //将第一个字符设置为最长前缀，后面的字符不断与之比较，从后往前截取，直到找到两个最长前缀（true）为止
            while (!strings[i].startsWith(prefix)) {
                prefix = prefix.substring(0,prefix.length()-1);
            // substring(index1,index2+1) 获取 index1 到index2的字符串
            }
        }
        return prefix;
    }

    /**
     * 字符串转换为整数
     * 请你来实现一个 atoi 函数，使其能将字符串转换成整数。
     * 首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
     * 当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
     * 该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
     * 注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
     * 在任何情况下，若函数不能进行有效的转换时，请返回 0。
     * 输入: "   -42"
     * 输出: -42
     * 解释: 第一个非空白字符为 '-', 它是一个负号。
     *      我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
     *
     *      重点：判断正负号，防止溢出，计算最终值
     */

     public int Atoi(String str){

         str = str.trim();
         if (str.isEmpty())
             return 0;
         //正负号
         int sign = 1;
         //转换值
         int base = 0;
         //索引位数
         int i = 0;

         //判断正负号
         if (str.charAt(i) == '-' || str.charAt(i) == '+') {
             sign = str.charAt(i++) == '-'? -1 : 1;
         }
         //索引有效数字字符
         while (i < str.length() && str.charAt(i) >= '0' && str.charAt(i) <= '9') {
             // 如果base > MAX_VALUE/10，那么base*10 + new_value > base*10 > MAX_VALUE，这种情况下就会发生溢出。
             // 若base == INT_MAX/10，而且new_value = str.charAt(i++) - '0'`大于`7`，也会发生溢出。因为MAX_VALUE = 2147483647
             if (base > Integer.MIN_VALUE / 10 || (base == Integer.MAX_VALUE / 10 && str.charAt(i) - '0' > '7')) {
                return (sign == 1)?Integer.MAX_VALUE:Integer.MIN_VALUE;
             }

             //计算转换值
             //为什么 -’0‘ ：计算unicode值，charAt计算出的是字符，字符和int类型不通，但是计算出ASCII码值就可以尽心比较或赋值
             base = 10*base+(str.charAt(i++)-'0');
         }

         //计算结果值
         return base = sign*base;

     }

    /**
     * 寻找两个有序数组的中位数
     * 给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
     * 请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
     * 你可以假设 nums1 和 nums2 不会同时为空。
     * 示例 1:
     * nums1 = [1, 3]
     * nums2 = [2]
     * 则中位数是 2.0
     *
     * 本题要求o(log(m+n))的时间复杂度一般来说都是分治法或者二分搜索。
     * 假设两个有序序列共有n个元素（根据中位数的定义我们要分两种情况考虑），当n为奇数时，搜寻第(n/2+1)个元素，当n为偶数时，搜寻第(n/2+1)和第(n/2)个元素，然后取他们的均值。我们可以把这题抽象为“搜索两个有序序列的第k个元素”。如果我们解决了这个k元素问题，那中位数不过是k的取值不同罢了。
     * 那如何搜索两个有序序列中第k个元素呢，这里又有个技巧。假设序列都是从小到大排列，对于第一个序列中前p个元素和第二个序列中前q个元素，我们想要的最终结果是：p+q等于k-1,且一序列第p个元素和二序列第q个元素都小于总序列第k个元素。因为总序列中，必然有k-1个元素小于等于第k个元素。这样第p+1个元素或者第q+1个元素就是我们要找的第k个元素。
     * 所以，我们可以通过二分法将问题规模缩小，假设p=k/2-1，则q=k-p-1，且p+q=k-1。如果第一个序列第p个元素小于第二个序列第q个元素，我们不确定二序列第q个元素是大了还是小了，但一序列的前p个元素肯定都小于目标，所以我们将第一个序列前p个元素全部抛弃，形成一个较短的新序列。然后，用新序列替代原先的第一个序列，再找其中的第k-p个元素（因为我们已经排除了p个元素，k需要更新为k-p），依次递归。同理，如果第一个序列第p个元素大于第二个序列第q个元素，我们则抛弃第二个序列的前q个元素。递归的终止条件有如下几种：
     * 较短序列所有元素都被抛弃，则返回较长序列的第k个元素（在数组中下标是k-1）
     * 一序列第p个元素等于二序列第q个元素，此时总序列第p+q=k-1个元素的后一个元素，也就是总序列的第k个元素
     * 每次递归不仅要更新数组起始位置（起始位置之前的元素被抛弃），也要更新k的大小（扣除被抛弃的元素）
     * 参考原文：https://blog.csdn.net/jek123456/article/details/80022075
     */

    public double findMedianSortedArrays(int[] num1,int[] num2){
        int m = num1.length,n = num2.length;
        int k = (m+n)/2;
        if ((m + n) % 2 == 0) {
            return (findKth(num1,num2,0,0,m,n,k)+findKth(num1,num2,0,0,m,n,k+1));
        }else {
            return (findKth(num1,num2,0,0,m,n,k+1));
        }

    }

    private double findKth(int[] arr1, int[] arr2, int start1, int start2, int len1, int len2, int k) {
        //保证arr1是一个较短的数组
        if (len1 > len1) {
            return findKth(arr2,arr1,start2,start1,len2,len1,k);
        }
        if (len1 == 0) {
            return arr2[start2+k-1];
        }
        if (k == 1) {
            return Math.min(arr1[start1],arr2[start2]);
        }

        int p1 = Math.min(k/2,len1);
        int p2 = k - p1;
        if (arr1[start1 + p1 - 1]<arr2[start2+p2-1]){
            return findKth(arr1,arr2,start1+p1,start2,len1-p1,len2,k-p1);
        }
        else if (arr1[start1 + p1 - 1 ]>arr2[start2 + p2 - 1]){
            return findKth(arr1,arr2,start1,start2+p2,len1,len2-p2,k-p2);
        }
        else {
            return arr1[start1 + p1 - 1];
        }
    }


}
