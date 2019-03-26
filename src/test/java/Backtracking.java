/**
 * @Auther: Think
 * @Date: 2019/3/14 16:22
 * @Description:
 * 回溯和动态规划
 */
public class Backtracking {
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

    /**
     * 正则表达式匹配
     * 给定一个字符串 (s) 和一个字符模式 (p)。实现支持 '.' 和 '*' 的正则表达式匹配。
     *
     * '.' 匹配任意单个字符。
     * '*' 匹配零个或多个前面的元素。
     *
     * 正则表达式如果期望着一个字符一个字符的匹配,是非常不现实的.而"匹配"这个问题,非 常容易转换成"匹配了一部分",整个匹配不匹配,要看"剩下的匹配"情况.这就很好的把 一个大的问题转换成了规模较小的问题:递归
     */
    public boolean isMatch(String s, String p) {
        //递归出口
        if (p.length() == 0)
            return s.length()==0;

        if (p.length() == 1) {
            return (s.length()==1)&&(p.charAt(0)==s.charAt(0)
                    || p.charAt(0)=='.');
        }

        if (p.charAt(1) != '*') {
            //next char不是*则需要一个一个递减
            if (s.length()==0)
                return false;
            else
                return (s.charAt(0)==p.charAt(0)||p.charAt(0)=='.')
                &&isMatch(s.substring(1),p.substring(1));
            //substring(1) begin为1，end为字符串结束
        }else {
            //next char is "*"
            while (s.length()>0&&(p.charAt(0)==s.charAt(0)
            ||p.charAt(0)=='.')){
                if (isMatch(s,p.substring(2))){
                    return true;
                }
                //如果*前面的能够匹配的话，则需要将s依次往后移动，直到出现第一个yup不匹配的字符串，p才能跳过*
                s = s.substring(1);
            }
            return isMatch(s,p.substring(2));
        }


    }


}
