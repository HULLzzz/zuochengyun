import dynamic.MinLenOfMatrix;

import java.lang.reflect.AnnotatedArrayType;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @Auther: Think
 * @Date: 2019/3/22 09:46
 * @Description:
 */
public class DynamicProgramming {

    /**
     * 最长回文字串
     */
    String res = "";
    int max = 0;
    public String longestPalindrome(String s) {
        if(s.length() <= 1)
            return s;
        for (int i = 0;i<s.length();i++){
            longestPalindromeHelper(s,i,i+1);
            longestPalindromeHelper(s,i,i);
        }
        return res;
    }
    private void longestPalindromeHelper(String s, int low, int high) {
        //开始循环
        while (low >= 0 && high <= s.length()-1) {
            if (s.charAt(low) == s.charAt(high)) {
                if (high - low + 1>max){
                    max = high - low + 1;
                    res = s.substring(low,high+1);
                }
                high++;
                low--;

            }else
                break;
        }
    }

    /**
     *     动态规划
     * * 关键是状态转移方程
     *dp[i][j] 表示s中的i个字符（在s.charAt的index就是i-1）和p中的j个字符匹配的情况 a*匹配0字符串的时候整个是为空的
     * * p.charAt(j) == s.charAt(i) : dp[i][j] = dp[i-1][j-1]
     *         * If p.charAt(j) == ‘.’ : dp[i][j] = dp[i-1][j-1];
     *         * If p.charAt(j) == ‘*’:
     *             * here are two sub conditions:
     *             *   if p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2] //in this case, a* only counts as empty
     *             *   if p.charAt(i-1) == s.charAt(i) or p.charAt(i-1) == ‘.’:
     *             *       dp[i][j] = dp[i-1][j] // in this case, a* counts as multiple a
     *             *       dp[i][j] = dp[i][j-1] // in this case, a* counts as single a
     *             *       dp[i][j] = dp[i][j-2] // in this case, a* counts as empty，如果是空的情况，实际的字符串匹配和匹配s的第i字符和p的j-2的字符一样
     */
    public boolean isMatch(String s, String p) {

        if (s== null||p == null){
            return false;
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m+1][n+1];
        dp[0][0] = true;
        //初始化第一行
        for (int i = 1;i<n;i++){
            if (p.charAt(i) == '*'&&dp[0][i-1]){
                dp[0][i+1] = true;
            }
        }

        for (int i = 1;i<=m;i++){
            for (int j = 1;j<=n;j++){
                //不是*的情况
                if (p.charAt(j-1)=='.'||p.charAt(j-1) == s.charAt(i-1)){
                    dp[i][j] = dp[i-1][j-1];
                }
                //p[j-1]是*的情况，要判断p[j-2]是否匹配当前的s[i-1]
                //若不匹配，则p[j-1]匹配空的字符串
                //否则三种情况：1.p[j-1]匹配空的字符串 2. p[j-1]匹配单一s[i-1]字符串 3. p[j-1]匹配多个s[i-1]字符串
                if (p.charAt(j-1) == '*'){
                    if (p.charAt(j-2)!=s.charAt(i-1)&&p.charAt(j-2)!='.'){
                        //dp[i][j]是s[i-1]和p[j-1]的匹配情况，
                        dp[i][j] = dp[i][j-2];
                    }else {
                        dp[i][j] = dp[i][j-2]||dp[i][j-1] ||dp[i-1][j];
                    }
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
     * '?' 可以匹配任何单个字符。
     * '*' 可以匹配任意字符串（包括空字符串）。
     */
    public boolean isMatch02(String s, String p) {
        if (s== null || p == null){
            return false;
        }

        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m+1][n+1];
        dp[0][0] = true;
        //初始化
        for (int j = 1;j<=n;j++){
            dp[0][j] = dp[0][j-1]&&p.charAt(j-1) == '*';
        }
        for (int i = 1;i<=m;i++){
            for (int j = 1;j<=n;j++){
                if (p.charAt(j-1) == s.charAt(i-1)||
                        p.charAt(j-1) == '?'){
                    dp[i][j] = dp[i-1][j-1];
                }else if (p.charAt(j-1) == '*'){
                    dp[i][j] = dp[i-1][j]||dp[i][j-1];
                }
            }

        }
        return dp[m][n];
    }
    /**
     * 有效的括号
     * 动态规划法将大问题化为小问题，我们不一定要一下子计算出整个字符串中最长括号对，我们可以先从后向前，一点一点计算。假设d[i]是从下标i开始到字符串结尾最长括号对长度，s[i]是字符串下标为i的括号。如果s[i-1]是左括号，如果i + d[i] + 1是右括号的话，那d[i-1] = d[i] + 1。如果不是则为0。如果s[i-1]是右括号，因为不可能有右括号开头的括号对，所以d[i-1] = 0。
     */
    public int longestValidParentheses(String s) {
        int[] dp = new int[s.length()];
        int max = 0;
        for (int i = s.length()-2;i>=0;i--){
            if (s.charAt(i) == '('){
                int end = i + dp[i+1] + 1;
                if (end<s.length()&&s.charAt(end) == ')'){
                    dp[i] = dp[i+1] + 2;
                    if (end + 1<s.length()){
                        dp[i] += dp[end+1];
                    }
                }
            }
            max = Math.max(max,dp[i]);
        }
        return max;

    }

    /**
     *给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * 示例
     * 输入: [-2,1,-3,4,-1,2,1,-5,4],
     * 输出: 6
     * 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
     * 经典的动态规划，局部最优和全局最优解法。基本思路是这样的，在每一步，我们维护两个变量，一个是全局最优，就是到当前元素为止最优的解是，一个是局部最优，就是必须包含当前元素的最优的解。接下来说说动态规划的递推式（这是动态规划最重要的步骤，递归式出来了，基本上代码框架也就出来了）。假设我们已知第i步的global[i]（全局最优）和local[i]（局部最优），那么第i+1步的表达式是：
     * local[i+1]=Math.max(A[i], local[i]+A[i])，就是局部最优是一定要包含当前元素，所以不然就是上一步的局部最优local[i]+当前元素A[i]（因为local[i]一定包含第i个元素，所以不违反条件），但是如果local[i]是负的，那么加上他就不如不需要的，所以不然就是直接用A[i]；
     * global[i+1]=Math(local[i+1],global[i])，有了当前一步的局部最优，那么全局最优就是当前的局部最优或者还是原来的全局最优（所有情况都会被涵盖进来，因为最优的解如果不包含当前元素，那么前面会被维护在全局最优里面，如果包含当前元素，那么就是这个局部最优）。
     */

    public int maxSubArray(int[] nums) {
        int res = nums[0];
        int sum = 0;
        for (int num:nums){
            if (sum > 0) { //一旦sum<0就可以放弃sum了，这段sum是<0的，累加一定会变小
                sum+=num;
            }
            else {
                sum = num;
            }
            res = Math.max(res,sum);
        }
        return res;

    }

    /**
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
     * 问总共有多少条不同的路径？
     * 想想要到最右下角。到达右下角的方法只有两个，从上面往下，和从右面往左。
     *  利用到达终点的唯一性，就可以写出递推公式（dp[i][j]表示到坐标（i,j）的走法数量）：
     *  dp[i][j] = dp[i-1][j] + dp[i][j-1],初始条件的话，当整个格子只有一行，那么到每个格子走法只有1种；只有一列的情况同理。
     */

    public int uniquePaths(int m, int n) {
        if ( m == 0||n==0)
            return 0;
        if (m==1||n==1) return 1;

        int[][] dp = new int[m][n];
        //只有一行的时候，到终点的每个格子只有一种走法
        for (int i = 0;i<n;i++){
            dp[0][i] = 1;
        }
        //列同理
        for (int i = 0;i<m;i++){
            dp[i][0] = 1;
        }
        for (int i =1;i<m;i++){
            for (int j = 1;j<n;j++){
                dp[i][j] = dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }

    /**
     * 不同路径2，网格中有障碍物，网格中的障碍物和空位置分别用 1 和 0 来表示。
     * 看初始条件的影响：
     * 假设整个棋盘只有一行，那么在第i个位置上设置一个障碍物后，说明位置i到最后一个格子这些路都没法走了
     * 列同理，所以说明，在初始条件时，如果一旦遇到障碍物，障碍物后面所有格子的走法都是0。
     * 再看求解过程，当然按照上一题的分析dp[i][j] = dp[i-1][j] + dp[i][j-1] 的递推式依然成立（机器人只能向下或者向右走嘛）。但是，一旦碰到了障碍物，那么这时的到这里的走法应该设为0，因为机器人只能向下走或者向右走，所以到这个点就无法通过。
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        if (m == 0||n==0)return 0;
        if (obstacleGrid[0][0] == 1||obstacleGrid[m-1][n-1]==1)
            return 0;
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for (int i = 1;i<n;i++){
            if (obstacleGrid[0][i] == 1){
                dp[0][i] = 0;
            }else {
                dp[0][i] = dp[0][i-1];
            }
        }
        for (int j = 1;j<m;j++){
            if (obstacleGrid[j][0] == 1){
                dp[j][0] = 0;
            }else
                dp[j][0] = dp[j-1][0];
        }

        for (int i = 1;i<m;i++){
            for (int j = 1;j<n;j++){
                if (obstacleGrid[i][j] == 1){
                    dp[i][j] = 0;
                }
                else {
                    dp[i][j] = dp[i-1][j]+dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];
    }

    /**
     * 最小路径和：给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     * 说明：每次只能向下或者向右移动一步。
     *
     */
    public int minPathSum(int[][] grid) {
        //dp[i][j]为到(i,j)坐标的最小路径和
        int m = grid.length;
        int n = grid[0].length;
        if (m==0||n==0) return 0;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        //初始化列
        for (int i = 1;i<n;i++){
            dp[0][i] = dp[0][i-1]+grid[0][i];
        }
        for (int i = 1;i<m;i++){
            dp[i][0] = dp[i-1][0]+grid[i][0];
        }
        for (int i = 1;i<m;i++){
            for (int j = 1;j<n;j++){
                dp[i][j] = Math.min(dp[i-1][j]+grid[i][j]
                ,dp[i][j-1]+grid[i][j]);
            }
        }
        return dp[m-1][n-1];
    }
    /**
     * 爬楼梯：假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     * 对于第一层楼梯只有一种方式：一阶爬上来，对于第二层楼梯有两种方式：1阶+1阶，2阶。所以除他们俩外爬到当前楼梯的方式数就是它的前面两个楼梯爬梯方式之和。即动态规划方程为：n[i] = n[i-1]+n[i-2];
     */
    public int climbStairs(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        int[] dp = new int[n];
        //初始化
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2;i<n;i++){
            dp[i] = dp[i-1]+dp[i-2];
        }
        return dp[n-1];
    }

    /**
     * 编辑距离
     * 给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数 。
     * 你可以对一个单词进行如下三种操作：插入一个字符，删除一个字符，替换一个字符
     * 输入: word1 = "horse", word2 = "ros"
     * 输出: 3
     * 解释:
     * horse -> rorse (将 'h' 替换为 'r')
     * rorse -> rose (删除 'r')
     * rose -> ros (删除 'e')
     * 动态数组dp[word1.length+1][word2.length+1]，dp[i][j]表示从word1前i个字符转换到word2前j个字符最少的步骤数。
     * 假设word1现在遍历到字符x，word2遍历到字符y（word1当前遍历到的长度为i，word2为j）。
     *
     * 以下两种可能性：
     *
     * 1. x==y，那么不用做任何编辑操作，所以dp[i][j] = dp[i-1][j-1]
     *
     * 2. x != y
     *
     *    (1) 在word1插入y， 那么dp[i][j] = dp[i][j-1] + 1
     *
     *    (2) 在word1删除x， 那么dp[i][j] = dp[i-1][j] + 1
     *    (3) 把word1中的x用y来替换，那么dp[i][j] = dp[i-1][j-1] + 1
     *  最少的步骤就是取这三个中的最小值。
     * 最后返回 dp[word1.length+1][word2.length+1] 即可。
     */
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(),len2 = word2.length();
        int[][] dp = new int[len1+1][len2+1];
        for (int i = 0;i<=len1;i++){
            dp[i][0] = i;
        }
        for (int j = 0;j<=len2;j++){
            dp[0][j] = j;
        }
        for (int i = 1;i<=len1;i++){
            char c1 = word1.charAt(i-1);
            for (int j = 1;j<=len2;j++){
                char c2 = word2.charAt(j-1);
                if (c2 == c1){
                    dp[i][j] = dp[i-1][j-1];
                }else {
                    int replace = dp[i-1][j-1]+1;
                    int insert = dp[i-1][j]+1;
                    int delete = dp[i][j-1]+1;

                    int min = Math.min(replace,insert);
                    min = Math.min(min,delete);
                    dp[i][j] = min;
                }
            }
        }
        return dp[len1][len2];

    }
    /**
     * 给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
     * 输入:[["1","0","1","0","0"],
     *    ["1","0","1","1","1"],
     *    ["1","1","1","1","1"],
     *    ["1","0","0","1","0"]]输出: 6
     * 这道题的解法灵感来自于Largest Rectangle in Histogram这道题，假设我们把矩阵沿着某一行切下来，然后把切的行作为底面，将自底面往上的矩阵看成一个直方图（histogram）。直方图的中每个项的高度就是从底面行开始往上1的数量。根据Largest Rectangle in Histogram我们就可以求出当前行作为矩阵下边缘的一个最大矩阵。接下来如果对每一行都做一次Largest Rectangle in Histogram，从其中选出最大的矩阵，那么它就是整个矩阵中面积最大的子矩阵。
     * 算法的基本思路已经出来了，剩下的就是一些节省时间空间的问题了。
     * 我们如何计算某一行为底面时直方图的高度呢？ 如果重新计算，那么每次需要的计算数量就是当前行数乘以列数。然而在这里我们会发现一些动态规划的踪迹，如果我们知道上一行直方图的高度，我们只需要看新加进来的行（底面）上对应的列元素是不是0，如果是，则高度是0，否则则是上一行直方图的高度加1。利用历史信息，我们就可以在线行时间内完成对高度的更新。我们知道，Largest Rectangle in Histogram的算法复杂度是O(n)。所以完成对一行为底边的矩阵求解复杂度是O(n+n)=O(n)。接下来对每一行都做一次，那么算法总时间复杂度是O(m*n)。
     * 空间上，我们只需要保存上一行直方图的高度O(n)，加上Largest Rectangle in Histogram中所使用的空间O(n)，所以总空间复杂度还是O(n)
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null||matrix.length == 0||matrix[0].length == 0)
            return 0;
        int maxArea = 0;
        int[] height = new int[matrix[0].length];
        for (int i = 0;i<matrix.length;i++){
            for (int j = 0;j<matrix[0].length;j++){
                height[j] = matrix[i][j] =='0'?0:height[j]+1;
            }
            maxArea = Math.max(largestRectangleArea(height),maxArea);
        }
        return maxArea;


    }

    private int largestRectangleArea(int[] height) {

        if (height == null||height.length == 0)
            return 0;
        int maxArea = 0;
        LinkedList<Integer> stack = new LinkedList<>();
        for (int i = 0;i<height.length;i++){
            while (!stack.isEmpty()&&height[i]<=height[stack.peek()]){
                int index = stack.pop();
                int curArea = stack.isEmpty()?i*height[index]:(i-stack.peek()-1)*height[index];
                maxArea = Math.max(maxArea,curArea);
            }
            stack.push(i);
        }
        while (!stack.isEmpty()){
            int index = stack.pop();
            int curArea = stack.isEmpty()?height.length*height[index]:(height.length-stack.peek()-1)*height[index];
            maxArea = Math.max(maxArea,curArea);
        }
        return maxArea;
        }


    /**
     * 给定一个字符串 s1，我们可以把它递归地分割成两个非空子字符串，从而将其表示为二叉树。
     *
     * 下图是字符串 s1 = "great" 的一种可能的表示形式。
     *
     *     great
     *    /    \
     *   gr    eat
     *  / \    /  \
     * g   r  e   at
     *            / \
     *           a   t
     * 在扰乱这个字符串的过程中，我们可以挑选任何一个非叶节点，然后交换它的两个子节点。
     *
     * 例如，如果我们挑选非叶节点 "gr" ，交换它的两个子节点，将会产生扰乱字符串 "rgeat" 。
     *
     *     rgeat
     *    /    \
     *   rg    eat
     *  / \    /  \
     * r   g  e   at
     *            / \
     *           a   t
     * 我们将 "rgeat” 称作 "great" 的一个扰乱字符串。
     *
     * 同样地，如果我们继续将其节点 "eat" 和 "at" 进行交换，将会产生另一个新的扰乱字符串 "rgtae" 。
     *
     *     rgtae
     *    /    \
     *   rg    tae
     *  / \    /  \
     * r   g  ta  e
     *        / \
     *       t   a
     * 我们将 "rgtae” 称作 "great" 的一个扰乱字符串。
     *
     * 给出两个长度相等的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。
     *
     * 示例 1:
     *
     * 输入: s1 = "great", s2 = "rgeat"
     * 输出: true
     * 这其实是一道三维动态规划的题目，我们提出维护量res[i][j][n]，其中i是s1的起始字符，j是s2的起始字符，而n是当前的字符串长度，res[i][j][len]表示的是以i和j分别为s1和s2起点的长度为len的字符串是不是互为scramble。
     * 有了维护量我们接下来看看递推式，也就是怎么根据历史信息来得到res[i][j][len]。判断这个是不是满足，其实我们首先是把当前 s1[i...i+len-1]字符串劈一刀分成两部分，然后分两种情况：第一种是左边和s2[j...j+len-1]左边部分是不是 scramble，以及右边和s2[j...j+len-1]右边部分是不是scramble；第二种情况是左边和s2[j...j+len-1]右边部 分是不是scramble，以及右边和s2[j...j+len-1]左边部分是不是scramble。如果以上两种情况有一种成立，说明 s1[i...i+len-1]和s2[j...j+len-1]是scramble的。而对于判断这些左右部分是不是scramble我们是有历史信息 的，因为长度小于n的所有情况我们都在前面求解过了（也就是长度是最外层循环）。
     *总 结起来递推式是res[i][j][len] = || (res[i][j][k]&&res[i+k][j+k][len-k] || res[i][j+len-k][k]&&res[i+k][j][len-k]) 对于所有1<=k<len，也就是对于所有len-1种劈法的结果求或运算。因为信息都是计算过的，对于每种劈法只需要常量操作即可完成，因 此求解递推式是需要O(len)（因为len-1种劈法）。
     * 如此总时间复杂度因为是三维动态规划，需要三层循环，加上每一步需要线行时间求解递推式，所以是O(n^4)。虽然已经比较高了，但是至少不是指数量级的，动态规划还是有很大有事的，空间复杂度是O(n^3)
     */
    public boolean isScramble(String s1, String s2) {
        if (s1 == null||s2 == null||s1.length()!=s2.length())
            return false;
        if (s1.length() == 0){
            return true;
        }
        boolean[][][] res = new boolean[s1.length()][s2.length()][s1.length()+1];
        for (int i = 0;i<s1.length();i++){
            for(int j = 0;j<s2.length();j++){
                res[i][j][1] = s1.charAt(i) == s2.charAt(j);
            }
        }
        for (int len = 2;len<=s1.length();len++){
            for (int i = 0;i<s1.length()-len+1;i++){
                for (int j = 0;j<s2.length()-len+1;j++){
                    for (int k = 1;k<len;k++){
                        res[i][j][len] |= res[i][j][k]&&res[i+k][j+k][len-k] || res[i][j+len-k][k]&&res[i+k][j][len-k];
                    }
                }
            }
        }
        return res[0][0][s1.length()];
    }

    /**
     * 一条包含字母 A-Z 的消息通过以下方式进行了编码：
     * 'A' -> 1
     * 'B' -> 2
     * ...
     * 'Z' -> 26
     * 给定一个只包含数字的非空字符串，请计算解码方法的总数。
     * 示例 1:
     * 输入: "12"
     * 输出: 2
     * 解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
     * 本题考查动态规划。用nums数组记录解码种数，nums[i]表示到第 i 个字符，字符串s的解码种数，它由 nums[i - 1]和 nums[i - 2]的解码种数之和，但是会有一些限制，限制条件就是for循环中的2个if语句（第 i - 1个字符的值不能为0，第 i - 2和第 i - 1表示的2位数是大于0，且小于27的）：
     */
    public int numDecodings(String s) {
        if (s== null|| s.length() == 0)
            return 0;
        int[] nums = new int[s.length()+1];
        nums[0] = 1;
        nums[1] = s.charAt(0)!='0'?1:0;
        for (int i = 2;i<=s.length();i++){
            if (s.charAt(i-1)!='0'){
                nums[i] = nums[i-1];
            }
            if (s.charAt(i-2)!='0'&&Integer.parseInt("" + s.charAt(i - 2) + s.charAt(i - 1)) < 27){
                nums[i] += nums[i-2];
            }
        }
        return nums[s.length()];
    }



      public class TreeNode {
          int val;
          TreeNode left;
          TreeNode right;
         TreeNode(int x) { val = x; }
      }

    /**
     * 给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树。
     *
     * 示例:
     *
     * 输入: 3
     * 输出:
     * [
     *   [1,null,3,2],
     *   [3,2,null,1],
     *   [3,1,null,null,2],
     *   [2,1,3],
     *   [1,null,2,null,3]
     * ]
     * 解释:
     * 以上的输出对应以下 5 种不同结构的二叉搜索树：
     *
     *    1         3     3      2      1
     *     \       /     /      / \      \
     *      3     2     1      1   3      2
     *     /     /       \                 \
     *    2     1         2                 3
     * 这是卡特兰数的一种应用，采用动态规划：
     *
     * 1.从start到end，逐个取出一个rootVal作为根节点（n阶原问题）
     *
     * 2.以根rootVal为界划分为左右子树，并指向左右子树（n-1阶子问题）
     */
    public List<TreeNode> generateTrees(int n) {
        if (n <= 0) return new ArrayList<TreeNode>();
        return generateSubTree(1, n);
    }
    public ArrayList<TreeNode> generateSubTree(int start, int end) {
        ArrayList<TreeNode> result = new ArrayList<TreeNode>();
        if (start > end) {
            result.add(null);
            return result;
        }
        for (int rootVal = start; rootVal <= end; rootVal++)
            for (TreeNode leftSubTreeRoot : generateSubTree(start, rootVal - 1))
                for (TreeNode rightSubTreeRoot : generateSubTree(rootVal + 1, end)) {
                    TreeNode root = new TreeNode(rootVal);
                    root.left = leftSubTreeRoot;
                    root.right = rightSubTreeRoot;
                    result.add(root);
                }
        return result;
    }

    /**
     * 给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
     *
     * 示例:
     *
     * 输入: 3
     * 输出: 5
     * 解释:
     * 给定 n = 3, 一共有 5 种不同结构的二叉搜索树:
     *
     *    1         3     3      2      1
     *     \       /     /      / \      \
     *      3     2     1      1   3      2
     *     /     /       \                 \
     *    2     1         2                 3
     *    熟悉卡特兰数的朋友可能已经发现了，这正是卡特兰数的一种定义方式，是一个典型的动态规划的定义方式（根据其实条件和递推式求解结果）。所以思路也很明确了，维护量res[i]表示含有i个结点的二叉查找树的数量。根据上述递推式依次求出1到n的的结果即可。
     * 时间上每次求解i个结点的二叉查找树数量的需要一个i步的循环，总体要求n次，所以总时间复杂度是O(1+2+...+n)=O(n^2)。空间上需要一个数组来维护，并且需要前i个的所有信息，所以是O(n)。
     * 选取一个结点为根，就把结点切成左右子树，以这个结点为根的可行二叉树数量就是左右子树可行二叉树数量的乘积，所以总的数量是将以所有结点为根的可行结果累加起来。
     */
    public int numTrees(int n) {
        if (n<=0)
            return 0;
        int[] res = new int[n+1];
        res[0] = 1;
        res[1] = 1;
        for (int i = 2;i<=n;i++){
            for (int j = 0;j<i;j++){
                res[i] += res[j]*res[i-j-1];
            }
        }
        return res[n];
    }

    /**
     * 给定三个字符串 s1, s2, s3, 验证 s3 是否是由 s1 和 s2 交错组成的。
     *
     * 示例 1:
     *
     * 输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
     * 输出: true
     */


}
