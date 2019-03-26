import jdk.management.resource.internal.inst.FileOutputStreamRMHooks;

/**
 * @Auther: Think
 * @Date: 2019/3/10 14:07
 * @Description:
 */
public class leetcode {
    /**
     * 在解决问题的时候总是做出当前来看最好的选择
     * 适用贪心的场景：问题能够分解成子问题来解决，子问题的最优解能递推到最终问题的最优解
     * 贪心和动态规划的不同：它堆每个子问题的解决方案都做出选择，不能回退
     *      动态规划会保存以前的运算结果，并根据以前的结果对当前进行选择，有回退的功能
     */

    //应用：
    /**
     * 通配符的匹配  使用动态规划
     * 使用动态规划的场景：判断题目是否有最优子结构和重叠子问题（fobonacci）
     * 动态规划的步骤：
     *          1. 判断是否能使用动态规划
     *          2. 描述状态
     *          3. 找出状态转移方程和初始状态/状态转移方程就是分析当前状态是由和三层的哪个或者哪些以及处理过的状态转移过来的
     *
     * '?' Matches any single character.
     * '*' Matches any sequence of characters (including the empty sequence).
     * Input:
     * s = "aa"
     * p = "*"
     * Output: true
     * Explanation: '*' matches any sequence.
     * Input:
     * s = "adceb"
     * p = "*a*b"
     * Output: true
     * Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".
     *
     * 本题目中状态描述： boolean f[i][j] ：表示子串s[0~i-1] 与子串p[0~j-1]是否match
     * 假设题中没有*，那就简单了，状态转移方程为f[i][j] = f[i-1][j-1] && s[i]==p[j]
     * 现在加上，我们就可以按情况讨论嘛，如果p[j]!='*' ,那么状态转移方程就是f[i][j] = f[i-1][j-1] && s[i]==p[j]
     * 如果p[j]=='*' , 即可以代替0个、1个或者多个，对于的状态f[i][j]可以由f[i-1][j-1]，f[i][j-1]和f[i-1][j]而来，即f[i][j] = f[i - 1][j - 1] || f[i - 1][j] || f[i][j - 1];
     *
     * 初始状态，显然f[0][0]=true
     */

    public boolean isMatch(String s,String p){
        String tp = "";
        //处理p中多余的*
        for (int i = 0;i<p.length();i++){
            if (p.charAt(i) == '*'){
                tp += '*';
                while (i<p.length()&&p.charAt(i)=='*') i++;
            }else {
                tp += p.charAt(i);
            }
        }
        p = tp;

        //描述状态方程：
        boolean[][] f = new boolean[s.length() + 1][p.length()+1];

        //判断初始状态
        f[0][0] = true;

        //p以*开头的时候
        if (p.length() > 0 && p.charAt(0) == '*') {
            f[0][1] = true;
        }
        //描述状态转移
        for (int i = 0;i<s.length();i++){
            for (int j = 0;j<p.length();j++){
                if (p.charAt(j) == '*') {
                    f[i][j] = f[i-1][j] || f[i-1][j-1] ||f[i][j-1];
                }else {
                    f[i][j] = f[i-1][j-1] && (s.charAt(i)==p.charAt(j))||p.charAt(j) == '?';
                }
            }
        }
        return f[s.length()][p.length()];
    }

    /**
     * 跳跃游戏
     * 根据题目要求，数组里的每个元素表示从该位置可以跳出的最远距离，要求问从第一个元素（index=0）开始，能否达到数组的最后一个元素，这里认为最后一个元素为终点。这里是到达，说明超过也行
     * A = [2,3,1,1,4], return true.
     *
     * A = [3,2,1,0,4], return false.
     * 所以这里可以使用贪心算法，计算出某个点时 能够跳出的最大距离（当前的最大值和（当前点+能跳出的最大距离）的较大的值），如果能跳出的最大距离大于最后一个点的位置，那么返回true，能到达；如果到达当前点后，不能在往后跳，那么不能达到最后点，返回false。
     */
    public boolean canJump(int[] num){
        if (num.length<1)
            return false;
        if (num.length==1)
            return true;
        //max用来存储跳的最远的距离
        int max = 0;
        for (int i = 0;i<num.length;i++){
            max = Math.max(max,num[i]+i);
            if (max<i+1)
                return false;
            else if (max>(num.length-1))
                return true;
        }
        return false;
    }



    /**
     * 买卖股票
     * 最多持有一股，可以买卖无数次
     * Input: [1,2,3,4,5]
     * Output: 4
     * Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
     *              Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
     *              engaging multiple transactions at the same time. You must sell before buying again.
     *
     *              由于可以进行无数次的买进和卖出，只要前一天比后一天的价钱低我们就买进卖出获取利
     */

    public int maxProfit(int[] prices){
        int res = 0;
        for (int i = 0;i<prices.length;i++){
            if (prices[i]>prices[i-1]){
                res += prices[i]+prices[i-1];
            }
        }
        return res;
    }

    /**
     * 在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
     * 你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
     * 如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
     * 说明:
     * 如果题目有解，该答案即为唯一答案。
     * 输入数组均为非空数组，且长度相同。
     * 输入数组中的元素均为非负数。
     * 示例 1:输入:gas  = [1,2,3,4,5]  cost = [3,4,5,1,2]
     * 输出: 3
     * 解释:
     * 从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
     * 开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
     * 开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
     * 开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
     * 开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
     * 开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
     * 因此，3 可为起始索引。
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int sum = 0;
        int total = 0;
        int k = 0;
        for (int i = 0;i<gas.length;i++){
            sum+=gas[i]-cost[i];
            if (sum < 0) {
                sum = 0;
                k = i+1;
            }
            //加油站全部的油都没有cost多，返回-1
            total += gas[i] - cost[i];
        }
        return total<0?-1:k;

    }

    /**
     * 老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
     * 你需要按照以下要求，帮助老师给这些孩子分发糖果：
     * 每个孩子至少分配到 1 个糖果。
     * 相邻的孩子中，评分高的孩子必须获得更多的糖果。
     * 那么这样下来，老师至少需要准备多少颗糖果呢？
     * 示例 1:
     * 输入: [1,0,2]
     * 输出: 5
     * 解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。
     * 初始化所有小孩糖数目为1，从前往后扫描，如果第i个小孩等级比第i-1个高，那么i的糖数目等于i-1的糖数目+1；从后往前扫描，如果第i个的小孩的等级比i+1个小孩高,但是糖的数目却小或者相等，那么i的糖数目等于i+1的糖数目+1。该算法时间复杂度为O(N)。之所以两次扫描，即可达成要求，是因为：第一遍，保证了每一点比他左边candy更多(如果得分更高的话)。第二遍，保证每一点比他右边candy更多(如果得分更高的话)，同时也会保证比他左边的candy更多，因为当前位置的candy只增不减。
     */
    public int candy(int[] rating) {
        int len = rating.length;
        int[] candy = new int[len];
        //经典的双向问题
        //将candy初始化为1
        for (int i = 1;i<len;i++){
            if (rating[i]>rating[i-1]){
                candy[i] = candy[i-1]+1;
            }
        }
        for (int i = len -2;i>=0;i--){
            if ((rating[i]>rating[i+1])&&(candy[i]<=candy[i+1])){
                candy[i] = candy[i]+1;
            }
        }
        int num = 0;
        for (int i = 0;i<len;i++){
            num += candy[i];
        }
        return num;
    }


}
