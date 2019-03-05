package dynamic;

/**
 * @Auther: Think
 * @Date: 2019/1/24 11:19
 * @Description:
 */
public class Fibonacci {
    //递归的方法：2^N
    public static long solution01(int n ){
        if (n == 1 || n == 2) {
            return 1;
        }
        else {
            return solution01(n-1)+solution01(n-2);
        }
    }
    //按照顺序计算：N
    public static long solution02(int n){
        if (n == 1 || n == 2) {
            return 1;
        }
        int res = 1;
        int tmp = 1;
        int pre = 0;
        for (int i = 3;i<=n;i++){
            tmp = res;
            res = res + pre;
            pre = tmp;
        }
        return res;
    }

    //log(N)解法：递归式满足F（N） = aF(n-1)+bF(n-2)+...kF(n-i),那么他一定可以用一个i*i的状态矩阵有关的矩阵乘法表示出来
    //使用加速矩阵乘法的动态规划的方法将时间复杂度降为log(N)


}
