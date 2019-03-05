package dynamic;

/**
 * @Auther: Think
 * @Date: 2019/1/24 12:34
 * @Description:
 * 给定数组arr，arr中所有的值都是正数且不重复，每个值代表一种面值的货币，每个货币可以使用任意张，
 * 再给定aim表示要找的钱数，求换钱需要多少种方法？
 */
public class changeMoney {

    /*
    * 暴力递归的方法： 如果arr = [5,10,25,1] aim = 1000,
    * 1. 0张5元货币，[10,25,1]组成剩下的1000元，记为res1；
    * 2. 1张5元货币，[10,25,1]组成剩下的995元，记为res2；
    * .....
    * 那么总的方法数就是res1+res2+.....
    * */
    public int solution01(int[] arr,int aim){
        if (arr == null || arr.length == 0 || aim < 0) {
            return 0;
        }
        return help01(arr,0,aim);


    }

    private int help01(int[] arr, int index, int aim) {
        //表示用arr[index...N-1]这些面值的钱组成的aim
        int res = 0;
        if (index == arr.length) {
            res = aim == 0 ?1:0;
        }else {
            for (int i = 0;arr[index]*i<=aim;i++){
                res += help01(arr,index+1,aim - arr[index]*i);
            }
        }
        return res;
    }


    /*
    * 暴力递归会有大量的重复计算，比如使用0张5元和一张十元的时候，需要求[25,1]组成剩下990元的方法总数，
    * 使用2张5元和0张10元的时候还需要求[25,1]组成剩下990元的方法总数
    *
    * 重复计算之所以大量的发生是因为递归的中间过程没有记录下来，所以可以使用一个map，每进行一次递归过程就将结果记录在map中
    * 即记忆化搜索的方式
    */
    public int solution02(int[ ]arr,int aim){
        if (arr == null || arr.length == 0 || aim < 0) {
            return 0;
        }
        int[][] map = new int[arr.length+1][aim+1];
        return help02(arr,0,aim,map);
    }

    private int help02(int[] arr, int index, int aim, int[][] map) {
        int res = 0;
        if (index == arr.length) {
            res = aim == 0?1:0;
        }
        else {
            int mapValue = 0;
            for (int i = 0;arr[i]*index <= aim;i++){
                if (mapValue != 0){
                    res += mapValue == -1?0:mapValue;
                }else {
                    res += help02(arr,index+1,aim-arr[index]*i,map);
                }
            }
        }
        map[index][aim] = res == 0?-1:res; //每次递归都记录一下求得的res值
        return res;
    }



    /*
    * 使用动态规划的方法：dp[i][j]表示使用钱币arr[0...i]的情况下，组成钱数j有多少种方法
    * 第一列统一设置成1中，也就是组成钱数为0的方式只有1种，即不使用任何货币；
    * 第一行，即使用arr[0]一种货币的情况下，组成j的方法数，所以dp[0][k*arr[0]] = 1;
    * 剩下的就是其他位置（i，j）的时候dp的值
    *   完全不用arr[i]货币，方法数为dp[i-1][j]
    *   使用一张arr[i]货币，方法数为dp[i-1][j-arr[i]]
    *   ....
    *   使用k张arr[i]货币，方法数为dp[i-1][j-k*arr[i]]
    * */

    public int solution03(int[] arr,int aim){
        if (arr.length == 0||aim<0||arr == null){
            return 0;
        }
        int[][] dp = new int[arr.length][aim+1];
        for (int i = 0;i<arr.length;i++){
            dp[i][0] = 1;
        }
        for (int j = 1;arr[0]*j <= aim;j++){
            dp[0][arr[0]*j] = 1;
        }
        int num = 0;
        for (int i = 1;i<arr.length;i++) {
            for (int j = 1;j<=aim;j++) {
                num = 0;
                for (int k = 0;j-arr[0]*k>=0;k++)
                {
                    num += dp[i-1][j-arr[i]*k];
                }
                dp[i][j] = num;
            }
        }
        return dp[arr.length-1][aim];

    }
}
