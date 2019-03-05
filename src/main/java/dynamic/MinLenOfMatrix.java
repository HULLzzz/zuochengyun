package dynamic;

/**
 * @Auther: Think
 * @Date: 2019/1/24 11:40
 * @Description:
 */
public class MinLenOfMatrix {
    //求矩阵中的最小路径 只能向右或者向下走
    //动态规划的算法，使用dp矩阵保存最短的路径，时间复杂度为M*N,空间复杂度为M*N
    public int solution01(int[][] m){
        if (m == null || m.length == 0 || m[0] == null || m[0].length == 0) {
            return 0;
        }
        int row = m.length;
        int col = m[0].length;
        int[][] dp = new int[row][col];
        dp[0][0] = m[0][0];
        for (int i = 1;i<row;i++){
            dp[i][0] = dp[i-1][0]+m[i][0]; //由于只能向右走，dp矩阵的第一行就是左边的元素不断累加
        }
        for (int j = 1;j<col;j++){
            dp[0][j] = dp[0][j-1]+m[0][j]; //由于只能向下走，dp矩阵的第一列就是上边的元素不断累加
        }
        for (int i = 1;i<row;i++){
            for (int j = 1;j<col;j++){
                dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1])+m[i][j];
                //走到(i,j)位置的上一步是上面dp[i][j-1]或者是左边dp[i-1][j],最短路径只需将上一步的最短路径和这一步需走的路径相加即可
            }
        }
        return dp[row-1][col-1];
    }

    //空间压缩的方法是不用记录所有的子问题的解。
    //所以就可以只用一个行数组记录第一行、第二行...一次计算。直到最后一行，得到dp[N-1]就是左上角到右下角的最小路径和。
    //这种二维动态规划的空间压缩几乎可以应用到所有的二维动态规划的题目中，通过一个数组（列数组或者航数组）滚动更新的方式节省了大量的空间。
    // 但是在滚动的过程中动态规划表不断的被行数组或者列数组覆盖更新，最后得到的仅仅是动态规划表的最后一行或者最后一列的最小路径和。所以真正的最小路径是不能通过动态规划表回溯得到的。
    public int solution02(int[][] m){
        if (m == null || m.length == 0 || m[0] == null || m[0].length == 0) {
            return 0;
        }
        int more = Math.max(m.length,m[0].length);
        int less = Math.min(m.length,m[0].length);//行数和列数较小的是less
        boolean rowmore = more == m.length ;//行数是不是大于等于列数
        int[] arr = new int[less]; // 将空间压缩为min(M,N)
        arr[0] = m[0][0];
        for (int i = 1;i<less;i++){
            arr[i] = arr[i-1]+(rowmore?m[0][i]:m[i][0]); //行数大于列数先算一行，否则先算一列
        }
        for(int i = 1;i<more;i++){
            arr[0] = arr[0] + (rowmore?m[0][i]:m[i][0]);
            for (int j = 1;j<less;j++){
                arr[j] = Math.min(arr[j-1],arr[j]) +
                        (rowmore?m[i][j]:m[j][i]);
            }
        }
        return arr[less-1];
    }

}
