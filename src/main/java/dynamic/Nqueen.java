package dynamic;

/**
 * @Auther: Think
 * @Date: 2019/1/15 17:11
 * @Description:
 *
 * N皇后的的问题，在n*n棋盘上布置n个皇后，要求任何两个皇后不同行，不同列，也不在同一个对角线上
 *
 * 如果(i,j)位置摆放了皇后，那么接下来
 * 1. 整个i行都不能摆放皇后
 * 2. 整个j列都不能摆放皇后
 * 3. 位置（a，b）满足|a-i| = |b-j|则不能放置
 *
 * 把递归的过程设计成逐行放置皇后的方式，条件2可以使用一个数组保存第i行皇后所在的列数record[i]
 * 递归计算到i行j列的时候，查看record[0...k](k<i)的值满足|k-i| = |record[k]-j|
 */
public class Nqueen {
    public int solution(int n){
        if (n<1)
            return 0;
        int[] record  = new int[n];
        return process(0,record,n);
    }

    private int process(int i, int[] record, int n) {
        if (i == n){
            return 1; //i==n的时候即一条完整的摆放方式已经完成了
        }
        int res = 0;
        for (int j = 0;j<n;j++){
            if (isValid(record,i,j)){
                record[i] = j;
                res += process(i+1,record,n);
            }
        }
        return res;
    }

    private boolean isValid(int[] record, int i, int j) {
        for (int k = 0;k<i;k++){
            if (j == record[k] || Math.abs(record[k] - j) == Math.abs(k - i)) {
                return false;
            }
        }
        return true;
    }


}
