package tencent;

import sun.security.action.GetLongAction;

import java.util.ArrayList;

/**
 * @Auther: Think
 * @Date: 2019/3/6 17:05
 * @Description:        排序和搜索部分
 */
public class Sort {
    public class TreeNode{
        TreeNode left;
        TreeNode right;
        int val;
    }
    /**
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null) {
            return null;
        }
        if (root.val == p.val || root.val == q.val) {
            return root;
        }
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        TreeNode left = lowestCommonAncestor(root,p,q);

        if (right != null && left != null) {
            return root;
        }
        if (right == null){
            return left;
        }else {
            return left;
        }


    }


    //
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        int[] arr = new int[k];
        for (int i = 0;i<k;i++){
            arr[i] = input[i];
        }

        headSort(arr);
        for (int i = k;i<input.length;i++){
            if (arr[0]>input[i]){
                arr[0] = input[i];
                maxHeap(arr,arr.length,0);
            }
        }
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = 0;i<arr.length;i++){
            res.add(arr[i]);
        }
        return res;
    }
    
    public void headSort(int[] array){
        if (array == null || array.length <= 1) {
            return;
        }
        buildMaxHeap(array);
    }

    private void buildMaxHeap(int[] array) {
        //创建大顶堆
        if (array == null || array.length <= 1) {
            return;
        }
        //从最后一个非叶子节点开始向上排
        int half = (array.length-1)/2;
        for (int i = half;i>=0;i--){
            maxHeap(array,array.length,i);
        }
    }

    private void maxHeap(int[] array, int heapSize, int index) {
        int left = index * 2 + 1;
        int right = index*2+2;
        int largest = index;
        if (left<heapSize&&array[left]>array[largest]){
            largest = left;
        }
        if (right<heapSize&&array[right]>array[largest]){
            largest = right;
        }
        if (index!= largest){
            int tmp = array[index];
            array[index] = array[largest];
            array[largest] = tmp;
            maxHeap(array,heapSize,largest);    //下放的点需要在其子树中进行调整
        }

    }


    public static void main(String[] args) {
        int[] input = new int[]{4,5,1,6,2,7,3,8};


    }



}
