/**
 * @Auther: Think
 * @Date: 2019/3/11 08:58
 * @Description:
 */
public class DepthFirstSearch {

    public class TreeNode{
        TreeNode left;
        TreeNode right;
        int val;
        TreeNode(int x){
            this.val = x;
        }
    }
    /**
     * 深度优先搜索
     *
     */

    /**
     * 验证二叉搜索树
     * 二叉搜索树：左子节点必须小于根节点小于右子节点
     * 左右子树都是二叉树
     *     5
     *    / \
     *   1   4
     *      / \
     *     3   6
     * Output: false
     * Explanation: The input is: [5,1,4,null,null,3,6]. The root node's value
     *              is 5 but its right child's value is 4.
     */
    public boolean isValidBST(TreeNode root) {
        return helper(root,Long.MAX_VALUE,Long.MIN_VALUE);

    }

    private boolean helper(TreeNode root, long maxValue, long minValue) {
        if (root == null) {
            return true;
        }
        if (root.val>=maxValue||root.val<=minValue)
            return false;
        return helper(root.right,maxValue,root.val)&&helper(root.left,root.val,minValue);
    }

    /**
     * Recover Binary Search Tree
     * 二叉排序树有两个节点被打乱了，请将他们交换过来
     * Input: [3,1,4,null,null,2]
     *
     *   3
     *  / \
     * 1   4
     *    /
     *   2
     *
     * Output: [2,1,4,null,null,3]
     *
     *   2
     *  / \
     * 1   4
     *    /
     *   3
     *
     */
    TreeNode mistake1;
    TreeNode mistake2;
    TreeNode pre;
    public void recoverTree(TreeNode root) {
        recoverHelper(root);
        if (mistake1!=null&&mistake2!=null) {
            int tmp = mistake1.val;
            mistake1.val = mistake2.val;
            mistake2.val = tmp;

        }
    }

    private void recoverHelper(TreeNode root) {
        if (root == null) {
            return;
        }
        if (root.left != null) {
            recoverHelper(root.left);
        }
        if (pre != null && pre.val > root.val) {
            if (mistake1 == null) {
                mistake1 = pre;
                mistake2 = root;
            }else {
                mistake2 = root;
            }
        }
        pre = root;
        if (root.right != null) {
            recoverHelper(root.right);
        }
    }

    /**
     * Input:     1         1
     *           / \       / \
     *          2   1     1   2
     *
     *         [1,2,1],   [1,1,2]
     *
     * Output: false
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if ((p == null && q != null) || (p != null && q == null)) {
            return false;
        }
        return p.val == q.val && isSameTree(p.left,q.left)&&isSameTree(p.right,q.right);

    }

    /**
     * 对称二叉树
     * For example, this binary tree [1,2,2,3,4,4,3] is symmetric:
     *
     *     1
     *    / \
     *   2   2
     *  / \ / \
     * 3  4 4  3
     *
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null){
            return true;
        }
        return isSymmetricHelper(root.left,root.right);

    }

    private boolean isSymmetricHelper(TreeNode left, TreeNode right) {
        if (left == null && right == null){
            return true;
        }
        if ((left == null && right != null) ||(left!=null&&right==null)) {
            return false;
        }
        return left.val == right.val&&isSymmetricHelper(left.left,right.right)&&isSymmetricHelper(left.right,right.left);
    }

    /**
     * 二叉树的最大深度
     *
     */
    public int maxDepth(TreeNode root) {

        if (root == null){
            return 0;
        }else {
            //分别求出左右子树的深度
            int left = maxDepth(root.left)+1;
            int right = maxDepth(root.right)+1;
            return Math.max(right,left);
        }
    }


}
