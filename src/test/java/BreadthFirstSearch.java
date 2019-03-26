import org.omg.PortableServer.LIFESPAN_POLICY_ID;

import java.util.*;

/**
 * @Auther: Think
 * @Date: 2019/3/12 08:50
 * @Description:
 *
 * 一般DFS需要用递归实现，BFS需要用队列实现
 */
public class BreadthFirstSearch {
    //Definition for a binary tree node.
      public class TreeNode {
          int val;
          TreeNode left;
          TreeNode right;
          TreeNode(int x) { val = x; }
      }

    //对称二叉树
        public boolean isSymmetric(TreeNode root) {
          if(root == null)
              return true;
          return isSymmetricHelper(root.right,root.left);

        }

    private boolean isSymmetricHelper(TreeNode right, TreeNode left) {
        if (right == null&&left == null){
            return true;
        }
        if ((right !=null&&left==null)
                ||(right == null&&left!=null)){
            return false;
        }
        if (right.val == left.val){
            return isSymmetricHelper(right.right,left.left)&&isSymmetricHelper(right.left,left.right);
        }
       return false;
    }

    //二叉树的层次遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
          List<List<Integer>> res = new ArrayList<>();
        if (root == null){
            return res;
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()){
            int count = q.size();
            List<Integer> list = new ArrayList<>();
            while (count>0){
                TreeNode cur = q.peek();
                q.poll();
                list.add(cur.val);
                if (cur.left!=null) ((LinkedList<TreeNode>) q).add(cur.left);
                if (cur.right!=null) ((LinkedList<TreeNode>) q).add(cur.right);
                count --;

            }
            res.add(list);

        }
        return res;
    }

    /**
     * 之字形打印二叉树
     * 奇数是从左往右偶数是从右往左
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        if (root == null){return res;}
        Queue<TreeNode> q = new LinkedList<>();
        ((LinkedList<TreeNode>) q).add(root);

        boolean zigzag = false;
        while (!q.isEmpty()){
            List<Integer> list = new ArrayList<>();
            int count = q.size();

            while (count > 0) {
                TreeNode node = q.peek();
                q.poll();
                if (zigzag){
                    list.add(0,node.val);
                }else {
                    list.add(node.val);
                }
                if (node.left != null) {
                    ((LinkedList<TreeNode>) q).add(node.left);
                }
                if (node.right != null) {
                    ((LinkedList<TreeNode>) q).add(node.right);
                }
                count--;

            }
            zigzag = !zigzag;
            res.add(list);
        }
        return res;

    }

    /**
     * 二叉树的层次遍历，要求输出结果是自底向上
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        if (root == null){
            return res;
        }
        Queue<TreeNode> q = new LinkedList<>();
        ((LinkedList<TreeNode>) q).add(root);

        while (!q.isEmpty()){
            List<Integer> list = new LinkedList<>();
            int count = q.size();
            while (count > 0) {
                TreeNode node = q.peek();
                q.poll();
                list.add(node.val);
                if (node.left!=null){
                    ((LinkedList<TreeNode>) q).add(node.left);
                }
                if (node.right!=null){
                    ((LinkedList<TreeNode>) q).add(node.right);
                }
                count--;
            }
            res.add(0,list);
        }
        return res;

    }
















}
