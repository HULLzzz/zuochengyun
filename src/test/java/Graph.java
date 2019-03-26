import com.sun.org.apache.bcel.internal.generic.ANEWARRAY;

import java.util.*;

/**
 * @Auther: Think
 * @Date: 2019/3/13 09:21
 * @Description:
 * 有关图，并查集，拓扑排序等
 */
public class Graph {
    /**
     * 并查集
     * 给定一个未排序的整数数组，找出最长连续序列的长度。
     * 输入: [100, 4, 200, 1, 3, 2]
     * 输出: 4
     * 解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
     * 使用一个集合HashSet存入所有的数字，然后遍历数组中的每个数字，如果其在集合中存在，那么将其移除，然后分别用两个变量pre和next算出其前一个数跟后一个数，然后在集合中循环查找，如果pre在集合中，那么将pre移除集合，然后pre再自减1，直至pre不在集合之中，对next采用同样的方法，那么next-pre-1就是当前数字的最长连续序列，更新res即可。这里再说下，为啥当检测某数字在集合中存在当时候，都要移除数字。这是为了避免大量的重复计算，就拿题目中的例子来说吧，我们在遍历到4的时候，会向下遍历3，2，1，如果都不移除数字的话，遍历到1的时候，还会遍历2，3，4。同样，遍历到3的时候，向上遍历4，向下遍历2，1，等等等。如果数组中有大量的连续数字的话，那么就有大量的重复计算，十分的不高效，所以我们要从HashSet中移除数字
     */
    public int longestConsecutive(int[] nums) {
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for (int num:nums){
            set.add(num);
        }
        for (int num:nums){
            if (set.remove(num)){
                int pre = num - 1,next = num+1;
                while (set.remove(pre)) -- pre;
                while (set.remove(next)) ++next;
                res = Math.max(res,next-pre-1);
            }
        }
        return res;

    }


    /**
     * 给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
     * 找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
     * 示例:
     * X X X X
     * X O O X
     * X X O X
     * X O X X
     * 运行你的函数后，矩阵变为：
     * X X X X
     * X X X X
     * X X X X
     * X O X X
     *      在矩阵中有两种O，一种是跟边界连接的O，一种是不与边界连接的O。假设，我们找到了所有与边界连接的O，那么我们可以将这些O换成另一个字符(这里是 * )，然后重新遍历数组，凡是遇到O的均变成X。这样就能达到我们的目的了。
     *   那么怎么找到所有与边界连接的O呢？一个矩阵无非是4条边，我们可以从4条边上可以查找O，凡是找到了O，将它替换成为*，然后在递归的查找4个方向。这样我们就能找到所有与边界连接的O了。
     *
     */

    public void solve(char[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0) {
            return;
        }
        //遍历第一列和最后一列
        for (int i = 0;i<board.length;i++){
            //第一列找到与边界相连的O，进入递归查找其他的O
            if (board[i][0] == '0') {
                search(board,i,0);
            }
            //最后一列
            if (board[i][board[0].length - 1] == '0') {
                search(board,i,board[0].length-1);
            }
        }
        //遍历第一行和最后一行
        for (int i = 0;i<board[0].length;i++){
            if (board[0][i] == '0') {
                search(board,0,i);
            }
            if (board[board.length - 1][i] == '0') {
                search(board,board.length-1,i);
            }
        }
        //新数组
        for (int i = 0;i<board.length;i++){
            for (int j = 0;j<board[0].length;j++){
                if (board[i][j] == '0') {
                    board[i][j] = 'X';
                }
                if (board[i][j] == '*') {
                    board[i][j] = '0';
                }
            }
        }

    }

    private void search(char[][] board, int row, int col) {
        if (row < 0 || col < 0 || row >= board.length || col >= board[0].length) {
            return;
        }
        //!=0的原因是：*和X是不递归的，只有0才递归
        if (board[row][col] != '0') {
            return;
        }
        //与边界连接的我们都替换为“*”
        //接着查找与边界的0连接的0，同样替换，遇到不是0的退出递归，这样就能找到所有与边界连通的0
        board[row][col] = '*';
        //四个方向
        search(board,row,col-1);
        search(board,row-1,col);
        search(board,row+1,col);
        search(board,row,col+1);
    }

    /**
     * 克隆图
     * 给定无向连通图中一个节点的引用，返回该图的深拷贝（克隆）。图中的每个节点都包含它的值 val（Int） 和其邻居的列表（list[Node]）。
     *
     * 示例：
     * 输入：
     * {"$id":"1","neighbors":[{"$id":"2","neighbors":[{"$ref":"1"},{"$id":"3","neighbors":[{"$ref":"2"},{"$id":"4","neighbors":[{"$ref":"3"},{"$ref":"1"}],"val":4}],"val":3}],"val":2},{"$ref":"4"}],"val":1}
     *
     * 解释：
     * 节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
     * 节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
     * 节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
     * 节点 4 的值是 4，它有两个邻居：节点 1 和 3 。
     *
     * 对图的遍历就是两个经典的方法DFS和BFS。BFS经常用Queue实现，DFS经常用递归实现（可改为栈实现）。
     * 拷贝方法是用用HashMap，key存原始值，value存copy的值，用DFS,BFS方法遍历帮助拷贝neighbors的值。
     */


// Definition for a Node.
class Node {
    public int val;
    public List<Node> neighbors;

    public Node() {}

    public Node(int _val,List<Node> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

//BFS 第一种实现方法是BFS的，就是先将头节点入queue，每一次queue出列一个node，然后检查这个node的所有的neighbors，如果没visited过，就入队，并更新neighbor。

    public Node cloneGraph(Node node) {
        if (node == null) {
            return null;
        }
        //hm判断neighbor有没有被访问过
        HashMap<Node,Node> hm = new HashMap<>();
        //queue记录前一个节点
        LinkedList<Node> queue = new LinkedList<>();

        Node head = new Node(node.val,node.neighbors);
        hm.put(node,head);
        queue.add(node);
        while (queue != null) {
            Node curNode = queue.poll();
            for (Node aneighbors:curNode.neighbors){
                if (!hm.containsKey(aneighbors)) {//没有访问过
                    queue.add(aneighbors);

                    Node newneighbor = new Node(aneighbors.val,aneighbors.neighbors);
                    hm.put(aneighbors,newneighbor);
                }
                //图关系的复制
                hm.get(curNode).neighbors.add(hm.get(aneighbors));
            }
        }
        return head;

    }

    /**
     * 给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。
     *
     * 示例 1:
     *
     * 输入:
     * 11110
     * 11010
     * 11000
     * 00000
     *
     * 输出: 1
     * 示例 2:
     *
     * 输入:
     * 11000
     * 11000
     * 00100
     * 00011
     *
     * 输出: 3
     *
     * 思路：DFS、BFS。只要遍历一遍，碰到一个1，就把它周围所有相连的1都标记为非1，这样整个遍历过程中碰到的1的个数就是所求解。
     */

    public int numIslands(char[][] grid) {
        int m = grid.length;
        if (m == 0) {
            return 0;
        }
        int n = grid[0].length;
        if (n == 0) {
            return 0;
        }
        int ans = 0;
        for (int i = 0;i<m;i++){
            for (int j = 0;j<n;j++){
                if (grid[i][j]!='1')
                    continue;
                ans++;
                dfs(grid,i,j);
            }
        }
        return ans;

    }

    private void dfs(char[][] grid, int row, int col) {
        if (row < 0 || col < 0 || row > grid.length || col > grid[0].length) {
            return;
        }
        if (grid[row][col] != '1') {
            return;
        }
        grid[row][col] = '2';
        dfs(grid,row-1,col);
        dfs(grid,row,col-1);
        dfs(grid,row+1,col);
        dfs(grid,row,col+1);
    }


}
