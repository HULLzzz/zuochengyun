import java.util.*;

/**
 * @Auther: Think
 * @Date: 2019/3/5 19:46
 * @Description: 剑指offer练习
 */


public class offer {
    /**
     * 顺时针打印矩阵
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        if(matrix == null ){
            return res;
        }
        int x  = 0;
        int y  = 0;
        while(x*2<matrix.length&&y*2<matrix[0].length){
            helper(x,y,matrix,res);
            x++;
            y++;
        }
        return res;

    }

    public void helper(int x,int y,int[][] matrix,ArrayList res){

        int rows = matrix.length;
        int cols = matrix[0].length;
        //左到右
        for(int i = x;i<cols - 1 - y;i++){
            res.add(matrix[x][i]);
        }
        //右上到右下
        if (rows - x - 1 > x) {
            for (int i = x+1;i<rows - x - 1;i++){
                res.add(matrix[i][cols - 1 - y]);
            }
        }

        //右下到左下
        if ((rows - x - 1 > x) && (cols - 1 - y > y)) {
            for (int i = cols - 2 - y;i>=y;i--){
                res.add(matrix[rows - x - 1][i]);
            }
        }
        //左下到左上
        if ((rows-1-x>x+1)&&(cols-1-y>y)){
            for (int i = rows - 1 - x -1 ;i>=x+1;i--){
                res.add(matrix[i][y]);
            }
        }

    }

    //19

    public class TreeNode{
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
        TreeNode(int x){
            this.val = x;
        }
    }

    public void mirror(TreeNode node){
        if (node != null) {
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;

            mirror(node.left);
            mirror(node.right);
        }
    }

    //18
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {

        if (root1 == null) {
            return false;
        }
        if (root2 == null)
            return false;
        if (root1 == root2)
            return true;
        boolean res =false;
        //节点的值相同
        if (root1.val == root2.val){
            if (root2.left == null&&root2.right == null)
                return true;
            res = match(root1,root2);
        }
        if (res)
            return true;
        return HasSubtree(root1.right,root2)
                || HasSubtree(root1.left,root2);



    }

    private boolean match(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return false;
        }
        if (root2 == null)
            return true;
        if (root1 == root2)
            return true;
        if (root1.val == root2.val){
            return match(root1.left,root2.left)&&match(root1.right,root2.right);
        }
        return false;
    }

    //17
    public class ListNode{
        ListNode next;
        int val;
        ListNode(int x){
            this.val = x;
        }
    }
    public ListNode Merge(ListNode list1,ListNode list2) {

        if (list1 == null || list2 == null) {
            return (list1 == null)?list2:list1;
        }

        ListNode head = (list1.val>list2.val) ? list2:list1;
        ListNode cur1 = (head==list2) ? list2:list1;
        ListNode cur2 = (cur1 == list1)?list2:list1;

        ListNode pre = null;
        ListNode next = null;
        while (cur1 != null && cur2 != null) {
            if (cur1.val < cur2.val) {
                pre = cur1;
                cur1 = cur1.next;

            }else {
                next = cur2.next;
                pre.next = cur2;

                cur2.next = cur1;
                pre = cur2;
                cur2 = next;
            }

        }

        pre.next = cur1 == null?cur2:cur1;
        return head;

    }

    //
    public ListNode FindKthToTail(ListNode head,int k) {
        ListNode fast = head;
        for (int i = 0;i<k;i++){
            fast = fast.next;
        }
        ListNode slow = head;
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;

    }


    //

    public class Solution {
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
            ArrayList<Integer> res = new ArrayList<Integer>();
            for (int i = 0;i<arr.length;i++){
                res.add(arr[i]);
            }
            return res;
        }

        private  void headSort(int[] array){
            if (array == null || array.length <= 1) {
                return;
            }
            buildMaxHeap(array);
        }

        public void buildMaxHeap(int[] array) {
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
    }



    //
    public int MoreThanHalfNum_Solution(int [] array) {

        if (array == null || array.length < 1) {
            return 0;
        }
        int res =array[0];
        int count = 1;
        for (int i = 1;i<array.length-1;i++){
            if (count == 0){
                res = array[i];
                count = 1;
            }
            else if (res == array[i]){
                count++;
            }else {
                count--;
            }

        }
        count = 0;
        for (int num:array){
            if (res == num) {
                count++;
            }
        }
        if (count > array.length / 2) {
            return res;
        }else {
            return 0;
        }
    }


    //
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> res = new ArrayList<>();
        char[] chars = str.toCharArray();
        if (chars.length < 1 || chars == null) {
            return res;
        }
        helper(chars,0,res);

        return res;
    }

    private void helper(char[] chars, int begin , ArrayList<String> res) {
        if (chars.length - 1 == begin) {
            res.add(new String(chars));
        }
        for (int i = begin;i<=chars.length;i++){
            char tmp;
            tmp = chars[begin];
            chars[begin] = chars[i];
            chars[i] = tmp;

            helper(chars,begin+1,res);
            tmp = chars[begin];
            chars[begin] = chars[i];
            chars[i] = tmp;
        }
    }



    //



    //


    public int solution(int m,int n,int k,int[][] matrix){
        int count = 0;
        int row = matrix.length;
        int num = 0;
        int area = 0;
        int col = 0;
        int index = 1;
        while (index<(row-1)){
            for (int i = 1;i<matrix.length;i ++){
                for (int j = 0;j<matrix[0].length;j++){
                    if ( (matrix[i][j] == 1)&&(matrix[i+1][j] == 1)
                            &&(matrix[i-1][j+1] == 1)&&(matrix[i+1][j+1]==1)&&(matrix[i+1][j-1]==1)){
                        count ++;
                        area +=  matrix[0].length - j;
                    }
                }
            }
            index++;

        }


        return count;
    }


    //
    public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
        if(array == null|| array.length == 1){
            return;
        }
        int xor = 0;
        for (int i : array){
            xor ^= i;
        }
        int index = findfirstBit1(xor);
        for (int i: array){
            if (isBit(i,index)){
                num1[0] ^= i;
            }else {
                num2[0] ^= i;
            }
        }

    }

    private boolean isBit(int num, int index) {
        num>>>=index;
        return (num&1)==1;
    }

    private int findfirstBit1(int num) {
        int index = 0;
        while ((num & 1) == 0 && index < 32) {
            index++;
            num>>>=1;
        }
        return index;
    }


    //
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null){
            return true;
        }
        int left = helper(root.left);
        int right = helper(root.right);

        int dif = left - right;
        if (dif>1||dif<1)
            return false;

        return IsBalanced_Solution(root.left)&&IsBalanced_Solution(root.right);
    }

    private int helper(TreeNode root){
        if (root == null){
            return 0;
        }

        int left = helper(root.left);
        int right = helper(root.right);

        return left>right?(left+1):(right+1);
    }

    //
    public int GetNumberOfK(int [] array , int k) {
        int num = 0;
        if (array != null || array.length > 0) {
            int first = GetfirstOfK(array,k,0,array.length-1);
            int last = getLastK(array,k,0,array.length-1);

            if (first > -1 && last > -1) {
                num = last - first;
            }
        }
        return num;

    }

    private int GetfirstOfK(int [] array , int k,int start,int end) {
        if (array == null || array.length < 1 || start > end) {
            return -1;
        }
        int midIndex = start+(end-start)/2;
        int dataMid = array[midIndex];
        if (dataMid == k) {
            if (midIndex > 0 && array[midIndex - 1] != k || midIndex == 0) {
                return midIndex;
            }else {
                end = midIndex-1;

            }
        } else if (dataMid > k) {
            end = midIndex - 1;
        }else {
            start = midIndex + 1;
        }
        return GetfirstOfK(array,k,start,end);
    }

    private int getLastK(int[] array,int k,int start,int end){
        if (array == null || array.length < 1 || start > end) {
            return -1;
        }
        int midIndex = (start + end)/2;
        int dataMid = array[midIndex];

        if (dataMid == k) {
            if (midIndex+1<array.length&&array[midIndex+1]!=k||midIndex == array.length-1){
                return midIndex;
            }else {
                start = midIndex + 1;
            }
        }else if (dataMid>k){
            start = midIndex+1;
        }else {
            end = midIndex - 1;
        }
        return getLastK(array,k,start,end);
    }

    //
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {

        int len1=0,len2=0;
        ListNode cur1 = pHead1;
        ListNode cur2 = pHead2;
        while (cur1!=null){
            cur1 = cur1.next;
            len1++;
        }
        while (cur2 != null) {
            cur2 = cur2.next;
            len2++;
        }

        int diff = len1 - len2;
        for (int i = 0;i<diff;i++){
            pHead1 = pHead1.next;
        }
        for (int i = 0;i<len1;i++){
            if (pHead1 == pHead2){
                return pHead1;
            } else {
                pHead1 = pHead1.next;
                pHead2 = pHead2.next;
            }
        }
        return null;


    }

    //
    public int InversePairs(int [] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int[] copy = new int[array.length];
        for (int i = 0;i<array.length;i++){
            copy[i] = array[i];
        }
        int count = InversePairsCore(array,copy,0,array.length-1);
        return count;

    }

    private int InversePairsCore(int[] array, int[] copy, int low, int high) {
        if (low == high) {
            return 0;
        }
        int mid = (low+high)>>1;
        int leftCount = InversePairsCore(array,copy,low,mid)%1000000007;
        int rightCount = InversePairsCore(array,copy,mid+1,high)%1000000007;
        int count = 0;
        int i = mid;
        int j = high;
        int locCopy = high;
        while (i>=low&&j>mid){
            if (array[i]>array[j]){
                count += j - mid;
                copy[locCopy--] = array[i--];
                //数值过大求余
                if(count>=1000000007)//数值过大求余
                {
                    count%=1000000007;
                }

            }
            else {
                copy[locCopy--] = array[j--];
            }
        }
        for (;i>low;i--){
            copy[locCopy--] = array[i];
        }
        for (;j>mid;j--){
            copy[locCopy--] = array[j];
        }

        for (int s = low;s<=high;s++){
            array[s] = copy[s];
        }
        return (leftCount+rightCount+count)%1000000007;

    }


    public int FirstNotRepeatingChar(String str) {
        char[] chars = str.toCharArray();
        if (chars.length == 1) {
            return chars[0];
        }
        Map<Character,Integer> map = new HashMap<>();
        for (int i = 0;i<chars.length;i++){
            if (!map.containsKey(chars[i])) {
                map.put(chars[i],1);
            }else {
                map.put(chars[i],2);
            }
        }

        for (int i = 0;i<chars.length;i++){
            if (map.get(chars[i])==1){
                return i;
            }
        }
        return -1;
    }
//

    public int GetUglyNumber_Solution(int index) {
        if (index<=0)
            return 0;
        int[] pUnglyNum = new int[index];
        pUnglyNum[0] = 1;
        int nextUglyIndex = 1;
        int p2 = 0;
        int p3 = 0;
        int p5 = 0;
        while (nextUglyIndex < index) {
            int min = getMin(pUnglyNum[p2]*2,pUnglyNum[3]*3,pUnglyNum[5]*5);
            pUnglyNum[nextUglyIndex] = min;
            while (pUnglyNum[p2]*2<=pUnglyNum[nextUglyIndex]){
                p2++;
            }
            while (pUnglyNum[p3]*3<=pUnglyNum[nextUglyIndex]){
                p3++;
            }
            while (pUnglyNum[p5]*3<=pUnglyNum[nextUglyIndex]){
                p5++;
            }
            nextUglyIndex++;
        }
        return pUnglyNum[nextUglyIndex-1];
    }

    private int getMin(int i, int i1, int i2) {
        int num = (i<i1)?i:i1;
        return (num<i2?num:i2);
    }


    //

        private class Mcomparator implements Comparator<String>{
            @Override
            public int compare(String o1, String o2){
                if (o1== null||o2 == null){
                    return -1;
                }
                String s1 = o1 + o2 ;
                String s2 = o2 + o1;
                return s1.compareTo(s2);
            }

        }




    private void quickSort(String[] numbers, int start, int end, Mcomparator mcomparator) {
        if (start < end) {
            String pivot = numbers[start];
            int left = start;
            int right = end;
            while (start<end){
                while (start < end && mcomparator.compare(numbers[end], pivot) <= 0) {
                    end--;
                }
                numbers[start] = numbers[end];
                while (start<end&&mcomparator.compare(numbers[start],pivot)<=0){
                    start++;
                }
                numbers[end] = numbers[start];
            }
            numbers[start] = pivot;
            quickSort(numbers,left,start-1,mcomparator);
            quickSort(numbers,start+1,end,mcomparator);
        }
    }

    //
    public String PrintMinNumber(int [] numbers) {
        if (numbers.length == 0){
            return " ";
        }
        ArrayList<String> arrayList = new ArrayList<>();
        for (int j = 0;j<numbers.length;j++){
            arrayList.add(String.valueOf(numbers[j]));
        }

        Collections.sort(arrayList, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String string1 = o1 + o2;
                String string2 = o2 + o1;
                return string1.compareTo(string2);
            }
        });
        StringBuilder s = new StringBuilder();
        for (String str : arrayList){
            s.append(str);
        }
        return s.toString();

    }

    //
    public int FindGreatestSumOfSubArray(int[] array) {

        int n = array.length;
        int sum = array[0];
        int max = array[0];
        for(int i = 1;i<n;i++){
            sum = getMax(sum+array[i],array[i]);
            if (sum > max) {
                max = sum;
            }

        } return max;
    }

    private int getMax(int a,int b) {
        return a>b?a:b;
    }

    //
    public int NumberOf1Between1AndN_Solution(int n) {
        if (n <= 0) {
            return 0;
        }
        String value = n+"";
        int[] nums = new int[value.length()];
        for(int i = 0;i<nums.length;i++){
            nums[i] = value.charAt(i)-'0';
        }
        return helper(nums,0);

    }

    private int helper(int[] nums, int index) {
        if (nums == null||index>nums.length||index<0){
            return 0;
        }
        //待处理的第一个数字
        int first = nums[index];
        //待处理的位数
        int len = nums.length - index;
        //如果只有一位且》1
        if (len==1&&first>0){
            return 1;
        }
        if (len == 1 && first == 0) {
            return 0;
        }
        //假设数是21345，先处理最高位
        int numFirstDigit = 0;
        if (first == 1){
            numFirstDigit = atoi(nums,index+1)+1;
        }
        else if (first>1){
            numFirstDigit = power(len-1);
        }
        //处理其他位
        int otherNumDigit = nums[index]*(len-1)*power(len-2);
        //处理0-1234中1的数目
        int recursive = helper(nums,index+1);
        return recursive + otherNumDigit + numFirstDigit;

    }

    private int power(int i) {
        int res = 1;
        for (int j = 0;j<i;j++){
            res *= 10;
        }
        return res;
    }

    private int atoi(int[] nums, int index) {
        //将数组转化位数
        int res = 0;
        for (int i = 0;i<index;i++){
            res = res*10 + nums[i];
        }
        return res;
    }

    //找到当前节点到目标节点的路径，当前节点，目标节点，路径
    public static void getNodePath(TreeNode root,TreeNode target,List<TreeNode> path){
        if (root == null) {
            return;
        }
        //添加当前节点
        path.add(root);
        TreeNode left = root.left;
        TreeNode right = root.right;

        if (right == target){
            path.add(right);
        }else if (left == target){
            path.add(left);
        }else {
            getNodePath(left.right,target,path);
            getNodePath(root.right,target,path);
        }
    }

    //找两个路径的最后一个公共节点
    public static TreeNode getLastCommonNode(List<TreeNode> p1, List<TreeNode> p2){
        Iterator<TreeNode> ite1 = p1.iterator();
        Iterator<TreeNode> ite2 = p2.iterator();
        TreeNode lastNode = null;
        while (ite1.hasNext()&&ite2.hasNext()){
            TreeNode tmp = ite1.next();
            if (tmp == ite2.next()) {
                lastNode = tmp;
            }
        }
        return lastNode;
    }

    //

    public static int movingCount(int threshold, int rows, int cols)
    {
        if (threshold < 0 || rows < 1 || cols < 1) {
            return 0;
        }
        //使用一个数组判断是否能够进入下一个方格中
        boolean[] isVisited = new boolean[rows*cols];
        for (int i = 0;i<isVisited.length;i++){
            isVisited[i] = false;
        }

        return movingCountHelper(threshold,rows,cols,0,0,isVisited);
    }

    private static int movingCountHelper(int threshold, int rows, int cols, int row, int col, boolean[] isVisited) {
        int count = 0;
        //如果（i，j）能够进入，判断周围的四个方格能不能进入
        if (check(threshold,rows,cols,row,col,isVisited)){
            isVisited[row*cols+col] = true;
            count = 1 +
                    movingCountHelper(threshold,rows,cols,row+1,col,isVisited)+
                    movingCountHelper(threshold,rows,cols,row,col+1,isVisited)+
                    movingCountHelper(threshold,rows,cols,row-1,col,isVisited)+
                    movingCountHelper(threshold,rows,cols,row,col-1,isVisited);
    }
    return count;

}

    private static boolean check(int threshold, int rows, int cols, int row, int col, boolean[] isVisited) {
        //判断能不能进入
        return row<rows&&col<cols
                &&row>=0&&col>=0
                &&(Sum(col)+Sum(row))<=threshold
                &&!isVisited[row*cols+col];
    }

    private static int Sum(int num) {
        int res = 0;
        while (num > 0) {
            res += (num%10);
            num /= 10;
        }
        return res;
    }

    //
    public static boolean hasPath(char[] matrix, int rows, int cols, char[] str)
    {
        if (rows<0||cols<0||str.length<0)
            return false;

        //需要一个数组记录状态
        boolean[] isVisited = new boolean[rows*cols];
        for (int i = 0;i<str.length;i++){
            isVisited[i] = false;
        }

        //需要一个数组记录匹配的字母位数
        int[] positive = new int[]{0};

        for (int i = 0;i<rows;i++){
            for (int j = 0;j<cols;j++){
                if (hasPathHelper(matrix,rows,cols,i,j,str,positive,isVisited)){
                    return true;
                }
            }
        }
        return false;

    }

    private static boolean hasPathHelper(char[] matrix, int rows, int cols, int row, int col, char[] str, int[] positive,boolean[] isVisited) {
       boolean hasPath = false;
       //退出条件
        if (positive[0] == str.length){
            return true;
        }

       //按照上下左右进行回溯
        if (row<rows
                &&row>=0
                &&col<cols
                &&col>=0
                &&!isVisited[row*cols+col]
                &&matrix[row*cols+col]==str[positive[0]]){
            isVisited[row*cols+col] = true;
            positive[0]++;
            hasPath = hasPathHelper(matrix,rows,cols,row-1,col,str,positive,isVisited)
                    ||hasPathHelper(matrix,rows,cols,row,col-1,str,positive,isVisited)
                    ||hasPathHelper(matrix,rows,cols,row+1,col,str,positive,isVisited)
                    ||hasPathHelper(matrix,rows,cols,row,col+1,str,positive,isVisited);

            if (!hasPath){
                positive[0]--;
                isVisited[row*cols+col] = false;
            }

        }
        return hasPath;

    }


    public static void main(String[] args) {
        String str = "AAAAAAAAAAAA";

        char[] chars = new char[str.length()];
        char[] in = new char[str.length()];
        for (int i = 0;i<str.length();i++){
            chars[i] = str.charAt(i);
            in[i] = str.charAt(i);
        }

       if (hasPath(chars,3,4,in)){
           System.out.println("true");
       }

    }

    //
    String Serialize(TreeNode root) {

        List<Integer> res = new LinkedList<>();
        List<TreeNode> list = new LinkedList<>();

        list.add(root);
        while (list.size() > 0) {
            TreeNode node = list.remove(0);
            if (node == null) {
                res.add(null);
            }else {
                res.add(node.val);
                list.add(node.left);
                list.add(node.right);
            }
        }
        return res.toString();
        }
    TreeNode Deserialize(String str) {
        if (str.length()<1)
            return null;

        List<Integer> res = new LinkedList<>();
        for (int i = 0;i<str.length();i++){
            //res.set(i) = str.charAt(i);
        }
        TreeNode root = DeserializeHelper(str,0);
        return root;

    }

    private TreeNode DeserializeHelper(String str, int index) {
        if (str.length() < 1 || str.length() < index || str.charAt(index) == 0) {
            return null;
        }
        TreeNode node = new TreeNode(str.charAt(index));
        node.left = DeserializeHelper(str,index*2+1);
        node.right = DeserializeHelper(str,index*2+2);
        return node;

    }

    //
    int index=0;
    TreeNode KthNode(TreeNode pRoot, int k){
        //如果BST不为空，则递归查找
        if(pRoot != null && k >=0){
            TreeNode node=KthNode(pRoot.left,k);
            if(node != null)
                return node;
            if(++index==k)
                return pRoot;
            node=KthNode(pRoot.right,k);
            if(node != null)
                return node;
        }
        //如果BST为空，则返回null
        return null;
    }


    //

    //升序比较器
    private static class IncComparator implements Comparator<Integer>{
        @Override
        public int compare(Integer o1,Integer o2){
            return o1 - o2;
        }
    }

    //降序比较器
    private static class DecComparator implements Comparator<Integer>{
        @Override
        public int compare(Integer o1,Integer o2){
            return o2 - o1;
        }
    }

    private int count = 0;
    private PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    private PriorityQueue<Integer> maxHeap = new PriorityQueue<>(15,
            new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    return o2 - o1;
                }
            });

    public void Insert(Integer num) {
        //偶数的时候下次加入小顶堆
        if (((maxHeap.size()+minHeap.size())&1)==0) {
            if (!maxHeap.isEmpty()&&maxHeap.peek()>num){
                maxHeap.offer(num);
                int filteredMaxNum = maxHeap.poll();
                minHeap.offer(filteredMaxNum);
            }
            minHeap.offer(num);

        } else {
            //奇数时放入大顶堆
            if (!maxHeap.isEmpty()&&minHeap.peek()<num){
                minHeap.offer(num);
                num = minHeap.poll();
            }
            maxHeap.offer(num);
        }

    }

        public Double GetMedian(){
            if (count % 2 == 0) {
                return new Double((minHeap.peek()+maxHeap.peek())/2);
            }else {
                return new Double(minHeap.peek());

        }


    }

}
