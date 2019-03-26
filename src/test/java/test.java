import com.sun.org.apache.bcel.internal.generic.LNEG;
import sun.java2d.pipe.AAShapePipe;
import sun.security.ssl.SSLContextImpl;

import javax.sound.sampled.ReverbType;
import java.lang.reflect.AnnotatedArrayType;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @Auther: Think
 * @Date: 2019/3/8 16:44
 * @Description:
 */
public class test {


    public static int Jump(int n,int[] height){
        //使用dp表示每层楼的最大高度
        int[] dp = new int[height.length+1];

        //第一层的时间为




        //前三个跳一次
        dp[0] =  height[0];
        dp[1] =0;
        dp[2] = height[2]+height[0];
        //记录跳跃的最短时间
        int min = 0;
        for (int i = 3;i<height.length;i++){
            //上个状态已经跳过一次
                if (dp[i-1] == dp[i-2]||dp[i-1]==dp[i-3]) {
                    dp[i] = Math.min(dp[i - 1] + height[i], dp[i - 2]);
                }else {
                    dp[i] = Math.min(dp[i-1],dp[i-2]);
                }

            }


        return dp[height.length];

    }




    public static double Power(double base, int exponent) {
        if (exponent == 0){
            return 1;
        }
        if (exponent == 1){
            return base;
        }
        double result = Power(base, exponent>>1);
        result *= result;
        if ((exponent & 0x1) == 1){
            result *= base;
        }
        return result;
    }

    public static int NumberOf1Between1AndN_Solution(int n) {
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

    private static  int helper(int[] nums, int index) {
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
            System.out.println("------nfd: "+numFirstDigit);
        }
        else if (first>1){
            numFirstDigit = power(len-1);
            System.out.println("------nfd2: "+numFirstDigit);
        }
        //处理其他位
        int otherNumDigit = nums[index]*(len-1)*power(len-2);
        System.out.println("------other : "+otherNumDigit);
        //处理0-1234中1的数目
        int recursive = helper(nums,index+1);
        System.out.println("------recur: "+recursive);
        return recursive + otherNumDigit + numFirstDigit;

    }

    private static  int power(int i) {
        int res = 1;
        for (int j = 0;j<i;j++){
            res *= 10;
        }
        return res;
    }

    private static int atoi(int[] nums, int index) {
        //将数组转化位数
        int res = 0;
        for (int i = index;i<nums.length;i++){
            res = res*10 + nums[i];
        }
        return res;
    }

    private static  class Mcomparator implements Comparator<String> {
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

    public static String PrintMinNumber(int [] numbers) {

        if (numbers.length<1||numbers == null){
            return null;
        }
        String[] arr = new String[numbers.length];
        for (int i = 0;i<numbers.length;i++){
            arr[i] = String.valueOf(numbers[i]);
        }
        Mcomparator mcomparator = new Mcomparator();
        quickSort(arr,0,numbers.length-1,mcomparator);
        StringBuilder stringBuilder = new StringBuilder();
        for (String s : arr){
            stringBuilder.append(s);
        }
        return stringBuilder.toString();

    }

    private static void quickSort(String[] numbers, int start, int end, Mcomparator mcomparator) {
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
            quickSort(numbers,start+1,right,mcomparator);
        }
    }

    public static int GetUglyNumber_Solution(int index) {
        if (index<=0)
            return 0;
        int[] pUnglyNum = new int[index];
        pUnglyNum[0] = 1;
        int nextUglyIndex = 1;
        int p2 = 0;
        int p3 = 0;
        int p5 = 0;
        while (nextUglyIndex < index) {
            int min = getMin(pUnglyNum[p2]*2,pUnglyNum[p3]*3,pUnglyNum[p5]*5);
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

    private static int getMin(int i, int i1, int i2) {
        int num = (i<i1)?i:i1;
        return (num<i2?num:i2);
    }

    //
    String res = "";
    int max = 0;
    public String longestPalindrome(String s) {

        for (int i = 0;i<s.length();i++){
            if (s.length()%2 == 0){
                longestPalindromeHelper(s,i,i+1);
            }else {
                longestPalindromeHelper(s,i,i);
            }
        }
        return res;

    }

    private void longestPalindromeHelper(String s, int low, int high) {
      //开始循环
        while (low > 0 && high < s.length()) {
            if (s.charAt(low) == s.charAt(high)) {
                if (high - low + 1>max){
                    max = high - low + 1;
                    res = s.substring(low,high+1);
                }
                high++;
                low--;

            }else
                break;
        }
    }

    public static int ReadBook(int chapter, List<List<Integer>> contains,int k){
        int res = 0;
        Map<Integer,Integer> map = new HashMap();
        int i = 0;
        int[] pages = new int[chapter];
        for (List<Integer> list:contains){
            pages[i] = Integer.valueOf(list.get(1)) ;
            i++;
        }
        for (int j = 0;j<pages.length;j++){
            map.put(pages[j],j);
        }
        int index = 0;
        int cur_chapter = 0;
        for (int n = 0;n<pages.length;n++){
            if (k<pages[n]){
                 cur_chapter = map.get(pages[n]);
                 res = chapter - cur_chapter ;
                return res;
            }else if (k == pages[n]){
                cur_chapter = map.get(pages[n]);
                res = chapter - cur_chapter-1;
                return res;
            }
        }
        return res;



    }

    //两个矩阵能否通过转置得到
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0;i<length;i++){
            if (map.containsKey(numbers[i])) {
                map.put(numbers[i],-1);
            }else {
                map.put(numbers[i],0);
            }
        }

        for (int i = 0;i<length;i++){
            if (map.get(numbers[i]) != 0) {
                duplication[i] = numbers[i];
                return true;
            }
        }
        return false;

    }

    //
    public int[] multiply(int[] A) {

        int[] res = new int[A.length];
        if (A.length == 0||A == null){
            return res;
        }
        res[0] = 1;
        for (int i = 1;i<A.length;i++){
            res[i] = res[i+1]*A[i-1];
        }

        int tmp = 1;
        for (int i = A.length-2;i>0;i--){
            tmp*=A[i+1];
            res[i] = res[i] * tmp;
        }
        return res;

    }
    //
    public static boolean match(char[] str, char[] pattern)
    {
        if (pattern.length == 0){
            return str.length == 0;
        }
        if (pattern.length == 1){
            return str.length == 1 && (str[0] == pattern[0] || pattern[0] == '.');
        }

        if (str.length == 0){
            return pattern[0] == '.' ||pattern[1] == '*';
        }

        //如果下一位不是*需要一位一位的递减
        if (pattern[1] != '*') {
            if (str.length == 0){
                return pattern.length == 0;
            }
            else {
                return  str[0] == pattern[0] || pattern[0] == '.'
                        && match(Arrays.copyOfRange(str,1,str.length),
                        Arrays.copyOfRange(pattern,1,pattern.length));

                }
            }else {
            if (str.length == 0){
                return pattern.length == 0;
            }
            else {
                int i = 0;
                while (str.length > 0&&(str[0] == pattern[0] || pattern[0] == '.')) {
                    if (match(str,Arrays.copyOfRange(pattern,2,pattern.length))){
                        return true;
                    }

                     str = Arrays.copyOfRange(str,1,str.length);
                }
                return match(str,Arrays.copyOfRange(pattern,2,pattern.length));
            }
        }

    }


public class TreeLinkNode {
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null;

    TreeLinkNode(int val) {
        this.val = val;
    }
}

    public class Solution {
        public TreeLinkNode GetNext(TreeLinkNode pNode)
        {
            if (pNode == null){
                return null;
            }
            //目标节点和当前节点
            TreeLinkNode target = null,cur = null;

            if (pNode.right!=null){
                target = pNode.right;
                while (target.left!= null){
                    target = target.left;
                }
            }else{
                //当前节点是右子节点
                cur = pNode;
                target = pNode.next;
                if (pNode.next != null&&target.left == pNode){
                    target = pNode.next;
                }else if (pNode.next!=null&&target.right==pNode){
                    while (target.next!=null&&target.left!=cur){
                        cur = target;
                        target = pNode.next;
                    }
                }
        }
        return target;
        }

        }


        //
        public ArrayList<Integer> maxInWindows(int [] num, int size)
        {

            ArrayList<Integer> res = new ArrayList<>();
            PriorityQueue<Integer> heap = new PriorityQueue<>();

            for (int i = 0;i< size;i++){
                heap.add(num[i]);
            }
            res.add(heap.peek());

            for (int i = 0,j=size+i;j<num.length;i++,j++){
                heap.remove(num[i]);
                heap.add(num[j]);
                res.add(heap.peek());
            }
            return res;
        }

        //
     public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {

        int pre = 0;
        int end = array.length-1;
        while (pre<end){
            int cur = array[pre]+array[end];
            if (cur == sum){
                return new ArrayList<>(Arrays.asList(array[pre],array[end]));
            }
            if (array[pre]+array[end]<sum){
                pre++;
            }
            if (array[pre]+array[end]>sum)
            {
                end --;
            }
        }
        return new ArrayList<>();

    }

    //
    public static ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        int small = 1;
        int big = 2;
        int middle = (sum+1)/2;
        int curSum = small + big;
        while (small<middle){
            if (curSum == sum){
                ArrayList<Integer> list = new ArrayList<>(2);
                for (int i = small;i<=big;i++){
                    list.add(i);
                }
                res.add(list);
            }
            while (curSum>sum&&small<middle)
            {
                curSum -= small;
                small++;
                if (curSum == sum){
                    ArrayList<Integer> list = new ArrayList<>(2);
                    for (int i = small;i<=big;i++){
                        list.add(i);
                    }
                    res.add(list);
                }
            }
            big++;
            curSum += big;
        }
        return res;
    }

    //

    public  String ReverseSentence(String str) {
        char[] chars = new char[str.length()];
        for (int i = 0;i<str.length();i++){
            chars[i] = str.charAt(i);
        }
        ReverseSentence01(chars);
        return String.valueOf(chars);
    }
    public  void ReverseSentence01(char[] str) {
        //先反转每个单词，再反转整体

        if (str.length == 1)
            return ;
        int pre = 0;
        int end = 1;
        while (pre < str.length){

            while (end<str.length&&str[end]!=' '){
                end++;
            }
            ReverseSentenceCore(str,pre,end-1);
            pre = end+1;
            end = pre + 1;
        }

        //反转整体
        ReverseSentenceCore(str,0,str.length-1);

    }

    private  void ReverseSentenceCore(char[] chars,int pre,int end) {

        while (pre<end){
            char tmp = chars[pre];
            chars[pre] = chars[end];
            chars[end] = tmp;
            pre ++ ;
            end -- ;
        }
    }

    //
    public String LeftRotateString(String str,int n) {
        if (str == null||str.length() == 1)
            return str;
        int num = n%str.length();
        char[] chars = new char[str.length()];
        for (int i = 0;i<str.length();i++){
            chars[i] = str.charAt(i);
        }

        ReverseSentence02(chars,n);
        return String.valueOf(chars);
    }
    public  void ReverseSentence02(char[] str,int n){
        ReverseSentenceCore(str,0,n-1);
        ReverseSentenceCore(str,n,str.length-1);
        ReverseSentenceCore(str,0,str.length-1);

    }

    //
    public static void soluiton(int num){
        if (num<1)
            return;
        //设置两个数组分别存放状态
        int[][] p = new int[2][6*num+1];
        //初始化状态数组
        for (int i = 0;i<num*6+1;i++){
            p[0][i] = 0;
            p[1][i] = 0;
        }
        //设置标记位置，区别当前状态和之前的状态
        int flag = 0;

        //设置状态转移方程的初始值
        //只投掷一次筛子
        for (int i = 1;i<=6;i++){
            p[0][i] = 1; //每个都出现了一次
        }


        //循环进行投掷
        for (int i = 2;i<=num;i++){
            //投掷两次筛子，出现的值最少应该是k，k前面的值都应该出现的次数为0
            for (int k = 1;k<i;k++){
                p[1-flag][k] = 0;
            }

            //数组的第几位表示筛子和出现的次数
            for (int k = 1;k<num*6+1;k++){
                p[1-flag][k] = 0;
            }
            for (int m = i;m<=num*6;m++){
                for (int j = 6;m-j>=0&&j>=0;j--){
                    p[1-flag][m] += p[flag][m-j];
                }
            }

            flag = 1-flag;
            }
        double total = Math.pow(6,num);
        for (int m = num;m<6*num;m++){
            double ratio = p[flag][m]/total;

        }
    }

    //
    public boolean isContinuous(int [] numbers) {
        if (numbers == null || numbers.length<5){
            return false;
        }

        Arrays.sort(numbers);
        int gap = 0;
        int zero = 0;
        //记录差几张牌
        for (int i = 0;i<numbers.length-1;i++){

            if (numbers[i] == 0) {
                zero++;
            }else {
                if (numbers[i+1]!=(numbers[i]+1)){
                    gap = numbers[i-1]-numbers[i];
                }
            }
        }
        return gap == zero ? true:false;

    }

    //
    public int LastRemaining_Solution(int n, int m) {
        if (n<1||m<1){
            return -1;
        }
        List<Integer> list = new LinkedList<>();
        for (int i = 0;i<n;i++){
            list.add(i);
        }
        //记录删去的位置
        int index = 0;
        int start = 0;

        while (list.size() > 1) {
            for (int i = 0;i<m;i++){
                index = (index+1)%list.size();
            }
            list.remove(index);

        }
        return list.get(0);

    }

    public int Sum_Solution(int n) {

        int sum = n;
        if (n == 1) {
            return sum;
        }
        sum += Sum_Solution(n-1);
        return sum;

    }

    //
    public int Add(int num1,int num2) {
        //不加进位
        int num = num1^num2;

        //进位
        int i = (num1&num2)<<1;

        while (i!=0){
            int tmp = num;
            num  = num^i;
            i = (num&i)<<1;


        }
        return num;

    }

    //
    public int StrToInt(String str) {
        if (str == null) {
            return -1;
        }
        str = str.trim();
        if (str.length()==0){
            return 1;
        }
        int index = 0;


        long res ;
        if (str.charAt(0) == '-'){
            return  transfer(str,1,false);
        }else if (str.charAt(0) == '+'){
            return  transfer(str,1,true);
        } else if (str.charAt(0)<='9'&&'0'<=str.charAt(0)){
                return   transfer(str,0,true);
        }else {
            return -1;
        }

    }

    private int transfer(String num, int index,boolean pos) {
        if (index >= num.length()) {
            return -1;
        }
        int res;
        long tmp = 0;
        for (int i = 0;i<num.length();i++){
            if (isDigit0(num.charAt(i))) {
                tmp = tmp*10+num.charAt(index) - '0';
                if (tmp > Integer.MAX_VALUE) {
                    return -1;
                }
            }else {
                return 0;
            }
        }

        if (pos) {
            if (tmp > Integer.MAX_VALUE) {
                return -1;
            }else {
                res = (int)tmp;
            }
        }
        else {
            if (tmp == Integer.MAX_VALUE) {
                res = Integer.MIN_VALUE;
            }else {
                res = (int) - tmp;
            }
        }
        return res;


    }

    private boolean isDigit0(char str) {
        return (str>='0')&&str<='9';
    }


 public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}

    public ListNode EntryNodeOfLoop(ListNode pHead)
    {
        ListNode fast =pHead.next.next,slow = pHead.next;

        while (fast.val!=slow.val){
            fast = fast.next.next;
            slow = slow.next;
        }

        fast = pHead;

        while (fast.val!=slow.val){
            fast = fast.next;
            slow = slow.next;
        }

        return fast;

    }

    //
    public boolean Find(int target, int [][] array) {
        if (array == null||array.length<1||array[0].length<1)
            return false;
        if (array.length == 0|| array[0].length == 0)
            return false;
        int rows = 0;
        int cols = array[0].length;

        int row = 0;
        int col = cols-1;
        while (row>=0&&col<cols&&row<rows&&col>=0){
            if (array[row][col]==target){
                return true;
            }else if (array[row][col]<target){
                row++;
            }else {
                col--;
            }
        }
        return false;

    }
//

    public String replaceSpace(StringBuffer str) {
        //计算新数组的长度
        int blank = 0;
        for (int i = 0;i<str.length();i++){
            if (str.charAt(i) == ' ') {
                blank++;
            }else
                continue;
        }
        char[] chars = new char[str.length()+blank*3];
        int p = chars.length-1;
        for (int i = str.length()-1;i>=0;i--){
            if (str.charAt(i) == ' '){
                chars[p] = '0';
                chars[p-1] = '2';
                chars[p-2] = '%';
                p = p-3;
            }else {
                chars[p] = str.charAt(i);
                p--;
            }
        }
        String res = "";
        for (char i:chars){
            res += i;
        }
        return res;

    }

    //
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> list = new ArrayList<>();
        Stack<ListNode> stack = new Stack<>();
        while (listNode!=null){
            stack.add(listNode);
            listNode = listNode.next;
        }
        while (!stack.isEmpty()) {
            list.add(stack.pop().val);
        }
        return list;

    }

      public class TreeNode {
         int val;
          TreeNode left;
          TreeNode right;
          TreeNode(int x) { val = x; }
      }

    //
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {

        if (pre == null || in == null) {
            return null;
        }
        return ConstructBinaryTree(pre,0,pre.length-1,in,0,in.length-1);

    }

    private TreeNode ConstructBinaryTree(int[] pre, int pres, int pree, int[] in, int ins, int ine) {
        if (pres>pree)
            return null;
        int val = pre[pres];
        //找到中序遍历根节点的index

        int index = 0;

        while (in[index]!=val&&index<=ine){
            index++;
        }
        if(index>ine){
            return null;
        }
        TreeNode node = new TreeNode(val);
        node.val = val;

        node.left = ConstructBinaryTree(pre,pres+1,pres+index-ins,in,ins,index-1);
        node.right = ConstructBinaryTree(pre,pres+index-ins+1,pree ,in,index+1,ine);
        return node;
    }

    //
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        if (!stack2.isEmpty()){
            while (!stack2.isEmpty()){
                stack1.add(stack2.pop());
                stack1.add(node);
            }
        }
            stack1.add(node);
    }

    public int pop() {
        while (!stack1.isEmpty()){
            stack2.add(stack1.pop());
        }
        int res;
        if (!stack2.isEmpty()){
            res = stack2.pop();
            return res;
        }
        return -1;
    }

    //

    //优化二分
    public int minNumberInRotateArray(int [] array) {
       if (array == null)
           return -1;
       int lo = 0;
       int hi = array.length - 1;
       int mid = (lo+hi)/2;
       //只要还是乱序的，就要进行二分
       while (array[lo]>=array[hi]){
           if (hi - lo == 1) {
               return array[hi];
           }
           if (array[lo]==array[mid]&&array[mid]==array[hi]){
               int res = array[0];
               for (int i = 1;i<array.length;i++){
                   if (res<=array[i]){
                       res =  array[i];
                   }
                   return res;
               }

           }
           if (array[mid]>=array[lo]){
               lo = mid;
           }
           else if (array[mid]<=array[hi]){
               hi = mid;
           }
       }
       return array[mid];


    }

    //
    public int Fibonacci(int n) {
        int[] res = new int[n+1];
        res[0] = 0;
        res[1] = 1;
        res[2] = 1;
        if (n == 0)
            return 0;
        if (n == 1){
            return 1;
        }

        if (n>=2){
            for (int i = 2;i<=n;i++){
                res[i] = res[i-1]+res[i-2];
            }
        }

        return res[n];


    }

    public int JumpFloor(int target) {

        //dp[i] 跳上i个台阶的方法
        if (target == 0)
            return 0;
        if (target == 1)
            return 1;
        if (target == 2)
            return 2;
        int[] dp = new int[target+1];
        if (target>=3){
            dp[0] = 1;
            dp[1] = 2;
            for (int i = 2;i<target;i++){
                dp[i] = dp[i-1]+dp[i-2];
            }
        }
        return dp[target-1];



    }

    //
    public int JumpFloorII(int target) {
        if (target == 0)
            return 0;
        if (target == 1)
            return 1;
        if (target == 2)
            return 2;

        int[] dp = new int[target+1];
        if (target >=3){
            dp[0] = 1;
            dp[1] = 2;
            for (int i = 2;i<target;i++){
                dp[i] = 2*dp[i-1];
            }
        }
        return dp[target-1];

    }

    //
    public int NumberOf1(int n) {
        int res = 0;
        if (n == 0)
            return 0;
        while (n!=0){
            res++;
            n = (n-1)&n;
        }
        return res;

    }

    //
    public ListNode ReverseList(ListNode head) {
//        Stack<ListNode> s = new Stack<>();
//
//        while (head.next!=null){
//            s.add(head);
//            head = head.next;
//        }
//
//        ListNode pre = head;
//        while (!s.isEmpty()){
//            pre.next = s.pop();
//            pre = pre.next;
//        }
//        return head;

        //头节点
        ListNode reversHead = new ListNode(1);
        reversHead.next = null;

        ListNode pre;
        while (head.next!=null){
            pre = head.next;
            head.next = reversHead.next;
            reversHead.next = head;

            head = pre;
        }
        head.next = reversHead.next;
        return head;


    }

    //
    public void reOrderArray(int [] array) {


    }




    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        int n = sc.nextInt();
        int[] a = new int[100];
        int i = 0;

        while (sc.hasNext()){
            a[i] = sc.nextInt();
            i++;
        }

        //计算原数组的和
        int sum = 0;
        for (int j= 0;j<a.length;j++){
            sum += a[j];
        }

        Arrays.sort(a);
        int one = a[0];
        int cur;

        //减小的最大值
        for (int m = 1;m<=n;m++){
            for (int j = 1;j<=a[i];j++){
                if (a[i]%j!=0){
                    continue;
                }
                cur = sum - one - a[i];
                sum=Math.min(sum,cur+one*j+a[i]/j);

            }
        }

        System.out.println(sum);

    }








}
