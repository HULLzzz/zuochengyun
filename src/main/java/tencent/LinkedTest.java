package tencent;

/**
 * @Auther: Think
 * @Date: 2019/3/5 10:35
 * @Description:    链表部分
 */
public class LinkedTest {
    public class Node{
        int val ;
        Node next;
        Node(int x){
            this.val = x;
        }
    }


    /**
     * 反转链表 ： 反转一个单链表
     * 示例:
     * 输入: 1->2->3->4->5->NULL
     * 输出: 5->4->3->2->1->NULL
     */
    public Node reverseLink(Node head){
        Node reverseHead = null;    //反转后的单链表的头节点
        Node node = head;   //定义一个node指向head
        Node prev = null;   //定义存储前一个节点
        while (node != null){
            Node next = node.next;  //定义next指向下一个节点
            if (next != null) {
                reverseHead = node;
            }
            node.next = prev;
            prev = node;
            node = next;
        }
        return reverseHead;
    }

    //使用递归实现
    public Node reverseLink02(Node head){
        if (head == null || head.next == null) {
            return head;
        }
        Node next = head.next;
        head.next = null;   //先把所有的键断开
        Node reverseHead = reverseLink02(next);
        next.next = head;   //将所有的元素都指向前一个元素
        return reverseHead;
    }

    /**
     * 利用链表进行两数相加
     * 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
     * 输出：7 -> 0 -> 8
     * 原因：342 + 465 = 807
     */
    public Node addTwoNum(Node l1,Node l2){
        Node node = new Node(0);
        Node p = new Node(0);
        p = node;
        int sum = 0;
        while (l1 != null || l2 != null || sum != 0) {
            if (l1 != null) {
                sum +=l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            p.next = new Node(sum%10);
            sum = sum/10;
            p = p.next;
        }
        return node.next;
    }

    /**
     * 合并两个排序链表
     * 和offer上的16题一样哒
     */

    public Node merge(Node h1, Node h2){
        if (h1 == null||h2==null) {
            return h1 == null ? h2:h1;
        }
        //使用插入的方式，选用头节点比较小的链表作为新链表的头
        //并往头较小的链表中插入另一个链表
        Node head = h1.val < h2.val ? h1:h2;
        Node cur1 = head == h1?h1:h2;
        Node cur2 = cur1 == h1?h2:h1;
        Node pre = null;
        Node next = null;
        while (cur1 != null && cur2 != null) {
            if (cur1.val <= cur2.val) {
                pre = cur1;
                cur1 = cur1.next;
            }else {
                next = cur2.next;
                pre.next = cur2;
                cur2.next = cur1;
                pre = cur2;  //保证pre一直在新链表的前一个
                cur2 = next;
            }
        }
        //任何一个链表为空，返回不为空的链表
        pre.next = cur1 == null?cur2:cur1;
        return head;
    }

    /**
     * 合并k个排序链表
     * 输入:
     * [
     *   1->4->5,
     *   1->3->4,
     *   2->6
     * ]
     * 输出: 1->1->2->3->4->4->5->6
     *
     * lists首尾进行两两合并 o(nlogn)
     */

    public Node mergeKlist(Node[] lists){
        if (lists == null || lists.length == 0) {
            return null;
        }
        if (lists.length == 1) {
            return lists[0];
        }
        int end = lists.length - 1;  //lists中的首尾进行两两合并
        int begin = 0;
        while (end > 0) {   //将结果存放在头，头向后移动，尾向前移动，直到end==begin，此时已经归并了一半，再将begin=0继续归并
            begin = 0;
            while (begin < end) {
                lists[begin] = merge(lists[begin],lists[end]);
                begin++;
                end--;
            }
        }
        return lists[0];
    }

    /**
     * 旋转链表
     * 输入: 1->2->3->4->5->NULL, k = 2
     * 输出: 4->5->1->2->3->NULL
     * 解释:
     * 向右旋转 1 步: 5->1->2->3->4->NULL
     * 向右旋转 2 步: 4->5->1->2->3->NULL
     */

    public Node rotateList(Node head,int k){
        if (head == null||k<=0)
            return head;
        //新建一个结点有利于操作
        Node tmpHead = new Node(0);
        tmpHead.next = head;
        //使用快慢指针计算倒数节点的数目
        Node fast = tmpHead;
        Node slow = tmpHead;
        int len = 0;
        //计算链表的长度
        while (slow.next != null) {
            len++;
            slow = slow.next;
        }
        slow = tmpHead;
        //k的有效长度！！！k可能超过了len
        k = k%len;
        if (k == 0) {
           //不需要反转的情况
            return tmpHead.next;
        }
        //快指针先走k步
        while (--k>0)
            fast = fast.next;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        //重新连接链表
        tmpHead.next = slow.next;
        fast.next = head;
        slow.next = null;
        return tmpHead.next;
    }

    /**
     * 判断链表是否有环
     *
     */

    public boolean hascycle(Node head){
        if (head == null||head.next==null)
            return false;
        Node fast = head;
        Node slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }

    /**
     * 环形链表链表的入环节点
     */

    public Node detectCycle(Node head){
        if (head == null || head.next == null) {
            return null;
        }
        //快慢指针sp fp
        Node sp  = head,fp = head;
        while (fp != null && fp.next != null) {
            sp = sp.next;
            fp = fp.next.next;
            if (fp == sp) {
                break;
            }
        }
        if (fp == null || fp.next == null) {
            return null;
        }
        sp = head;
        //说明有环
        while (fp != sp) {
            sp = sp.next;
            fp = fp.next;
        }
        return sp;
    }


    /**
     * 相交链表的交点
     * Y型链表
     * 两个链表的长度相差为N，则快慢指针，长链表先走N步
     */
    //无环情况下判断是否相交，只需要遍历两个链表，尾节点相同即相交
    public boolean isJionNoLoop(Node h1,Node h2){
        Node p1 = h1,p2 = h2;
        while (p1.next != null) {
            p1 = p1.next;
        }
        while (p2.next != null) {
            p2 = p2.next;
        }
        return p1==p2;
    }

    //判断是否相交
    public Node getFirstJionNode(Node h1,Node h2){
        int len1 = 0;
        int len2 = 0;
        while (h1.next != null) {
            len1 ++;
            h1 = h1.next;
        }
        while (h2.next != null) {
            len2++;
            h2 = h2.next;
        }
        int x = Math.abs(len1-len2);
        return len1>len2?getNode(h1,h2,x):getNode(h2,h1,x);
    }

    public Node getNode(Node n1,Node n2,int x){
        Node cur1 = n1;
        Node cur2 = n2;
        for (int i = 0;i<x;i++){
            cur1 = cur1.next;
        }
        while (cur1 != cur2) {
            cur1 = cur1.next;
            cur2 = cur2.next;
        }
        return cur1;
    }

    /**
     * 删除链表中的节点
     * 输入: head = [4,5,1,9], node = 5
     * 输出: [4,1,9]
     * 解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
     */

    //由于不知道前一个节点，但是我们根据next知道后一个节点，可以将后一个节点直接复制到本节点
    public void deleteNode(Node node){
        node.val = node.next.val;
        node.next = node.next.next;
    }




























}
