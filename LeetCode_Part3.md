# Leetcode Part3 共55道
## 111 Binary Tree Level Order Traversal II
#### _easy_
#### 描述：层序遍历二叉树，要求最后的一层在最前面
#### 思路：深度优先，广度优先。
#### 代码：
#### solution1 DFS :
```
public List<List<Integer>> levelOrderBottom(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        List<List<Integer>> wrapList = new LinkedList<List<Integer>>();
        
        if(root == null) return wrapList;
        
        queue.offer(root);
        while(!queue.isEmpty()){
            int levelNum = queue.size();
            List<Integer> subList = new LinkedList<Integer>();
            for(int i=0; i<levelNum; i++) {
                if(queue.peek().left != null) queue.offer(queue.peek().left);
                if(queue.peek().right != null) queue.offer(queue.peek().right);
                subList.add(queue.poll().val);
            }
            wrapList.add(0, subList);
        }
        return wrapList;
    }
```
#### solution2 BFS ：
```
        public List<List<Integer>> levelOrderBottom(TreeNode root) {
            List<List<Integer>> wrapList = new LinkedList<List<Integer>>();
            levelMaker(wrapList, root, 0);
            return wrapList;
        }
        
        public void levelMaker(List<List<Integer>> list, TreeNode root, int level) {
            if(root == null) return;
            if(level >= list.size()) {
                list.add(0, new LinkedList<Integer>());
            }
            levelMaker(list, root.left, level+1);
            levelMaker(list, root.right, level+1);
            list.get(list.size()-level-1).add(root.val);
        }

```
## 112 Balanced Binary Tree
#### _easy_
#### 描述：查看二叉树是否是二叉平衡数
#### 思路：先求子节点的高度，通过左右子树高度来判断。
#### 代码：
```
public boolean isBalanced(TreeNode root) {
    boolean[] result = new boolean[] {true};
    helper(root, 1, result);
    return result[0];
}
public int helper(TreeNode root, int depth, boolean result[]) {
    if (root == null) {    
        return depth;
    }
    int leftDepth = helper(root.left, depth + 1, result);
    int rightDepth = helper(root.right, depth + 1, result);
    if (result[0] &&  Math.abs(rightDepth - leftDepth) > 1) {
        result[0] =  false;
    }
    return Math.max(leftDepth, rightDepth);
}
```
## 113 Minimum Depth of Binary Tree
#### _easy_
#### 描述：给定一个二叉树，返回离叶子节点的最小距离
#### 思路：用递归，值得注意的是，如果一个节点左或者右子树为空，这种情况的多加考虑。下面代码比较巧妙。
#### 代码：
```
public int minDepth(TreeNode root) {
        if(root == null) return 0;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        return (left == 0 || right == 0) ? left + right + 1: Math.min(left,right) + 1;
       
    }
```
## 114 Path Sum
#### _easy_
#### 描述：给定一个二叉树，和一个整数值。问是否存在从根节点到叶子节点的路径和为该整数值
#### 思路：依然递归。
#### 代码：
```
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root==null)
            return false;
        if(sum==root.val && root.left==null && root.right==null)
            return true;
        return hasPathSum(root.left, sum-root.val) || hasPathSum(root.right, sum-root.val);
    }
```
## 115 Symmetric Tree
#### _easy_
#### 描述：判断二叉树是否对称
#### 思路：详细见代码。
#### 代码：
#### solution1 递归
```
public boolean isSymmetric(TreeNode root) {
    return root==null || isSymmetricHelp(root.left, root.right);
}

private boolean isSymmetricHelp(TreeNode left, TreeNode right){
    if(left==null || right==null)
        return left==right;
    if(left.val!=right.val)
        return false;
    return isSymmetricHelp(left.left, right.right) && isSymmetricHelp(left.right, right.left);
}
```
#### solution2 非递归
```
public boolean isSymmetric(TreeNode root) {
    if(root==null)  return true;
    
    Stack<TreeNode> stack = new Stack<TreeNode>();
    TreeNode left, right;
    if(root.left!=null){
        if(root.right==null) return false;
        stack.push(root.left);
        stack.push(root.right);
    }
    else if(root.right!=null){
        return false;
    }
        
    while(!stack.empty()){
        if(stack.size()%2!=0)   return false;
        right = stack.pop();
        left = stack.pop();
        if(right.val!=left.val) return false;
        
        if(left.left!=null){
            if(right.right==null)   return false;
            stack.push(left.left);
            stack.push(right.right);
        }
        else if(right.right!=null){
            return false;
        }
            
        if(left.right!=null){
            if(right.left==null)   return false;
            stack.push(left.right);
            stack.push(right.left);
        }
        else if(right.left!=null){
            return false;
        }
    }
    
    return true;
}
```
## 116 Maximum Depth of Binary Tree
#### _easy_
#### 描述：返回二叉树的最大深度。
#### 思路：最小深度的变种，最后一个return的逻辑很巧妙。
#### 代码：
```
public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return (left == 0 || right == 0) ?left + right+1:Math.max(left,right)+1;
    }
```
## 117 Convert Sorted Array to Binary Search Tree
#### _easy_
#### 描述：给定一个增序数组，返回一个二叉平衡查找数
#### 思路：递归即可。
#### 代码：
```
public TreeNode sortedArrayToBST(int[] nums) {
        return formBST(nums, 0, nums.length-1);
    }
    
    public TreeNode formBST(int[] nums, int l, int r){
        if(l>r) return null;
        int mid = (l+r+1)/2;
        TreeNode node = new TreeNode(nums[mid]);
        if(l==r) return node;
        node.left = formBST(nums,l,mid-1);
        node.right = formBST(nums,mid+1,r);
        return node;
    }
```
## 118 Same Tree
#### _easy_
#### 描述：判断两个树是否相等
#### 思路：利用递归。感觉树的题都是可以利用递归来做的。
#### 代码：
```
if(p == null || q == null)
            return p == q;
        if(p.val != q.val)
            return false;
        return isSameTree(p.left,q.left) &&isSameTree(p.right,q.right);
```
## 119 Binary Tree Zigzag Level Order Traversal
#### _medium_
#### 描述：给定一个二叉树，返回层次遍历，不过按照Z字形遍历。
#### 思路：递归实现。利用一个%2。实现该功能。
#### 代码：
```
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new LinkedList();
        helper(res,root,0);
        return res;
    }
    public void helper(List<List<Integer>> res,TreeNode root,int level){
        if(root == null)
            return;
        if(res.size() <= level)
            res.add(new LinkedList<Integer>());
        if(level % 2 == 0)
            res.get(level).add(root.val);
        
        else
            res.get(level).add(0,root.val);

        helper(res,root.left,level+1);
            helper(res,root.right,level+1);
    }
```
## 120 Construct Binary Tree from Inorder and Postorder Traversal
#### _medium_
#### 描述：给定一个树的中序和后序遍历，求这个树
#### 思路：利用递归，详细见代码。
#### 代码：
```
public TreeNode buildTree(int[] inorder, int[] postorder) {
        TreeNode nowNode = buildChildTree(postorder,postorder.length -1,inorder,0,postorder.length -1);
        return nowNode;
    }
    public TreeNode buildChildTree(int[] postorder,int postIndex,int[] inorder,int start,int end){
        TreeNode nowNode = null;
        int flag = -1;
        for(int i = start;i <= end;i++){
            if(postorder[postIndex] == inorder[i])
                flag = i;
        }
        if(flag >= 0){
            nowNode = new TreeNode(inorder[flag]);
            nowNode.left = buildChildTree(postorder,postIndex+flag-end-1,inorder,start,flag -1);
            nowNode.right = buildChildTree(postorder,postIndex-1,inorder,flag + 1,end);
        }
        return nowNode;
    }
```
## 121 Path Sum II
#### _medium_
#### 描述：给定一个二叉树和一指定的数sum。求出所以从根节点到叶子节点路径和为sum的路径。
#### 思路：递归调用，回溯法。有两个值得注意的地方，第一必须要到达根节点，才能结束（因为有负数的存在所以）。第二要复制list，采用new ArrayList<>(list)方法。
#### 代码：
```
public List<List<Integer>> pathSum(TreeNode root, int sum) {
	List<List<Integer>> res = new ArrayList<>();
	List<Integer> list = new ArrayList<>();
	helper(res, list, root, sum);
	return res;
}
private void helper(List<List<Integer>> res, List<Integer> list, TreeNode root, int sum) {
	if (root == null) return;
	list.add(root.val);
	if (root.left == null && root.right == null && root.val == sum) {
		res.add(new ArrayList<>(list));
	}
	helper(res, list, root.left, sum - root.val);
	helper(res, list, root.right, sum - root.val);
	list.remove(list.size() - 1);
}
```
## 122 Populating Next Right Pointers in Each Node
#### _medium_
#### 描述：给定一个二叉树，节点中多了一个指针，指向层次遍历中的后一个元素。每层的最后一个指向null
#### 思路：简单的迭代，层序遍历的变种。不过下面代码就不用额外的空间。
#### 代码：
```
    public void connect(TreeLinkNode root) {
        while(root != null){
            TreeLinkNode tempChild = new TreeLinkNode(0);
            TreeLinkNode currentChild = tempChild;
            while(root!=null){
                if(root.left != null) { currentChild.next = root.left; currentChild = currentChild.next;}
                if(root.right != null) { currentChild.next = root.right; currentChild = currentChild.next;}
                root = root.next;
            }
            root = tempChild.next;
        }
     }
```
## 123 Unique Binary Search Trees
#### _medium_
#### 描述：给定数n，问n个节点的二分搜素数有多少种
#### 思路：动态规划，在part1里有更难的，问的是列出这些情况。
#### 代码：
```
    public int numTrees(int n) {
        int [] G = new int[n+1];
        G[0] = G[1] = 1;
    
        for(int i=2; i<=n; ++i) {
    	    for(int j=1; j<=i; ++j) {
    		G[i] += G[j-1] * G[i-j];
    	    }
        }

        return G[n];
    }
```
## 124 
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```

```
## 125 
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```

```
## 126 
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```

```
## 127 
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```

```
## 128 
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```

```
## 129 
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```

```
## 130 
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```

```
