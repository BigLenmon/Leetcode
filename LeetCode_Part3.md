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
## 116 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 117 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 118 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 119 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 120 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
