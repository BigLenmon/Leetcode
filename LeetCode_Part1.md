# Leetcode Part1 共55道
## 1 plusOne
### _easy_
#### 描述：用一组数据表示一个整数，实现整数加一的操作
#### 主要思路：主要考虑最高位进位的情况，可以创建一个长度加一的数组，原数组进行加一操作的时候，同时也将每位的结果复制到新数组中，最后判断最高位是否有进位，如果有返回新数组，没有则返回旧数组。
#### 代码
```
public int[] plusOne(int[] digits) {
        int flag = 1;
        int[] res = new int[digits.length+1];
        for(int i = digits.length-1;i >= 0;i--){
            if(flag == 1){
                if(digits[i] == 9)
                    digits[i] = 0;
                else{
                    digits[i]++;
                    flag = 0;
                }
            }
            res[i] = digits[i];
        }
        if(flag == 1){
            res[0] = 1;
            return res;
        }
        else
            return digits;
    }
```
## 2 Pascal's Triangle II
### _easy_
#### 描述：输出指定行的杨辉三角。从0开始计数
#### 思路：第一位保持不变，其他位a[i] = a[i]+a[i-1]来更新数组（记得从后往前更新），最后加上一位1。
#### 代码：
```
public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<Integer>(rowIndex+1);
        res.add(1);
        if(rowIndex == 0) return res;
        for(int i = 1;i <= rowIndex;i++){
            for(int j = i-1;j >= 1;j--){
                res.set(j,res.get(j)+res.get(j-1));
            }
            res.add(1);
        }
        return res;
    }
```
## 3 Two Sum II - Input array is sorted
#### _easy_
#### 描述：给定一个数taget，且数组中有两个数相加等于target，求这两个数的索引
#### 思路：从数组两端遍历，left 和right。如果这两个数相加小于target，则left往右移一位。若大于则right往左移一位。其中最值得思考的是为什么不存在这种情况：当left和right相加小于target时，0到left之间的一个数ll和right到数组末尾之间的一个数rr，ll+rr等于target.
#### 代码：
```
public int[] twoSum(int[] numbers, int target) {
        int[] res = new int[2];
        int left = 0,right = numbers.length-1;
        while(left < right){
            if(numbers[left]+numbers[right] < target)
                left++;
            else if(numbers[left]+numbers[right] > target)
                right--;
            else
                break;
        }
        res[0] = left+1;
        res[1] = right+1;
        return res;
}
```
## 4 Rotate Array
#### _easy_
#### 描述：For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4]
#### 思路：一个是利用反转，先反转全部数组，再反转（0，k-1）和（k,length-1）的数组。另外一种思路是创建一个新的数组，将旧数组的数依次复制到新数组里。j=(i+k)%length;还有一个不用创建新数组，利用j=(i+k)%length一直复制，利用一个变量保存替换前的值。直到复制完一轮为止（但是这种方法我竟然没写出来。麻蛋）。
#### 代码：
```
public void rotate(int[] nums, int k) {
    k %= nums.length;
    reverse(nums, 0, nums.length - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, nums.length - 1);
}

public void reverse(int[] nums, int start, int end) {
    while (start < end) {
        int temp = nums[start];
        nums[start] = nums[end];
        nums[end] = temp;
        start++;
        end--;
    }
}
//方法二
public void rotate(int[] nums, int k) {
        k = k % nums.length; 
        int[] oldNums = nums.clone();
        for(int i = 0; i< nums.length;i++){
            nums[(i+k)%nums.length] = oldNums[i]; 
        }
    }
```
##  5 Container With Most Water
#### _ medium_
#### 描述：有一个正整数数组，数组的每一位数字i代表一个玻璃版的高度，求数组中任意两个玻璃板代表的容器的容量的最大值。
#### 思路：假设在数组中S10和S20代表的容量最大。我们从数组两端遍历数组。当左半边到达S10时，右半边到S21时。我们要做的是右半边网左移。根据题意分析，我们可以得到S10是大于S21.因为s10小于S21时，S10和S20代表的容量不可能是最大的容量。所以我们只需要判断左右两边的数值大小，小的移动即可。这个和求数组的任意两个数的值最大题类似。虽然判断的条件不同，但是主要思想类似。以后碰到类似的题（什么任意两个）可以从这个方向思考一下。
#### 代码：
```
public int maxArea(int[] height) {
        int left = 0,right= height.length-1,res= 0;
        while(left < right){
            res = Math.max(Math.min(height[left],height[right])*(right-left),res);
            if(height[left] > height[right])
                right--;
            else
                left++;
        }
        return res;
    }
```
## 6 3Sum Closes
#### _ medium_
#### 描述：给定一个数组和一个目标值。从数组中选取3个数，使得这三个数的和与目标值相近。求最相近的和。
#### 思路：将数组排序，对于每一位数i，从i+1到数组末尾遍历。找到与target最相近的和，这部分和两个数的最大值算法类似，也是从数组两端开始遍历。本来以为可以用背包来实现，结果发现不对。
#### 代码：
```
public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);，
        int res = nums[0]+nums[1]+nums[2];
        for(int i = 0;i < nums.length-2;i++){
            int j = i+1;
            int k = nums.length -1;
            while(j < k){
                int sum = nums[i]+nums[j]+nums[k];
                if(Math.abs(target -sum) == 0)
                    return target;
                if(Math.abs(target-sum) < Math.abs(target-res)){
                    res = sum;
                }
                if(sum > target)
                    k--;
                else
                    j++;
            }
        }
        return res;
    }
```
## 7 Jump Game
#### _ medium_
#### 描述：给定一个数组，数组的值代表可以向前几步，判断给定的数组能否到达最后一位、
#### 思路：就是简单的用一个变量代表最远距离，最后判断是否大于数组长度。其实我还有一种想法，是不是除了最后一位可以为零以外，其他位为零就代表是不能到达了。我觉得可以.（麻蛋，被打脸了，【2,0，0】就是可以到达的）
#### 代码：
```
bool canJump(int A[], int n) {
    int i = 0;
    for (int reach = 0; i < n && i <= reach; ++i)
        reach = max(i + A[i], reach);
    return i == n;
}
```
## 8 Spiral Matrix II
#### _medium_
#### 描述：给一个数n,输出长度为n的spiralMatirx矩阵。
#### 思路：一行一列的顺序赋值很重要，下面的代码可以解决一些边界问题。注意边界变量减小的顺序，中间两个是减小的。
#### 代码：
```
public int[][] generateMatrix(int n) {
        // Declaration
        int[][] matrix = new int[n][n];
        
        // Edge Case
        if (n == 0) {
            return matrix;
        }
        
        // Normal Case
        int rowStart = 0;
        int rowEnd = n-1;
        int colStart = 0;
        int colEnd = n-1;
        int num = 1; //change
        
        while (rowStart <= rowEnd && colStart <= colEnd) {
            for (int i = colStart; i <= colEnd; i ++) {
                matrix[rowStart][i] = num ++; //change
            }
            rowStart ++;
            
            for (int i = rowStart; i <= rowEnd; i ++) {
                matrix[i][colEnd] = num ++; //change
            }
            colEnd --;
            
            for (int i = colEnd; i >= colStart; i --) {
                if (rowStart <= rowEnd)
                    matrix[rowEnd][i] = num ++; //change
            }
            rowEnd --;
            
            for (int i = rowEnd; i >= rowStart; i --) {
                if (colStart <= colEnd)
                    matrix[i][colStart] = num ++; //change
            }
            colStart ++;
        }
        
        return matrix;
    }
```
## 9 Unique Paths II
#### _medium_
#### 描述：给定一个数组，数组内有零一值。零代表可以通过，1表示不能通过。问从矩阵左上角到右下角有多少种路径。只能往下和往右。
#### 思路：dp[i][j] = dp[i-1][j]+dp[i][j-1]。下面代码是优化的空间复杂度是O(n);
#### 代码：
```
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int width = obstacleGrid[0].length;
    int[] dp = new int[width];
    dp[0] = 1;
    for (int[] row : obstacleGrid) {
        for (int j = 0; j < width; j++) {
            if (row[j] == 1)
                dp[j] = 0;
            else if (j > 0)
                dp[j] += dp[j - 1];
        }
    }
    return dp[width - 1];
}
```
## 10 Minimum Path Sum
#### _medium_
#### 描述：给定一个数组，一个点从左上角走到右下角,每经过一个点就加上这个点上的值，并且该点只能往下和往右走，问到右下角最小的值是多少？
#### 思路：dp[i][j] =Math.min( dp[i-1][j],dp[i][j-1])+value。下面代码是优化的空间复杂度是O(n);
#### 代码：
```
public int minPathSum(int[][] grid) {
        if(grid.length == 0) return 0;
        int[] dp = new int[grid[0].length];
        for(int i = 0;i < grid.length;i++){
            for(int j = 0;j < grid[0].length;j++){
                if(i == 0 && j == 0)
                    dp[j] = grid[i][j];
                else if(j == 0)
                    dp[j] += grid[i][j];
                else if(i == 0)
                    dp[j] = dp[j-1]+grid[i][j];
                else
                    dp[j] = Math.min(dp[j],dp[j-1])+grid[i][j];
            }
        }
        return dp[grid[0].length -1];
    }
```
## 11 Set Matrix Zeroes
#### _medium_
#### 描述：给定一个数组，如果数组中存在为零的值，将这个位置的行和列都设置为0.
#### 思路：三种方法：第一种最简单，创建新的数组，将旧数组中的值复制到新数组中，如果值为零就设置该行该列为0.不过空间复杂度为o(m*n).第二种方法存储值为零的列。等将行改为零后，将列也改成0.复杂度o(M).最后一种，如果有一个值r[i][j]为零，就将r[i][0]和r[0][j]改为0.等将所有值标记完成后，将行列头为零的行列都改为零，不过该为零得从后往前改。空间复杂度o(1);
#### 代码：
```
public void setZeroes(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        int flag = 1;
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == 0) flag = 0;
            for (int j = 1; j < col; j++)
                if (matrix[i][j] == 0)
                    matrix[i][0] = matrix[0][j] = 0;
        }

        for (int i = row - 1; i >= 0; i--) {
            for (int j = col - 1; j >= 1; j--)
                if (matrix[i][0] == 0 || matrix[0][j] == 0)
                    matrix[i][j] = 0;
            if (flag == 0) matrix[i][0] = 0;
        }    
    }
```
## 12 Search a 2D Matrix
#### _medium_
#### 描述：给定一个二维数组，数组内每行都是增序，并且每行的第一个数都大于上一行的最后一个数。给一个值target，问target在二维数组内吗？
#### 思路：两种思路，第一种就是当做两个二分查找，先查找数所在的行，在找所在的列。另一种方法就是将二维数组直接当做一个增序的一维数组，用二分查找。下面代码是用的第二种方法，简洁明了。
#### 代码：
```
bool searchMatrix(vector<vector<int> > &matrix, int target) {
        int n = matrix.size();
        int m = matrix[0].size();
        int l = 0, r = m * n - 1;
        while (l != r){
            int mid = (l + r - 1) >> 1;
            if (matrix[mid / m][mid % m] < target)
                l = mid + 1;
            else 
                r = mid;
        }
        return matrix[r / m][r % m] == target;
    }
```
## 13 Subsets
#### _medium_
#### 描述：给定一个数组(不重复），求它的子集
#### 思路：利用溯源法即可到结果
#### 代码：
```
public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> tempList = new ArrayList<>();
        backTrack(res,tempList,nums,0);
        return res;
    }
    public void backTrack(List<List<Integer>> res,List<Integer> tempList,int[] nums,int start){
        
            res.add(new ArrayList<>(tempList));
            for(int i = start;i < nums.length;i++){
                tempList.add(nums[i]);
                backTrack(res,tempList,nums,i+1);
                tempList.remove(tempList.size() - 1);
            }
        
    }
```
## 14 Word Search
#### _medium_
#### 描述：给定一个字符的二维数组，和一字符串s.判断字符串是否在二维数组内。匹配字符串的下一位必须得和当前位临近（在上下左右）。
#### 思路：依旧按照图的广度优先遍历算法。先匹配当前位置的数组字符和字符串的头字符。如果匹配成功，就匹配该字符的上下左右位是否和字符串的下一位。为了让下一次遍历的时候不再返回之前匹配的，我们用与上256来让当前字符改变不能匹配。详细见代码。注意代码的书写，很简洁。可以参考一下。
#### 代码：
```
public boolean exist(char[][] board, String word) {
    char[] w = word.toCharArray();
    for (int y=0; y<board.length; y++) {
    	for (int x=0; x<board[y].length; x++) {
    		if (exist(board, y, x, w, 0)) return true;
    	}
    }
    return false;
}

private boolean exist(char[][] board, int y, int x, char[] word, int i) {
	if (i == word.length) return true;
	if (y<0 || x<0 || y == board.length || x == board[y].length) return false;
	if (board[y][x] != word[i]) return false;
	board[y][x] ^= 256;
	boolean exist = exist(board, y, x+1, word, i+1)
		|| exist(board, y, x-1, word, i+1)
		|| exist(board, y+1, x, word, i+1)
		|| exist(board, y-1, x, word, i+1);
	board[y][x] ^= 256;
	return exist;
}
```
## 15 Remove Duplicates from Sorted Array II
#### _medium_
#### 描述：给定一个增序数组，修改数组，让每个数字最多出现两次。并返回修改后数组的长度。
#### 思路：设置一个变量存储数字出现的次数，如果一个数字和前一个数字相同则加一，否则改成初始值。另一个变量存储新数组的长度，如果新的数组符合规范就移动到新数组的后面。因为之前的数字都已经遍历过了，所以可以当做新数组的存储值，而不用自己新创建一个数组。
#### 代码：
```
public int removeDuplicates(int[] nums) {
        if(nums.length == 0)
            return 0;
        int flag = 1;
        int res = 1;
        for(int i = 1;i < nums.length;i++){
            if(nums[i] == nums[i-1]){
                if(flag < 2){
                    res ++;
                    flag++;
                    nums[res - 1] = nums[i];
                }
            }
            else{
                flag = 1;
                res ++;
                nums[res - 1] = nums[i];
            }
        }
        return res;
    }
```
## 16 Search in Rotated Sorted Array II
#### _medium_
#### 描述：给定一个数组，该数组是递增有序的，但是循环右移了几位。问给定一个数target，返回数组是否包括target. 
#### 思路：利用二分法查找。可以更加中间数和数组两端来判断那边是有序的，那边是移动过的。详见下面代码。我还有一种解法：首先遍历整个数组，找到循环右移的数字k,然后利用二分查找，并且不用判断那边是有序的那边不是有序的。直接通过偏移来查找。（但是复杂度。。前面那个是O(logn）,最差O（n）。我这O(n）起步啊)
#### 代码：
```
public boolean search(int[] nums, int target) {
        int start = 0, end = nums.length - 1, mid = -1;
        while(start <= end) {
            mid = (start + end) / 2;
            if (nums[mid] == target) {
                return true;
            }
            //If we know for sure right side is sorted or left side is unsorted
            if (nums[mid] < nums[end] || nums[mid] < nums[start]) {
                if (target > nums[mid] && target <= nums[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            //If we know for sure left side is sorted or right side is unsorted
            } else if (nums[mid] > nums[start] || nums[mid] > nums[end]) {
                if (target < nums[mid] && target >= nums[start]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            //If we get here, that means nums[start] == nums[mid] == nums[end], then shifting out
            //any of the two sides won't change the result but can help remove duplicate from
            //consideration, here we just use end-- but left++ works too
            } else {
                end--;
            }
        }
        
        return false;
    }
```
## 17 Construct Binary Tree from Preorder and Inorder Traversal
#### _medium_
#### 描述：给出一个数的前序和中序遍历，构建出该数
#### 思路：利用递归思想，划分成子问题。不过最后的判断要注意，最好是要向下面代码那样，如果flag找不到了，就说明到末尾了。我之前写的是preIndex 等于preorder数组的长度时，就是末尾。但是这样会导致重复的，因为左子树永远不会到达数组末尾。
#### 代码：
```
public TreeNode buildTree(int[] preorder, int[] inorder) {
        TreeNode nowNode = buildChildTree(preorder,0,inorder,0,preorder.length -1);
        return nowNode;
    }
    public TreeNode buildChildTree(int[] preorder,int preIndex,int[] inorder,int start,int end){
        TreeNode nowNode = null;
        int flag = -1;
        for(int i = start;i <= end;i++){
            if(preorder[preIndex] == inorder[i])
                flag = i;
        }
        if(flag >= 0){
            nowNode = new TreeNode(inorder[flag]);
            nowNode.left = buildChildTree(preorder,preIndex + 1,inorder,start,flag -1);
            nowNode.right = buildChildTree(preorder,preIndex+flag-start+1,inorder,flag + 1,end);
        }
        return nowNode;
    }
```
## 18 Triangle
#### _medium_
#### 描述：给定一个三角形，从三角形的头往下走，问到达底部边时，最小值是多少，往下只能往正下方和右下方。
#### 思路：dp[i][j] = Math.min(dp[i-1][j-1],dp[i-1][j)。在这题中如果不用二维数组来存储的话，那就要两个一维数组。
#### 代码：
```
public int minimumTotal(List<List<Integer>> triangle) {
        int res[] = new int[triangle.size()];
        int temp[] = new int[triangle.size()];
        for(int i= 0;i < triangle.size();i++){
            for(int j = 0;j <= i;j++){
                if(j == 0){
                    res[j] = temp[j]+triangle.get(i).get(j);
                }else if(j == i){
                    res[j] = temp[j-1]+triangle.get(i).get(j);
                }else{
                    res[j] = Math.min(temp[j-1],temp[j]) + triangle.get(i).get(j);
                }
            }
            for(int j = 0;j <= i;j++)
                temp[j] = res[j];
            
        }
        int min = res[0];
        for(int i = 1;i < res.length;i++){
            min = Math.min(min,res[i]);
        }
        return min;
    }
```
## 19 Maximum Product Subarray
#### _medium_
#### 描述：给定一个数组（里面有正有负），求连续子串的最大乘积；
#### 思路：因为是乘积的最大值，所以一般情况下，乘上最新的一位后。当前值不是最大就是最小。所以利用两个变量存储当前以来的最大值和最小值。另外为了防止乘上零。每次都得对最大值和最新一位的值进行比较，更新最大值。详细见代码（写的nb）.
#### 代码：
```
public int maxProduct(int[] nums) {
        int r = nums[0];
        int max,min;
        max = min = r;
        for(int i= 1;i < nums.length;i++){
            int item= nums[i];
            if(item < 0){
                int temp = min;
                min = max;
                max = temp;
            }
            max = Math.max(item,item * max);
            min = Math.min(item,item * min);
            r = Math.max(r,max);
        }
        return r;
    }
```
## 20 Find Minimum in Rotated Sorted Array
#### _medium_
#### 描述：给定一个数组，这个数组是一个递增排序数组，但是循环右移了几位。求这个数组的最小值
#### 思路：利用二分查找，首先通过数组两端判断是否有序，如果有序就返回第一位。否则利用mid = （lo+hi）/2和lo,hi判断数组最小值在哪半边（最小值在不是有序的半边）。下面代码nb。我的想法比没有这个简洁。
#### 代码：
```
 int findMin(vector<int> &num) {
        int start=0,end=num.size()-1;
        
        while (start<end) {
            if (num[start]<num[end])
                return num[start];
            
            int mid = (start+end)/2;
            
            if (num[mid]>=num[start]) {
                start = mid+1;
            } else {
                end = mid;
            }
        }
        
        return num[start];
    }
```
## 21 Minimum Size Subarray Sum
#### _medium_
#### 描述：给定一个整数组，和一个目标值。问当连续子串的和大于等于目标值时，其连续子串的最小值为多少。如果找不到返回0
#### 思路：设置两个变量，分别表示子串的头和尾。如果子串相加小于目标值，尾变量加一。否则头加一。这样就可以最小长度了。
#### 代码：
```
public int minSubArrayLen(int s, int[] a) {
        if (a == null || a.length == 0)
            return 0;
        int i = 0, j = 0, sum = 0, min = a.length+1;
  
        while (j < a.length) {
            sum += a[j++];
    
            while (sum >= s) {
                min = Math.min(min, j - i);
                sum -= a[i++];
            }
        }
  
        return min == a.legnth+1 ? 0 : min;
    }
```
## 22 Next Permutation
#### _medium_
#### 描述：给定一个数组，找到下一个permutation的数组。如果没有就返回最小的。
#### 思路：从后往前遍历，找到后一位大于前一位的。设前一位为val。表示再往前的几位就可以不用动。然后再从后往前遍历，找到第一个大于val的值（k）。交换k与val。再将k之后的数组反序，就可以得到了后一位了。
#### 代码：
```
public void nextPermutation(int[] num) {
    int n=num.length;
    if(n<2)
        return;
    int index=n-1;        
    while(index>0){
        if(num[index-1]<num[index])
            break;
        index--;
    }
    if(index==0){
        reverseSort(num,0,n-1);
        return;
    }
    else{
        int val=num[index-1];
        int j=n-1;
        while(j>=index){
            if(num[j]>val)
                break;
            j--;
        }
        swap(num,j,index-1);
        reverseSort(num,index,n-1);
        return;
    }
}

public void swap(int[] num, int i, int j){
    int temp=0;
    temp=num[i];
    num[i]=num[j];
    num[j]=temp;
}

public void reverseSort(int[] num, int start, int end){   
    if(start>end)
        return;
    for(int i=start;i<=(end+start)/2;i++)
        swap(num,i,start+end-i);
}
```
## 23 Subsets II
#### _medium_
#### 描述：给定一个数组（可重复），求它的子集
#### 思路：还是用溯源法。不过遇到相同的数字，就只加入第一个，之后的跳过去。
#### 代码：
```
public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        Arrays.sort(nums);
        traceBack(temp,res,nums,0);
        return res;
    }
    public  void traceBack(List<Integer> temp,List<List<Integer>> res,int[] nums,int start){
        res.add(new ArrayList<>(temp));
        int i = start;
        while (i < nums.length){
            temp.add(nums[i]);
            traceBack(temp,res,nums,i+1);
            temp.remove(temp.size()-1);
            i++;
            while (i < nums.length  && nums[i] == nums[i-1]) i++;
        }
    }
```
## 24 Search for a Range
#### _medium_
#### 描述：给出一个递增有序数组（包含重复元素），和一个目标值。返回目标值是否在数组内，如果在给出开始位置和结束位置。
#### 思路：二分查找两次，一次查找左边界，一次查找右边界。
#### 代码：
```
vector<int> searchRange(int A[], int n, int target) {
    int i = 0, j = n - 1;
    vector<int> ret(2, -1);
    // Search for the left one
    while (i < j)
    {
        int mid = (i + j) /2;
        if (A[mid] < target) i = mid + 1;
        else j = mid;
    }
    if (A[i]!=target) return ret;
    else ret[0] = i;
    
    // Search for the right one
    j = n-1;  // We don't have to set i to 0 the second time.
    while (i < j)
    {
        int mid = (i + j) /2 + 1;	// Make mid biased to the right
        if (A[mid] > target) j = mid - 1;  
        else i = mid;				// So that this won't make the search range stuck.
    }
    ret[1] = j;
    return ret; 
}
```
## 25 Spiral Matrix
#### _medium_
#### 描述：给一个二维数组，返回这个数组的sprial遍历
#### 思路：和Spiral Matrix II题很像，注意遍历顺序，和while的判断条件。
#### 代码：
```
public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<Integer>();
        
        if (matrix.length == 0) {
            return res;
        }
        
        int rowBegin = 0;
        int rowEnd = matrix.length-1;
        int colBegin = 0;
        int colEnd = matrix[0].length - 1;
        
        while (rowBegin <= rowEnd && colBegin <= colEnd) {
            // Traverse Right
            for (int j = colBegin; j <= colEnd; j ++) {
                res.add(matrix[rowBegin][j]);
            }
            rowBegin++;
            
            // Traverse Down
            for (int j = rowBegin; j <= rowEnd; j ++) {
                res.add(matrix[j][colEnd]);
            }
            colEnd--;
            
            if (rowBegin <= rowEnd) {
                // Traverse Left
                for (int j = colEnd; j >= colBegin; j --) {
                    res.add(matrix[rowEnd][j]);
                }
            }
            rowEnd--;
            
            if (colBegin <= colEnd) {
                // Traver Up
                for (int j = rowEnd; j >= rowBegin; j --) {
                    res.add(matrix[j][colBegin]);
                }
            }
            colBegin ++;
        }
        
        return res;
    }
```
## 26 Rotate Image
#### _medium_
#### 描述：给定一个二维数组，求将这个数组顺时针旋转90度。
#### 思路：按一个个位置来变化。并且每变化一个位置，就讲相应的四个位置的值都顺序变化，详见代码。
#### 代码：
```
public void rotate(int[][] matrix) {
        for(int i = 0;i <= (matrix.length - 1)/2;i++){
            for(int j = i;j <= matrix.length - 2 - i;j++){
                swap(matrix,i,j);
            }
        }
    }
    public void swap(int[][] M,int a1,int b1){
        int temp = M[a1][b1];
        M[a1][b1] = M[M.length - 1 - b1][a1];
        M[M.length - 1 - b1][a1] = M[M.length - 1 - a1][M.length - 1 - b1];
        M[M.length - 1 - a1][M.length - 1- b1] = M[b1][M.length - 1- a1];
        M[b1][M.length - 1- a1] = temp;
    }
```
## 27 Generate Parentheses
#### _medium_
#### 描述：给定一个值，求符合规定的括号组合。 
For example, given n = 3, a solution set is:
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
#### 思路：利用溯源法。计算有串中已有的左括号数目和右括号数目。每次溯源都可以选择添加左括号或者右括号。直到字符串到达最大长度。
#### 代码：
```
public List<String> generateParenthesis(int n) {
        List<String> list = new ArrayList<String>();
        backtrack(list, "", 0, 0, n);
        return list;
    }
    
    public void backtrack(List<String> list, String str, int open, int close, int max){
        
        if(str.length() == max*2){
            list.add(str);
            return;
        }
        
        if(open < max)
            backtrack(list, str+"(", open+1, close, max);
        if(close < open)
            backtrack(list, str+")", open, close+1, max);
    }
```
## 28 Product of Array Except Self
#### _medium_
#### 描述：给定一个整数数组，返回一个数组，返回数组h的每个元素等于给定数组除了该元素以外元素的乘积。（不允许用除法）
#### 思路：遍历两次数组，第一次从头到尾，得到一个数组，数组每个值等于当前位置以前的值的乘积。第二次从尾到头遍历数组，保存一个值，这值是当前位置以后树的乘积。这样就可以得到结果。
#### 代码：
```
public int[] productExceptSelf(int[] nums) {
    int n = nums.length;
    int[] res = new int[n];
    res[0] = 1;
    for (int i = 1; i < n; i++) {
        res[i] = res[i - 1] * nums[i - 1];
    }
    int right = 1;
    for (int i = n - 1; i >= 0; i--) {
        res[i] *= right;
        right *= nums[i];
    }
    return res;
}
```
## 29 Find Peak Element
#### _medium_
#### 描述：给定一个数组，求数组中任意一个nums[i] > nums[i-1] && nums[i] > nums[i+1]的数
#### 思路：一开始用的是遍历，复杂度是O（n）.后来看了别人的代码，发现二分查找也能行，就是保证数组的边界旁边的数小于边界。
#### 代码：
```
int findPeakElement(const vector<int> &num) 
    {
        int low = 0;
        int high = num.size()-1;
        
        while(low < high)
        {
            int mid1 = (low+high)/2;
            int mid2 = mid1+1;
            if(num[mid1] < num[mid2])
                low = mid2;
            else
                high = mid1;
        }
        return low;
    }
```
## 30 Unique Paths
#### _medium_
#### 描述：给定一个二维数组，一个机器人从左上角开始，每次只能往下或者往右。问到右下角共有多少不同的路径
#### 思路：简单的动态规划，dp[i[[j] = d[i-1][j]+dp[i][j-1];
#### 代码：
```
public int uniquePaths(int m, int n) {
        int[] res = new int[n];
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j ++){
                if(i == 0)
                    res[j] = 1;
                else if(j == 0)
                    res[j] = 1;
                else
                    res[j] = res[j] + res[j-1];
            }
        }
        return res[n-1];
}
```
## 31 Sort Colors
#### _medium_
#### 描述：给定一个数组，里面包括数组1,2,0.请对数组排序，数组1都在中间，0和2两端。
#### 思路：设两个变量，second表示2之前的数的位置，zero表示0之后数的位置。遍历数组，然后交换数字。下面的代码超级简洁！！！bro
#### 代码：
```
public:
        void sortColors(int A[], int n) {
            int second=n-1, zero=0;
            for (int i=0; i<=second; i++) {
                while (A[i]==2 && i<second) swap(A[i], A[second--]);
                while (A[i]==0 && i>zero) swap(A[i], A[zero++]);
            }
        }
```
## 32 3Sum
#### _medium_
#### 描述：给定一个数组，求数组内三个元素相加为零的集合。
#### 思路：用了溯源法，结果超时。那只能用之前的3Sum close 算法用的方法。看来溯源法还是容易超时。
#### 代码：
```
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new LinkedList<>();
        for(int i = 0;i< nums.length -2;i++){
            if (i == 0 || (i > 0 && nums[i] != nums[i-1])) {
            int left = i + 1;
            int right = nums.length -1;
            int sum = 0 - nums[i];
            while (left < right){
                if(nums[left] + nums[right] == sum){
                    res.add(Arrays.asList(nums[i],nums[left],nums[right]));
                    while (left<right && nums[left] == nums[left+1]) left++;
                    while (left < right && nums[right] == nums[right-1]) right--;
                    left++;
                    right--;
                }
                else if(nums[left] + nums[right] < sum)
                    left++;
                else
                    right--;
            }
            }
        }
        return  res;
    }
```
## 33 Combination Sum
#### _medium_
#### 描述：给定一个数组和一个目标值，求数组内元素相加等于目标值的组合。元素可以重复。
#### 思路：利用回溯法。感觉可以总结出这类题的规律了。总是从一个大集合里找出子集来满足一定的条件。（目前感觉是这样，之后看看符不符合）。
#### 代码：
```
public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(res,new ArrayList<Integer>(),candidates,0,target);
        return res;
    }
    public void backtrack(List<List<Integer>> res,List<Integer> list,int[] candidates,int index,int target){
        if(target == 0)
            res.add(new ArrayList<>(list));
        else{
            while(index < candidates.length){
                if(target >= candidates[index]){
                    list.add(candidates[index]);
                    backtrack(res,list,candidates,index,target-candidates[index]); //这里是允许重复，要不然传入的index得加一
                    list.remove(list.size()-1);
                }
                index++;
            }
        }
    }
```
## 34 Combination Sum II
#### _medium_
#### 描述：和上一题一样，只不过不允许重复
#### 思路：也在上面代码写了，修改一个变量。
#### 代码：
```
public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList();
        Arrays.sort(candidates);
        backtracing(res,new ArrayList(),candidates,target,0);
        return res;
    }
    public void backtracing(List<List<Integer>> list, List<Integer> tempList,int[] nums,int target,int start){
        if(target < 0 ) return;
        else if(target == 0) list.add(new ArrayList<>(tempList));
        else{
            for(int i = start; i <= nums.length - 1; i++){
                if(i > start && nums[i] == nums[i-1]) continue;
                tempList.add(nums[i]);
                backtracing(list,tempList,nums,target - nums[i],i+1);
                tempList.remove(tempList.size() - 1);
            }
        }
    }
```
## 35 First Missing Positive
#### _hard_
#### 描述给定一个整数数组，找到一个缺失的整数，
For example,
Given [1,2,0] return 3,
and [3,4,-1,1] return 2. 
#### 思路：假设数组中有k个正整数，那缺失的正整数一定在区间【1，k+1】之间。所以我们先遍历一遍数组，将正整数全部移到左侧（0，k-1）。然后将数组前k-1位中值对应的位置变成负数。这个再遍历一遍数组，如果有i位置的数值大于0，那么缺失的值是i+1。（我之前有一种错误的思路:先计算数组中正整数的个数和他们的相加的和。这样通过最小的值和数组正整数的个数可以得到连续数组的应该和，和相加和比较，就可以得到缺失的值。但是这样是错误的，因为数组中的正整数不会就缺失一位整数，而其他的都是连续的。比如2,4,6,8。这样这个方法就得不到正确结果）
#### 代码：
```
public int firstMissingPositive(int[] A) {
    int n=A.length;
    if(n==0)
        return 1;
    int k=partition(A)+1;
    int temp=0;
    int first_missing_Index=k;
    for(int i=0;i<k;i++){
        temp=Math.abs(A[i]);
        if(temp<=k)
            A[temp-1]=(A[temp-1]<0)?A[temp-1]:-A[temp-1];
    }
    for(int i=0;i<k;i++){
        if(A[i]>0){
            first_missing_Index=i;
            break;
        }
    }
    return first_missing_Index+1;
}

public int partition(int[] A){
    int n=A.length;
    int q=-1;
    for(int i=0;i<n;i++){
        if(A[i]>0){
            q++;
            swap(A,q,i);
        }
    }
    return q;
}

public void swap(int[] A, int i, int j){
    if(i!=j){
        A[i]^=A[j];
        A[j]^=A[i];
        A[i]^=A[j];
    }
}
```
## 36 Jump Game II
#### _hard_
#### 描述：给定以一个数组，数组里的值表示最大可以移动的值。问从数组头到数组尾最短移动多少次。
#### 思路：用curEnd表示当前跳能到达最远的距离，curFurther表示下一跳能到达最远的距离。每次更新curend就讲跳数加一。（也可以用动态规划，不过复杂度有O（n* n）。还是之前的方法好）
#### 代码：
```
    public int jump(int[] nums) {
        int jumps = 0;
        int curEnd = 0;
        int curFurther = 0;
        for(int i = 0;i < nums.length - 1; i++){
            curFurther = Math.max(curFurther,i+nums[i]);
            if(i == curEnd){
                jumps++;
                curEnd = curFurther;
            }
        }
        return jumps;
    }
```
## 37 Largest Rectangle in Histogram
#### _hard_
#### 描述：给定一个数组（正整数）。里面的值表示宽度为一的长方形的高。问数组中可以找到最大面积的长方形是多少？
#### 思路：创建两个数组lessFromLeft和lessFromRight。分别表示比当前值小的左侧位置和右侧位置。这样面积就可以用height[i] * (lessFromRight[i] - lessFromLeft[i] - 1)求值。创建数组的目的就是能在O（1）时间内找到比当前值小的左侧和右侧位置。
#### 代码：
```
public int largestRectangleArea(int[] height) {
        if (height == null || height.length == 0) {
        return 0;
    }
    int[] lessFromLeft = new int[height.length]; // idx of the first bar the left that is lower than current
    int[] lessFromRight = new int[height.length]; // idx of the first bar the right that is lower than current
    lessFromRight[height.length - 1] = height.length;
    lessFromLeft[0] = -1;

    for (int i = 1; i < height.length; i++) {
        int p = i - 1;

        while (p >= 0 && height[p] >= height[i]) {
            p = lessFromLeft[p];
        }
        lessFromLeft[i] = p;
    }

    for (int i = height.length - 2; i >= 0; i--) {
        int p = i + 1;

        while (p < height.length && height[p] >= height[i]) {
            p = lessFromRight[p];
        }
        lessFromRight[i] = p;
    }

    int maxArea = 0;
    for (int i = 0; i < height.length; i++) {
        maxArea = Math.max(maxArea, height[i] * (lessFromRight[i] - lessFromLeft[i] - 1));
    }

    return maxArea;
    }
```
## 38 Best Time to Buy and Sell Stock III
#### _hard_
#### 描述：给定一个数组，数组中的值表示股票的价格，求操作后得到最大的利润。（只允许买卖两次）
#### 思路：有两种思路，一种是专门针对只允许买卖k次的（soltion1）。复杂度O(kn)。第二次则针对两次或者三次买卖的（多的就复杂了）（solution2）。代码里都有足够提示。PS：这类题真的难，得花时间记住。
#### 代码：
#### solution 1
```
int maxProfit(vector<int> &prices) {
        // f[k, ii] represents the max profit up until prices[ii] (Note: NOT ending with prices[ii]) using at most k transactions. 
        // f[k, ii] = max(f[k, ii-1], prices[ii] - prices[jj] + f[k-1, jj]) { jj in range of [0, ii-1] }
        //          = max(f[k, ii-1], prices[ii] + max(f[k-1, jj] - prices[jj]))
        // f[0, ii] = 0; 0 times transation makes 0 profit
        // f[k, 0] = 0; if there is only one price data point you can't make any money no matter how many times you can trade
        if (prices.size() <= 1) return 0;
        else {
            int K = 2; // number of max transation allowed
            int maxProf = 0;
            vector<vector<int>> f(K+1, vector<int>(prices.size(), 0));
            for (int kk = 1; kk <= K; kk++) {
                int tmpMax = f[kk-1][0] - prices[0];
                for (int ii = 1; ii < prices.size(); ii++) {
                    f[kk][ii] = max(f[kk][ii-1], prices[ii] + tmpMax);
                    tmpMax = max(tmpMax, f[kk-1][ii] - prices[ii]);
                    maxProf = max(f[kk][ii], maxProf);
                }
            }
            return maxProf;
        }
    }
```
#### solution2
```
public int maxProfit(int[] prices) {
        int hold1 = Integer.MIN_VALUE, hold2 = Integer.MIN_VALUE;
        int release1 = 0, release2 = 0;
        for(int i:prices){                              // Assume we only have 0 money at first
            release2 = Math.max(release2, hold2+i);     // The maximum if we've just sold 2nd stock so far.
            hold2    = Math.max(hold2,    release1-i);  // The maximum if we've just buy  2nd stock so far.
            release1 = Math.max(release1, hold1+i);     // The maximum if we've just sold 1nd stock so far.
            hold1    = Math.max(hold1,    -i);          // The maximum if we've just buy  1st stock so far. 
        }
        return release2; ///Since release1 is initiated as 0, so release2 will always higher than release1.
    }
```
## 39 Find Minimum in Rotated Sorted Array II
#### _hard_
#### 描述：给定一个递增有序的数组，但是数组循环右移了几位，求这个数组的最小值（数值可以重复）。
#### 思路：又是这种问题，对于循环右移的问题，找到最小值或者特定值。一般利用二分查找就可以解决问题，因为循环总会导致半边有序半边无序，我们可以对比数组两端来判断是否有序。但是如果有重复数组并且是找特定值，就得多判断几次，因为光判断中间和数组两端并不能判断特定值在哪个半边。
#### 代码：
```
int findMin(vector<int> &num) {
        int lo = 0;
        int hi = num.size() - 1;
        int mid = 0;
        
        while(lo < hi) {
            mid = lo + (hi - lo) / 2;
            
            if (num[mid] > num[hi]) {
                lo = mid + 1;
            }
            else if (num[mid] < num[hi]) {
                hi = mid;
            }
            else { // when num[mid] and num[hi] are same
                hi--;
            }
        }
        return num[lo];
    }
```
## 40 Maximal Rectangle
#### _hard_
#### 描述：给定一个二维数组，里面是0和1，问数组中由1组成的矩形面积最大为多少。
#### 思路：按一行行来计算最大的面积。每层只和上一层做比较。求出这一位的左边界和右边界（这些计算要考虑到上一层的左边界和右边界），这样我们求的面积就可能包括上面一层的情况。有点难以描述，看代码吧==！
#### 代码：
```
public int maximalRectangle(char[][] matrix) {
        if(matrix.length == 0 || matrix[0].length == 0)
            return 0;
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[] hign = new int[cols];
        int[] left = new int[cols];
        int[] right = new int[cols];
        Arrays.fill(right,cols);
        int res = 0;
        for(int i= 0;i < rows;i++){
            int cur_left = 0;
            int cur_rigth = cols;
            for(int j = 0;j < cols;j++){
                if(matrix[i][j] == '1'){
                    left[j] = Math.max(left[j],cur_left);
                    hign[j]++;
                }else{
                    hign[j] = 0;
                    left[j] = 0;
                    cur_left = j+1;
                }
            }
            for(int j = cols -1;j >= 0;j--){
                if(matrix[i][j] == '1')
                    right[j] = Math.min(right[j],cur_rigth);
                else{
                    right[j] = cols;
                    cur_rigth = j;
                }
            }
            for(int j = 0; j < cols;j++)
                res = Math.max(res,(right[j]-left[j]) * hign[j]);
        }
        return res;
    }
```
## 41  Insert Interval
#### _hard_
#### 描述：给定一连串的区间，问加入一个区间后，合并相同区间，求最后的区间
 Example 1:
Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].

Example 2:
Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16]. 
#### 思路：和正常想到方法一样，找到要合并的区间，合并。然后返回。主要是数据结构要选好，代码得简洁。
#### 代码：
```
public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
         List<Interval> result = new LinkedList<>();
        int i = 0;
    // add all the intervals ending before newInterval starts
        while (i < intervals.size() && intervals.get(i).end < newInterval.start)
            result.add(intervals.get(i++));
    // merge all overlapping intervals to one considering newInterval
        while (i < intervals.size() && intervals.get(i).start <= newInterval.end) {
            newInterval = new Interval( // we could mutate newInterval here also
                    Math.min(newInterval.start, intervals.get(i).start),
                    Math.max(newInterval.end, intervals.get(i).end));
            i++;
        }
        result.add(newInterval); // add the union of intervals we got
    // add all the rest
        while (i < intervals.size()) result.add(intervals.get(i++)); 
        return result;
    }
```
## 42 Longest Consecutive Sequence
#### _hard_
#### 描述：给定一个数组，问有子元素组成的最长连续串的长度（子元素不需要连续）
For example,
Given [100, 4, 200, 1, 3, 2],
The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4. 
#### 思路：利用hashMap来查找，只有边界元素才存储连续串的长度。每次添加新元素的时候，查看新元素是否会改变边界，如果改变了，就更新边界元素的值。
#### 代码：
```
public int longestConsecutive(int[] nums) {
        int res = 0;
        HashMap<Integer,Integer> myMap = new HashMap<Integer,Integer>();
        for(int n : nums){
            if(!myMap.containsKey(n)){
                int left = (myMap.containsKey(n-1)) ? myMap.get(n-1) : 0;
                int right = (myMap.containsKey(n+1)) ? myMap.get(n+1) : 0;
                int lengt = left + right + 1;
                res = Math.max(res,lengt);
                myMap.put(n,lengt);
                myMap.put(n-left,lengt);
                myMap.put(n+right,lengt);
            }
        }
        return res;
    }
```
## 43 Trapping Rain Water
#### _hard_
#### 描述：给定一个数组，数组里的值代表宽度为一的矩形高度。问有这些矩形组成的图形，最多能存储多少水？
#### 思路：从数组两端遍历，首先比较两端的高度，小的一方说明自己这边高度不是最大，可以存储水。并同时更新两端的高度。
#### 代码：
```
int trap(int A[], int n) {
        int left=0; int right=n-1;
        int res=0;
        int maxleft=0, maxright=0;
        while(left<=right){
            if(A[left]<=A[right]){
                if(A[left]>=maxleft) maxleft=A[left];
                else res+=maxleft-A[left];
                left++;
            }
            else{
                if(A[right]>=maxright) maxright= A[right];
                else res+=maxright-A[right];
                right--;
            }
        }
        return res;
    }
```
## 44 Word Ladder II
#### _hard_
#### 描述：给定String start,string end和List<String>list。start每次只从list中挑选一个海明距离为1的string。问最短从start变形为end
For example,
Given:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]

Return
  [
    ["hit","hot","dot","dog","cog"],
    ["hit","hot","lot","log","cog"]
  ]

#### 思路：首先是找到每个string在list中距离为1的所有string（下面代码是用迪杰斯特拉来求得的，并且求得了最短路径），然后利用BFS，来选择结果。
#### 代码：
```
Map<String,List<String>> map;
List<List<String>> results;
public List<List<String>> findLadders(String start, String end, List<String> dict) {   	
        results= new ArrayList<List<String>>();
        if (dict.size() == 0)
			return results;
        
        int min=Integer.MAX_VALUE;
        
        Queue<String> queue= new ArrayDeque<String>();
        queue.add(start);
        
	map = new HashMap<String,List<String>>();
		
	Map<String,Integer> ladder = new HashMap<String,Integer>();
	for (String string:dict)
		ladder.put(string, Integer.MAX_VALUE);
	ladder.put(start, 0);
				
	dict.add(end);
	//BFS: Dijisktra search
	while (!queue.isEmpty()) {
		   
		String word = queue.poll();
			
		int step = ladder.get(word)+1;//'step' indicates how many steps are needed to travel to one word. 
			
		if (step>min) break;
			
		for (int i = 0; i < word.length(); i++){
			StringBuilder builder = new StringBuilder(word); 
			for (char ch='a';  ch <= 'z'; ch++){
				builder.setCharAt(i,ch);
				String new_word=builder.toString();				
				if (ladder.containsKey(new_word)) {
						
					if (step>ladder.get(new_word))//Check if it is the shortest path to one word.
					    continue;
					else if (step<ladder.get(new_word)){
					    queue.add(new_word);
					    ladder.put(new_word, step);
					}else;// It is a KEY line. If one word already appeared in one ladder,
					 // Do not insert the same word inside the queue twice. Otherwise it gets TLE.
					    
					if (map.containsKey(new_word)) //Build adjacent Graph
					    map.get(new_word).add(word);
					else{
					    List<String> list= new LinkedList<String>();
					    list.add(word);
					    map.put(new_word,list);
					   //It is possible to write three lines in one:
					   //map.put(new_word,new LinkedList<String>(Arrays.asList(new String[]{word})));
					    //Which one is better?
					}
					    
					if (new_word.equals(end))
					    min=step;

				}//End if dict contains new_word
			}//End:Iteration from 'a' to 'z'
		}//End:Iteration from the first to the last
	}//End While

 //BackTracking
	LinkedList<String> result = new LinkedList<String>();
	backTrace(end,start,result);

	return results;        
}
private void backTrace(String word,String start,List<String> list){
    	if (word.equals(start)){
    		list.add(0,start);
    		results.add(new ArrayList<String>(list));
    		list.remove(0);
    		return;
    	}
    	list.add(0,word);
    	if (map.get(word)!=null)
    		for (String s:map.get(word))
    			backTrace(s,start,list);
    	list.remove(0);
}
```
## 45 Median of Two Sorted Arrays
#### _hard_
#### 描述：给定两个有序的数组，求这两个数组的中位数。
#### 思路：假设中位数分别在两个数组中的i,j位置。
left_part                |        right_part
A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]
所以可以得到以下结论
(1) i + j == m - i + n - j (or: m - i + n - j + 1)
    if n >= m, we just need to set: i = 0 ~ m, j = (m + n + 1)/2 - i
(2) B[j-1] <= A[i] and A[i-1] <= B[j]
所以要找满足这两个条件的i和j。首先我们找每个数组的中间，如果B[j-1] > A[i]则说明i太小了。就划分到右半边（imin = i+1）。否则在左半边。另外对于边界条件（i== 0或者i == m）则直接在另外的数组寻找就可以了。
#### 代码：
```
 def median(A, B):
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) / 2
    while imin <= imax:
        i = (imin + imax) / 2
        j = half_len - i
        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0
```
## 46 House Robber
#### _easy_
#### 描述：给定一个数组代表房子里面的钱。一个小偷在相连的房子中只能拿一家的钱，问小偷最多拿多少钱
#### 思路：简单的动态规划，dp[i]=Math.max(dp[i-1],dp[i-1]+nums[i])。我之前有个错误就是dp[1]我初始化的时候是nums[1].但是实际上得Math.max(nums[1],nums[0]).
#### 代码：
```
    public int rob(int[] nums) {
        if(nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        for(int i = 0;i < nums.length;i++){
            if(i == 0)
                dp[i] = nums[i];
            else if(i == 1)
                dp[i] = Math.max(nums[i],dp[i-1]);
            else
                dp[i] = Math.max(nums[i]+dp[i-2],dp[i-1]);
        }
        return dp[nums.length-1];
    }
```
## 47 Best Time to Buy and Sell Stock
#### _easy_
#### 描述：Best Time to Buy and Sell Stock2 的简易版本，只允许买卖一次
#### 思路：和之前的思路一样
#### 代码：
```
public int maxProfit(int[] prices) {
        int len = prices.length;
        if(len == 0)
            return 0;
        int temp = -prices[0];
        int res = 0;
        for(int i = 1;i < len;i++){
            temp = Math.max(temp,-prices[i]);
            res = Math.max(res,prices[i]+temp);
        }
        return res;
    }
```
## 48 Maximum Subarray
#### _easy_
#### 描述：给定一个数组，求数组中子串的最大和。
#### 思路：以前一直的想到的是：遍历数组，一直相加。结果用一个变量存储（maxEndingHere），如果大于至今的最大结果就更新。如果变量小于零了，就将变量更新为0。solution2代码则是用的dp。solution1代码则结合了我之前的思想和dp。
#### 代码：
##### solution1
```
public static int maxSubArray(int[] A) {
    int maxSoFar=A[0], maxEndingHere=A[0];
    for (int i=1;i<A.length;++i){
    	maxEndingHere= Math.max(maxEndingHere+A[i],A[i]);
    	maxSoFar=Math.max(maxSoFar, maxEndingHere);	
    }
    return maxSoFar;
}
```
##### solution2
```
public int maxSubArray(int[] A) {
        int n = A.length;
        int[] dp = new int[n];//dp[i] means the maximum subarray ending with A[i];
        dp[0] = A[0];
        int max = dp[0];
        
        for(int i = 1; i < n; i++){
            dp[i] = A[i] + (dp[i - 1] > 0 ? dp[i - 1] : 0);
            max = Math.max(max, dp[i]);
        }
        
        return max;
}
```
## 49 Climbing Stairs
#### _easy_
#### 描述：给定一个n阶的楼梯，你从第零阶开始，每次能走一步或者两步，问到达终点时有多少种走法
#### 思路：简单的dp:dp[i]=dp[i-1]+dp[i-2]。（斐波那契数列）
#### 代码：
```
public int climbStairs(int n) {
        if(n == 0) return 0;
        if(n <= 2) return n==2 ?  2: 1;
        int t1 = 1;
        int t2 = 2;
        for(int i = 3;i <= n;i++){
            int temp = t1 + t2;
            t1 = t2;
            t2 = temp;
        }
        return t2;
    }
```
## 50 Unique Binary Search Trees II
#### _medium_
#### 描述：给定目标值n,问从1到n可以组成多少种不同的二分查找数
#### 思路：我一开始看错题了，以为要返回的是多少种。不过思路也差不多，就新增一个数n后，数列变成了1,2,3...n。可以这样想，如果以i为根节点，1到i-1为左子树dp[i-1]，i+1到n为右子树dp[n-i-1],这样以i为根节点的总数为dp[i-1] * dp[n-i-1]。再依次将i从0到n。就求得了新加一个数n总类别了。
#### 代码：
```
public static List<TreeNode> generateTrees(int n) {
    List<TreeNode>[] result = new List[n + 1];
    result[0] = new ArrayList<TreeNode>();
    if (n == 0) {
        return result[0];
    }

    result[0].add(null);
    for (int len = 1; len <= n; len++) {
        result[len] = new ArrayList<TreeNode>();
        for (int j = 0; j < len; j++) {
            for (TreeNode nodeL : result[j]) {
                for (TreeNode nodeR : result[len - j - 1]) {
                    TreeNode node = new TreeNode(j + 1);
                    node.left = nodeL;
                    node.right = clone(nodeR, j + 1);
                    result[len].add(node);
                }
            }
        }
    }
    return result[n];
}

private static TreeNode clone(TreeNode n, int offset) {
    if (n == null) {
        return null;
    }
    TreeNode node = new TreeNode(n.val + offset);
    node.left = clone(n.left, offset);
    node.right = clone(n.right, offset);
    return node;
}
```
## 51 Triangle
#### _medium_
#### 描述：给一个类似杨辉三角的三角形（数值不同，形状相同），求从头到尾的最小值
#### 思路：简单dp:dp[i] = Min(dp[i+1],dp[i]）+nums[i]。下面代码写的简洁
#### 代码：
```
int minimumTotal(vector<vector<int> > &triangle) {
    int n = triangle.size();
    vector<int> minlen(triangle.back());
    for (int layer = n-2; layer >= 0; layer--) // For each layer
    {
        for (int i = 0; i <= layer; i++) // Check its every 'node'
        {
            // Find the lesser of its two children, and sum the current value in the triangle with it.
            minlen[i] = min(minlen[i], minlen[i+1]) + triangle[layer][i]; 
        }
    }
    return minlen[0];
}
```
## 52 House Robber II
#### _medium_
#### 描述：和之前的偷盗题一样，都是偷盗的房间不能相连，但是这一题的是1~n组成了一个圈，问小偷的最大收益
#### 思路：将圈分成两个情况0~n-2和1~n-1，比较两种的最大收益
#### 代码：
```
public int rob(int[] nums) {
    if (nums.length == 1) return nums[0];
    return Math.max(rob(nums, 0, nums.length - 2), rob(nums, 1, nums.length - 1));
}

    private int rob(int[] num, int lo, int hi) {
    int include = 0, exclude = 0;
    for (int j = lo; j <= hi; j++) {
        int i = include, e = exclude;
        include = e + num[j];
        exclude = Math.max(e, i);
    }
    return Math.max(include, exclude);
}
```
## 53 Ugly Number II
#### _medium_
#### 描述：ugly number:因子有2,3,5组成的数。问第n个ugly number是那个数
#### 思路：下面代码的思路是最新的丑数一定是有之前的丑数乘上（2,3,5）得来的，那么假设最新的数n,我们用t1表示，dp[t1]* 2<n<dp[t1+1]* 2,dp[t2]* 3<n<dp[t2+1]* 3,dp[t3]* 5<n<dp[t3+1]* 5。那么最新的一个丑数一定是dp[t1+1]* 2,dp[t2+1]* 3,dp[t3+1]* 5之间的最小的一个数。我的思路则是用了一个hashset存储之前的丑数，遍历全部整数，如果整数n除以二，或者除以三，除以五的值在hashset里，那么就是新的丑数了。
#### 代码：
```
public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        int t2= 0,t3 = 0,t5 = 0;
        dp[0] = 1;
        for(int i = 1;i < n;i++){
            dp[i] = Math.min(dp[t2] * 2,Math.min(dp[t3]*3,dp[t5]*5));
            if(dp[i] == dp[t2] * 2) t2++;
            if(dp[i] == dp[t3] * 3) t3++;
            if(dp[i] == dp[t5] * 5) t5++;
        }
        return dp[n-1];
    }
```
## 54 Perfect Squares
#### _medium_
#### 描述：给定一个整数n,n能有乘方数（i * i）相加得到。问n最少能被多少个乘方数组成。
#### 思路：完全背包的变种，背包重量是n,物品有重量有（1,4,9...），价值都是一，求最小价值。
#### 代码：
```
public int numSquares(int n) {
        int[] dp = new int[n+1];
        for(int i = 1;i <= n;i++){
            dp[i] = Integer.MAX_VALUE;
            for(int j = 1;j *j <= i;j++){
                dp[i] = Math.min(dp[i],dp[i-j*j]+1);
            }
        }
        return dp[n];
    }
```
## 55 Longest Increasing Subsequence
#### _medium_
#### 描述：给定一个数组，问该数组的最长递增子数组的长度是多少，子数组元素之间的前后位置和在原数组的位置保持不变。
For example,
Given [10, 9, 2, 5, 3, 7, 101, 18],
The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length. 
#### 思路：设立一个数组tail，tail[i]表示将原数组划分成长度为i+1的递增子数组，其中所以子数组最后元素中的最小的元素就是tail[i]。
For example, say we have nums = [4,5,6,3], then all the available increasing subsequences are:

len = 1   :      [4], [5], [6], [3]   => tails[0] = 3
len = 2   :      [4, 5], [5, 6]       => tails[1] = 5
len = 3   :      [4, 5, 6]            => tails[2] = 6
由此我们可以得到tail是递增的，并且如果新增一个数x，这个数大于所有的tail,tail可以把长度加一，最后一个复制为x。如果tail[i-1] < x <= tail[i]，我们就更新tail[i-1] = x。按所有的元素遍历，最后tail的长度就是最长递增子数组。可以看看下面代码，简洁的不行。
#### 代码：
```
public int lengthOfLIS(int[] nums) {
    int[] tails = new int[nums.length];
    int size = 0;
    for (int x : nums) {
        int i = 0, j = size;
        while (i != j) {
            int m = (i + j) / 2;
            if (tails[m] < x)
                i = m + 1;
            else
                j = m;
        }
        tails[i] = x;
        if (i == size) ++size;
    }
    return size;
}
```
