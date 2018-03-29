# Leetcode 共15道
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
## 13
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```
```
## 14
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```
```
## 15
#### _medium_
#### 描述：
#### 思路：
#### 代码：
```
```
