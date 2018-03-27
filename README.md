# Leetcode
## plusOne
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
## Pascal's Triangle II
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
## Two Sum II - Input array is sorted
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
## Rotate Array
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
##
#### _ _
#### 描述：
#### 思路：
#### 代码：
```
```
##
#### _ _
#### 描述：
#### 思路：
#### 代码：
```
```
##
#### _ _
#### 描述：
#### 思路：
#### 代码：
```
```
