# Leetcode Part2 共55道
## 56 Word Break
#### _medium_
#### 描述：给定一个string s，和一个dict(里面存储了多个string)，问dict是否存在元素，能组成s?
For example, given
s = "leetcode",
dict = ["leet", "code"].

Return true because "leetcode" can be segmented as "leet code". 
#### 思路：本来是简单的dp。dp[j] = dp[i] && dict.contains(s.substring(i,j));但是我以为很难。。人生的悲哀莫过于此
#### 代码：
```
public boolean wordBreak(String s, List<String> wordDict) {
       boolean[] f = new boolean[s.length() + 1];  
        f[0] = true;        
        //Second DP
        for(int i=1; i <= s.length(); i++){
            for(int j=0; j < i; j++){
                if(f[j] && wordDict.contains(s.substring(j, i))){
                    f[i] = true;
                    break;
                }
            }
        }
        return f[s.length()];
    }
```
## 57 Longest Palindromic Substring
#### _medium_
#### 描述：给定一个字符串，问字符串的的最大回文子串是多少
#### 思路：对字符串的每个字符，认定这个字符是回文串的中间（所以有两种情况，一种是奇数回文串，另一种是偶数回文串）。然后求这个回文子串的长度。（说好的dp标签呢，咋都不用dp方法呢）
#### 代码：
```
private int lo, maxLen;

public String longestPalindrome(String s) {
	int len = s.length();
	if (len < 2)
		return s;
	
    for (int i = 0; i < len-1; i++) {
     	extendPalindrome(s, i, i);  //assume odd length, try to extend Palindrome as possible
     	extendPalindrome(s, i, i+1); //assume even length.
    }
    return s.substring(lo, lo + maxLen);
}

private void extendPalindrome(String s, int j, int k) {
	while (j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)) {
		j--;
		k++;
	}
	if (maxLen < k - j - 1) {
		lo = j + 1;
		maxLen = k - j - 1;
	}
}}
```
## 58 Decode Ways
#### _medium_
#### 描述：给出一个编码规则，‘A’ 表示‘1’ ，‘B’表示‘2’ ...‘Z’表示“26”。给了一个数字组成的string，问用abc编码有多少种方法
For example,
Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12). 
#### 思路：用dp方法，如果字符都合法的话，dp[i] = dp[i-1]+dp[i-2]。我之前没有用Integer.parseInt(s.substring(i,i+2))<=26来判断字符是否合法，直接根据前一个字符判断，结果做不出来。后来看看代码，发现有两个要注意的地方，一个从后面往前面dp效果好一点，并且如果字符i为0,当前dp[i]为零。这样判断更方便，少了很多其他多余的判断。
#### 代码：
```
public int numDecodings(String s) {
        int n = s.length();
        if (n == 0) return 0;
        
        int[] memo = new int[n+1];
        memo[n]  = 1;
        memo[n-1] = s.charAt(n-1) != '0' ? 1 : 0;
        
        for (int i = n - 2; i >= 0; i--)
            if (s.charAt(i) == '0') continue;
            else memo[i] = (Integer.parseInt(s.substring(i,i+2))<=26) ? memo[i+1]+memo[i+2] : memo[i+1];
        
        return memo[0];
    }
```
## 59 Longest Valid Parentheses
#### _hard_
#### 描述：给定一个字符串，里面包含“（”和“）”，问最长有效括号是多少
For "(()", the longest valid parentheses substring is "()", which has length = 2.
Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4. 
#### 思路：最简单的方法就是用栈，碰见左括号就放入栈中，碰见见右括号就弹出一个左括号，长度加二。下面方法是不用栈的：遍历字符串，碰到右括号，就找到右括号匹配的左括号i,再看i-1位置的是否有有效长度，有就加上。
#### 代码：
```
int longestValidParentheses(string s) {
        if(s.length() <= 1) return 0;
        int curMax = 0;
        vector<int> longest(s.size(),0);
        for(int i=1; i < s.length(); i++){
            if(s[i] == ')' && i-longest[i-1]-1 >= 0 && s[i-longest[i-1]-1] == '('){
                    longest[i] = longest[i-1] + 2 + ((i-longest[i-1]-2 >= 0)?longest[i-longest[i-1]-2]:0);
                    curMax = max(longest[i],curMax);
            }
        }
        return curMax;
    }
```
## 60 Edit Distance
#### _hard_
#### 描述：给定两个字符串是s1,s2。问他们的编辑距离是多少（编辑距离为：s1每次可以选择插入，删除，修改任意字符.多少步后能和s2相同）
#### 思路：对于s1的第i位，s2的第j位。如果他们相同，则dp[i][j] = dp[i-1][j-1]。如果不相同则dp[i][j]为dp[i-1][j]和dp[i][j-1]，dp[i-1[j-1]中选择最小的，并且加上一。PS:一开始以为特别难，但是如果仔细想想其实也没有那么难，可以说简单，看来dp题得确立状态变化。
#### 代码：
```
public int minDistance(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();
        int[][] dp = new int[len1+1][len2+1];
        for(int i = 0;i<=len1;i++) dp[i][0] = i;
        for(int j = 1;j<=len2;j++) dp[0][j] = j;
        dp[0][0] = 0;
        for(int i = 0;i < len1;i++){
            for(int j = 0;j <len2;j++){
                if(word1.charAt(i) == word2.charAt(j))
                    dp[i+1][j+1] = dp[i][j];
                else{
                    int temp = Math.min(dp[i+1][j],dp[i][j+1]);
                    dp[i+1][j+1] = Math.min(temp,dp[i][j]) + 1;
                }
            }
        }
        return dp[len1][len2];
    }
```
## 61 Scramble String
#### _hard_
#### 描述：给定一个字符串，将该字符串用二叉树表示。改变二叉树中某个结点的左右子树位置。得到一个新的字符串。新的字符串和之前的字符串之间就是srcamble的.问给两个字符串，问他们是否是scramble。
例如以下两个字符串就是scramble的
    great               
   /    \
  rg    eat
 / \    /  \
r   g  e   at
           / \
          a   t
     rgeat
    /    \
  gr    eat
 / \    /  \
g   r  e   at
           / \
          a   t
#### 思路：有两种方法，递归和dp。递归就是将字符串一分为二:s1（0，i,len）。如果(isScramble(s1.substring(0,i), s2.substring(0,i)) && isScramble(s1.substring(i), s2.substring(i)))存在，或者(isScramble(s1.substring(0,i), s2.substring(s2.length()-i))  && isScramble(s1.substring(i), s2.substring(0,s2.length()-i)))存在。那么s1和s2是scramble。dp思路也类似；用三维数组dp[i][j][k]来存储值。i表示s1的起始位置,j表示s2的起始位置。k表示长度。dp[i][j][k]表示s1.substring(i,i+k-1)和s2.substring(j,j+k-1)是scramble。这样dp[i][j][len] = || (dp[i][j][k]&&dp[i+k][j+k][len-k] || dp[i][j+len-k][k]&&dp[i+k][j][len-k]) 对于所有1<=k<len，也就是对于所有len-1种劈法的结果求或运算，时间复杂度O(n^4)。下面代码是递归的方法表示。有一些小技巧来提前判断是不是scramble的。
#### 代码：
```
public boolean isScramble(String s1, String s2) {
        if (s1.equals(s2)) return true; 
        
        int[] letters = new int[26];
        for (int i=0; i<s1.length(); i++) {
            letters[s1.charAt(i)-'a']++;
            letters[s2.charAt(i)-'a']--;
        }
        for (int i=0; i<26; i++) if (letters[i]!=0) return false;
    
        for (int i=1; i<s1.length(); i++) {
            if (isScramble(s1.substring(0,i), s2.substring(0,i)) 
             && isScramble(s1.substring(i), s2.substring(i))) return true;
            if (isScramble(s1.substring(0,i), s2.substring(s2.length()-i)) 
             && isScramble(s1.substring(i), s2.substring(0,s2.length()-i))) return true;
        }
        return false;
    }
```
## 62 
#### _hard_
#### 描述：
#### 思路：
#### 代码：
```

```
## 63 
#### _hard_
#### 描述：
#### 思路：
#### 代码：
```

```
## 64 
#### _hard_
#### 描述：
#### 思路：
#### 代码：
```

```
