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
#### 思路：有两种方法，递归和dp。递归就是将字符串一分为二:s1（0，i,len）。如果(isScramble(s1.substring(0,i), s2.substring(0,i)) && isScramble(s1.substring(i), s2.substring(i)))存在，或者(isScramble(s1.substring(0,i), s2.substring(s2.length()-i))  && isScramble(s1.substring(i), s2.substring(0,s2.length()-i)))存在。那么s1和s2是scramble。dp思路也类似；用三维数组dp[i][j][k]来存储值。i表示s1的起始位置,j表示s2的起始位置。k表示长度。dp[i][j][k]表示s1.substring(i,i+k-1)和s2.substring(j,j+k-1)是scramble。这样dp[i][j][len] = || (dp[i][j][k]&&dp[i+k][j+k][len-k] || dp[i][j+len-k][k]&&dp[i+k][j][len-k]) 对于所有1<=k<len，也就是对于所有len-1种劈法的结果求或运算，时间复杂度O(n^4),一般而言，字符串问题一般可以用dp来解决。下面代码是递归的方法表示。有一些小技巧来提前判断是不是scramble的。
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
## 62 Interleaving String
#### _hard_
#### 描述：给定字符串 s1,s2,s3。问是s3是否有s1,s2组成
For example,
Given:
s1 = "aabcc",
s2 = "dbbca",

When s3 = "aadbbcbcac", return true.
When s3 = "aadbbbaccc", return false. 
#### 思路：用dp，建立一个二维数组搭配dp[i][j]表示s1的第i位和s2的第j位和s3的第i+j位interleaving。递推公式dp[i][j] = (dp[i-1][j] && s1.charAt(i-1) == s3.charAt(i + j - 1)) ||(dp[i][j-1] && s2.charAt(j-1) == s3.charAt(i + j - 1));我之前用dp是用的三维数组，最后一维存放的是s3的长度，后来发现可以用i和j来计算。这个题有个误区就是，以为可以像归并排序一样来组成s3.如果s1和s2字符都不一样就可以，一旦有相同的字符，那么就不知道选择哪个字符串的值了。所以也可以用递归来做。
#### 代码：
```
public boolean isInterleave(String s1, String s2, String s3) {
        int len1 = s1.length();
        int len2 = s2.length();
        int len3 = s3.length();
        if(len1 + len2 != len3) return false;
        boolean[][] dp = new boolean[len1+1][len2+1];
        for(int i = 0;i <= len1;i++){
            for(int j = 0;j <= len2; j++){
                if(i == 0 && j == 0)
                    dp[i][j] = true;
                else if (i == 0)
                    dp[0][j] = dp[0][j-1] && s2.charAt(j-1) == s3.charAt(j-1);
                else if (j == 0)
                    dp[i][0] = dp[i-1][0] && s1.charAt(i-1) == s3.charAt(i-1);
                else
                    dp[i][j] = (dp[i-1][j] && s1.charAt(i-1) == s3.charAt(i + j - 1)) ||
                               (dp[i][j-1] && s2.charAt(j-1) == s3.charAt(i + j - 1));
            }
        }
        return dp[len1][len2];
    }
```
## 63 Distinct Subsequences
#### _hard_
#### 描述：给定字符串s和T，问s有多少个子串等于T。
#### 思路：利用dp,dp【i】[j]代表s的第i位置和T第j位置不同子串有多少。递推公式为dp[i][j] = dp[i-1][j] + dp[i-1][j-1]（如果s.charAt(i-1) == t.charAt(j-1)）。当然也可以用溯源方法来做。这次的dp方法让我没有想到这样的问题也可以用dp来做，并且递推公式刷新了我对dp的理解。我以前一直以为是dp只能将之前的一个状态转换成新的一个状态。但是这题中是将之前的两个状态合成新的一个状态（其实之前也有，但是那种合并都比较简单容易理解，像这题就不好理解了。感觉是经典dp题了。）
#### 代码：
```
    public int numDistinct(String s, String t) {
        int slen = s.length();
        int tlen = t.length();
        int[][] dp = new int[slen + 1][tlen + 1];
        for(int i = 0;i <= slen;i++) dp[i][0] = 1;
        for(int j = 1;j <= tlen;j++) dp[0][j] = 0;
        for(int i = 1;i <= slen;i++){
            for(int j = 1;j <= tlen;j++){
                if(s.charAt(i-1) == t.charAt(j-1))
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-1];
                else
                    dp[i][j] = dp[i-1][j];
            }
        }
        return dp[slen][tlen];
    }
```
## 64 Palindrome Partitioning II
#### _hard_
#### 描述：给定一个字符串，问该字符串的回文子串最少有多少
#### 思路：两种解法，一种是dp，一种是类似暴力的解法。首先说dp，设两个数组，数组cut[i]表示在前i个字符最多有回文子串。pal[i][j]表示从i到j的字符是回文的。递推思想是：如果s[i+1]==s[j+1]且pal[i][j]是回文的，那么pal[i+1][j+1]也是回文的。同时cut[j]=cut[i]+1（这种情况是最小的）。所以我们遍历i和j。从而得到结果（代码solution1）。第二种有点类似于暴力。首先设置cut[i] = i-1,这种是最大情况，然后遍历数组，假设i为回文子串的中间，看该回文子串长度最大是多少，有点类似于57题的求最大回文子串长度的方法，同时更新cut，最后得到结果。
#### 代码：
#### solution 1
```
public int minCut(String s) {
    char[] c = s.toCharArray();
    int n = c.length;
    int[] cut = new int[n];
    boolean[][] pal = new boolean[n][n];
    
    for(int i = 0; i < n; i++) {
        int min = i;
        for(int j = 0; j <= i; j++) {
            if(c[j] == c[i] && (j + 1 > i - 1 || pal[j + 1][i - 1])) {
                pal[j][i] = true;  
                min = j == 0 ? 0 : Math.min(min, cut[j - 1] + 1);
            }
        }
        cut[i] = min;
    }
    return cut[n - 1];
}
```
#### solution 2
```
int minCut(string s) {
        int n = s.size();
        vector<int> cut(n+1, 0);  // number of cuts for the first k characters
        for (int i = 0; i <= n; i++) cut[i] = i-1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; i-j >= 0 && i+j < n && s[i-j]==s[i+j] ; j++) // odd length palindrome
                cut[i+j+1] = min(cut[i+j+1],1+cut[i-j]);

            for (int j = 1; i-j+1 >= 0 && i+j < n && s[i-j+1] == s[i+j]; j++) // even length palindrome
                cut[i+j+1] = min(cut[i+j+1],1+cut[i-j+1]);
        }
        return cut[n];
    }
```
## 65 Dungeon Game
#### _hard_
#### 描述：给定一个二维数组，代表一个矩阵。矩阵上的每个值代表可以增减生命值（负数就是减去生命值）。问一个勇士在左上角，公主在右下角。勇士初始生命值最少多少才能解救公主。
#### 思路：利用dp。从右下角到左上角，我们用dp[i][j]代表所需要的最小生命值。dp[i][j]从下方和右边过来，让dp[i+1][j]和dp[i][j+1]分别与num[i][j]相减，他们的最小值如果小于零就代表生命值足够，就更新dp[i][j]为1.就这样推到左上角。这道题如果从左上角到右下角的推，那么就需要两个矩阵，一个存当前生命，一个存当前所需的最小生命值。
#### 代码：
```
int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int M = dungeon.size();
        int N = dungeon[0].size();
        // hp[i][j] represents the min hp needed at position (i, j)
        // Add dummy row and column at bottom and right side
        vector<vector<int> > hp(M + 1, vector<int>(N + 1, INT_MAX));
        hp[M][N - 1] = 1;
        hp[M - 1][N] = 1;
        for (int i = M - 1; i >= 0; i--) {
            for (int j = N - 1; j >= 0; j--) {
                int need = min(hp[i + 1][j], hp[i][j + 1]) - dungeon[i][j];
                hp[i][j] = need <= 0 ? 1 : need;
            }
        }
        return hp[0][0];
    }
```
## 66 Best Time to Buy and Sell Stock IV
#### _hard_
#### 描述：还是买卖股票，这次限定k次，问最大利润
#### 思路：首先判断k是否大于长度的一半，大于长度的一半就表明相当于可以无限次的买卖。如果不大于，就是之前的问题了。之前买卖两次的时候用了那个方法。
#### 代码：
```
public int maxProfit(int k, int[] prices) {
	 int plen = prices.length;
        if(plen < 2) return 0;
        if(k >= plen / 2){
            int ans = 0;
            for(int i = 1;i < plen;i++){
                if(prices[i] - prices[i-1] > 0)
                    ans += (prices[i] - prices[i-1]);
            }
            return ans;
        }
        int[][] dp = new int[k+1][plen];
        for(int i = 1;i <= k;i++){
            int tmpMax = -prices[0];
            for(int j = 1;j < plen;j++){
                dp[i][j] = Math.max(prices[j]+tmpMax,dp[i][j-1]);
                tmpMax = Math.max(tmpMax,dp[i-1][j]-prices[j]);
            }
        }
        return dp[k][plen-1];
}
```
## 67 Regular Expression Matching
#### _hard_
#### 描述：模式匹配，‘.’表示可以匹配任意字符，‘ * ’表示可以有任意个字符（包括零个）
#### 思路：dp，从后往前。如果当前字符和匹配串相等或匹配串为‘.’。dp[i][j] = dp[i][j+2] || firstMatch && dp[i+1][j]。溯源法应该也可以做。
#### 代码：
```
public boolean isMatch(String s, String p) {
        boolean[][] dp = new boolean[s.length()+1][p.length()+1];
        dp[s.length()][p.length()] = true;
        for(int i = s.length();i >=0;i--){
            for(int j = p.length()-1;j>= 0;j--){
                boolean firstMatch = (i < s.length() && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.'));
                if(j + 1 < p.length() && p.charAt(j+1) == '*')
                    dp[i][j] = dp[i][j+2] || firstMatch && dp[i+1][j];
                else
                    dp[i][j] = firstMatch && dp[i+1][j+1];
            }
        }
        return dp[0][0];
    }
```
## 68 Word Break II
#### _hard_
#### 描述：给一个字符串s和一个字符串集合list，问list抽取元素组成s，问有多少种组成方法
 For example, given
s = "catsanddog",
dict = ["cat", "cats", "and", "sand", "dog"].

A solution is ["cats and dog", "cat sand dog"]. 
#### 思路：
#### 代码：
```
HashMap<String,List<String>> map = new HashMap<String,List<String>>();
    public List<String> wordBreak(String s, Set<String> wordDict) {
        List<String> res = new ArrayList<String>();
        if(s == null || s.length() == 0) {
            return res;
        }
        if(map.containsKey(s)) {
            return map.get(s);
        }
        if(wordDict.contains(s)) {
            res.add(s);
        }
        for(int i = 1 ; i < s.length() ; i++) {
            String t = s.substring(i);
            if(wordDict.contains(t)) {
                List<String> temp = wordBreak(s.substring(0 , i) , wordDict);
                if(temp.size() != 0) {
                    for(int j = 0 ; j < temp.size() ; j++) {
                        res.add(temp.get(j) + " " + t);
                    }
                }
            }
        }
        map.put(s , res);
        return res;
    }
```
## 69 Valid Palindrome
#### _easy_
#### 描述：给定一个字符串，看看是否符合回文子串，符号。
#### 思路：简单的匹配回文串的问题
#### 代码：
```
    public boolean isPalindrome(String s) {
        if (s.isEmpty()) {
        	return true;
        }
        int head = 0, tail = s.length() - 1;
        char cHead, cTail;
        while(head <= tail) {
        	cHead = s.charAt(head);
        	cTail = s.charAt(tail);
        	if (!Character.isLetterOrDigit(cHead)) {
        		head++;
        	} else if(!Character.isLetterOrDigit(cTail)) {
        		tail--;
        	} else {
        		if (Character.toLowerCase(cHead) != Character.toLowerCase(cTail)) {
        			return false;
        		}
        		head++;
        		tail--;
        	}
        }
        
        return true;
    }
```
## 70 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 71 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 72 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 73 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 74 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 75 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 76 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 77 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 78 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 79 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
## 80 
#### _easy_
#### 描述：
#### 思路：
#### 代码：
```

```
