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
## 70 Valid Parentheses
#### _easy_
#### 描述：给定一个有"{","}","(",")","[","]"组成的字符串，问该字符串是否符合括号的规范
#### 思路：看代码吧，太简单。
#### 代码：
```
public boolean isValid(String s) {
	Stack<Character> stack = new Stack<Character>();
	for (char c : s.toCharArray()) {
		if (c == '(')
			stack.push(')');
		else if (c == '{')
			stack.push('}');
		else if (c == '[')
			stack.push(']');
		else if (stack.isEmpty() || stack.pop() != c)
			return false;
	}
	return stack.isEmpty();
}
```
## 71 Implement strStr()
#### _easy_
#### 描述：给定两个字符串 S ，P。问p是否是s的子串，如果是返回s的位置，不是则返回1
#### 思路：kmp算法。（可算是写出来了==;）
#### 代码：
```
    int[] next;
    public int strStr(String haystack, String needle) {
        next = new int[needle.length()];
        if(needle.length() == 0)
            return 0;
        getnext(needle);
        int j = 0;
        int i = 0;
        while(i < haystack.length() && j < needle.length()){
            if(j == -1 || needle.charAt(j) == haystack.charAt(i)){
                i++;
                j++;
            }
            else
                j = next[j];
        }
        if(j == needle.length())
            return i-j;
        else
            return -1;
    }
    public void getnext(String p){
        next[0] = -1;
        int j = -1;
        int i = 0;
        while(i < p.length()-1){
            if(j== -1 || p.charAt(i) == p.charAt(j)){
                i++;
                j++;
                next[i] = j;
            }
            else
                j = next[j];
        }
    }
```
## 72 Length of Last Word
#### _easy_
#### 描述：给定一个字符串，里面用空格隔开各个单词，返回最后一个单词的长度
#### 思路：要考虑到多个空格和最后一个是空格的问题。
#### 代码：
```
public int lengthOfLastWord(String s) {
        int res = 0;
        int pre = 0;
        for(int i = 0;i < s.length();i++){
            if(s.charAt(i) == ' '){
                pre = res == 0 ? pre :res;
                res = 0;
            }
            else
                res ++;
        }
        if(s.length() > 0 && s.charAt(s.length()-1) == ' ')
            return pre;
        else 
            return res;
    }
```
## 73 Add Binary
#### _easy_
#### 描述：求两个二进制的加法
#### 思路：看代码。我的算法是先加，之后算进位，但是不知道哪里错了
#### 代码：
```
 public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1, j = b.length() -1, carry = 0;
        while (i >= 0 || j >= 0) {
            int sum = carry;
            if (j >= 0) sum += b.charAt(j--) - '0';
            if (i >= 0) sum += a.charAt(i--) - '0';
            sb.append(sum % 2);
            carry = sum / 2;
        }
        if (carry != 0) sb.append(carry);
        return sb.reverse().toString();
    }
```
## 74 ZigZag Conversion
#### _medium_
#### 描述：给定一个字符串s,讲s按照z字型排列，问输出结果是多少？
for example：
s = "PAYPALISHIRING"
P   A   H   N
A P L S I I G
Y   I   R
返回 "PAHNAPLSIIGYIR"
convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR". 
#### 思路：创建一个n行的字符串数组，按照z字依次每行给字符串加上字符，最后组装成一行。这题不能用先求第一行再求第二行的方法来，会很麻烦。
#### 代码：
```
public String convert(String s, int numRows) {
        char[] c = s.toCharArray();
        int len = c.length;
        StringBuffer[] sb = new StringBuffer[numRows];
        for (int i = 0; i < sb.length; i++) sb[i] = new StringBuffer();
    
        int i = 0;
        while (i < len) {
            for (int idx = 0; idx < numRows && i < len; idx++) // vertically down
                sb[idx].append(c[i++]);
            for (int idx = numRows-2; idx >= 1 && i < len; idx--) // obliquely up
                sb[idx].append(c[i++]);
        }
        for (int idx = 1; idx < sb.length; idx++)
            sb[0].append(sb[idx]);
        return sb[0].toString();
    }
```
## 75 String to Integer (atoi)
#### _medium_
#### 描述：给定一个字符串，返回对应的整数
#### 思路：主要考虑到字符串为空，有非法字符，正负号，溢出这几种情况
#### 代码：
```
public int myAtoi(String str) {
    int index = 0, sign = 1, total = 0;
    //1. Empty string
    if(str.length() == 0) return 0;

    //2. Remove Spaces
    while(str.charAt(index) == ' ' && index < str.length())
        index ++;

    //3. Handle signs
    if(str.charAt(index) == '+' || str.charAt(index) == '-'){
        sign = str.charAt(index) == '+' ? 1 : -1;
        index ++;
    }
    
    //4. Convert number and avoid overflow
    while(index < str.length()){
        int digit = str.charAt(index) - '0';
        if(digit < 0 || digit > 9) break;

        //check if total will be overflow after 10 times and add digit
        if(Integer.MAX_VALUE/10 < total || Integer.MAX_VALUE/10 == total && Integer.MAX_VALUE %10 < digit)
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;

        total = 10 * total + digit;
        index ++;
    }
    return total * sign;
}
```
## 76 Restore IP Addresses
#### _medium_
#### 描述：给定一个字符串，返回可能的ip地址集合
For example:
Given "25525511135",
return ["255.255.11.135", "255.255.111.35"]. (Order does not matter) 
#### 思路：溯源法可以做，或者直接三次for循环
#### 代码：
```
public List<String> restoreIpAddresses(String s) {
    List<String> solutions = new ArrayList<String>();
    restoreIp(s, solutions, 0, "", 0);
    return solutions;
}

private void restoreIp(String ip, List<String> solutions, int idx, String restored, int count) {
    if (count > 4) return;
    if (count == 4 && idx == ip.length()) solutions.add(restored);
    
    for (int i=1; i<4; i++) {
        if (idx+i > ip.length()) break;
        String s = ip.substring(idx,idx+i);
        if ((s.startsWith("0") && s.length()>1) || (i==3 && Integer.parseInt(s) >= 256)) continue;
        restoreIp(ip, solutions, idx+i, restored+s+(count==3?"" : "."), count+1);
    }
}
```
## 77 Group Anagrams
#### _medium_
#### 描述：给定一组字符串，返回各组字母组成相同的单词。
For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
Return:
[
  ["ate", "eat","tea"],
  ["nat","tan"],
  ["bat"]
]
#### 思路：首先利用排序算法，这样只要字母组成相同的单词都是相同的。然后利用hashMap节省查找时间，以排完序的字符串为key，组成单词字母相同的字符串组为value。这个方法其实也是遍历，我自己的想法是如果都是字母的话，就将每个字母的int值相加得到sum，然后将所有的sum相同的放在一起组成一个list，然后再通过对list分析其字母组成划分不同的list，这样复杂度应该为O(n)。少掉了对字符串排序的时间。
#### 代码：
```
public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0) return new ArrayList<List<String>>();
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String s : strs) {
            char[] ca = s.toCharArray();
            Arrays.sort(ca);
            String keyStr = String.valueOf(ca);
            if (!map.containsKey(keyStr)) map.put(keyStr, new ArrayList<String>());
            map.get(keyStr).add(s);
        }
        return new ArrayList<List<String>>(map.values());
    }
```
## 78 Simplify Path
#### _medium_
#### 描述：给一个字符串，该字符串表示Linux路径命令，返回最后的路径
For example,
path = "/home/", => "/home"
path = "/a/./b/../../c/", => "/c"
#### 思路：用双端队列，如果遇到'..'就弹出上一个文件夹（pop方法）。否则就加入到头插队列中（push）。最后将队列中的路径返回成一个字符串。（注意下面代码不能用stack代替deque。原因是最后合成一个的方法不一样。）。我之前的思路是利用stringbuilder，直接暴力。太可怕了，我为啥如此的暴力。
#### 代码：
```
public String simplifyPath(String path) {
    Deque<String> stack = new LinkedList<>();
    Set<String> skip = new HashSet<>(Arrays.asList("..",".",""));
    for (String dir : path.split("/")) {
        if (dir.equals("..") && !stack.isEmpty()) stack.pop();
        else if (!skip.contains(dir)) stack.push(dir);
    }
    String res = "";
    for (String dir : stack) res = "/" + dir + res;
    return res.isEmpty() ? "/" : res;
}
```
## 79 Compare Version Numbers
#### _medium_
#### 描述：设定一个字符串，两个整数用‘.’来间隔。这种字符串的大小比较是先比较前面的，前面相同再比较后面的。问给定两个这种字符串。判断大小
0.1 < 1.1 < 1.2 < 13.37
#### 思路：看代码，最重要的是看看split("\\.")对特殊字符的转换。
#### 代码：
```
public int compareVersion(String version1, String version2) {
    String[] levels1 = version1.split("\\.");
    String[] levels2 = version2.split("\\.");
    
    int length = Math.max(levels1.length, levels2.length);
    for (int i=0; i<length; i++) {
    	Integer v1 = i < levels1.length ? Integer.parseInt(levels1[i]) : 0;
    	Integer v2 = i < levels2.length ? Integer.parseInt(levels2[i]) : 0;
    	int compare = v1.compareTo(v2);
    	if (compare != 0) {
    		return compare;
    	}
    }
    
    return 0;
}
```
## 80 Multiply Strings
#### _medium_
#### 描述：给出两个整数（用string表示）,求返回这两个数的乘积
#### 思路：按照最简单的乘法表做，先求得每个位的值。最后在考虑进位。最后输出。下面代码是优化版。这类题不能像投机，就得慢慢写，今天还做了阿里的编程测验题，是阿拉伯数字转中文表达的。难啊
#### 代码：
```
public String multiply(String num1, String num2) {
    int m = num1.length(), n = num2.length();
    int[] pos = new int[m + n];
   
    for(int i = m - 1; i >= 0; i--) {
        for(int j = n - 1; j >= 0; j--) {
            int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0'); 
            int p1 = i + j, p2 = i + j + 1;
            int sum = mul + pos[p2];

            pos[p1] += sum / 10;
            pos[p2] = (sum) % 10;
        }
    }  
    
    StringBuilder sb = new StringBuilder();
    for(int p : pos) if(!(sb.length() == 0 && p == 0)) sb.append(p);
    return sb.length() == 0 ? "0" : sb.toString();
}
```
## 81 Generate Parentheses
#### _medium_
#### 描述：给定一个数n，返回有效的括号组合
#### 思路：溯源法。注意右括号不能大于左括号数目。
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
## 82 Reverse Words in a String
#### _medium_
#### 描述：给定一个包含空格的字符，将字符串按单词反序。
For example,
Given s = "the sky is blue",
return "blue is sky the". 
#### 思路：和之前循环右移的题有点类似，先反序整个字符串，在反转单词，最后清理空格。这样的思路很好，空间复杂度为O(n)。其实用栈最简单，比较繁琐的是从后往前抽取单词，添加到新字符串中。
#### 代码：
```
public String reverseWords(String s) {
    if (s == null) return null;
    
    char[] a = s.toCharArray();
    int n = a.length;
    
    // step 1. reverse the whole string
    reverse(a, 0, n - 1);
    // step 2. reverse each word
    reverseWords(a, n);
    // step 3. clean up spaces
    return cleanSpaces(a, n);
  }
  
  void reverseWords(char[] a, int n) {
    int i = 0, j = 0;
      
    while (i < n) {
      while (i < j || i < n && a[i] == ' ') i++; // skip spaces
      while (j < i || j < n && a[j] != ' ') j++; // skip non spaces
      reverse(a, i, j - 1);                      // reverse the word
    }
  }
  
  // trim leading, trailing and multiple spaces
  String cleanSpaces(char[] a, int n) {
    int i = 0, j = 0;
      
    while (j < n) {
      while (j < n && a[j] == ' ') j++;             // skip spaces
      while (j < n && a[j] != ' ') a[i++] = a[j++]; // keep non spaces
      while (j < n && a[j] == ' ') j++;             // skip spaces
      if (j < n) a[i++] = ' ';                      // keep only one space
    }
  
    return new String(a).substring(0, i);
  }
  
  // reverse a[] from a[i] to a[j]
  private void reverse(char[] a, int i, int j) {
    while (i < j) {
      char t = a[i];
      a[i++] = a[j];
      a[j--] = t;
    }
  }
```
## 83 Longest Substring Without Repeating Characters
#### _medium_
#### 描述：给定一个字符串，返回最大连续不重复子串长度。
#### 思路：遍历数组，创建一个hashMap存储字符和它对应的位置。每次都查看hashMap看是否有重复，有则更新子串的起始位置。
#### 代码：
```
public int lengthOfLongestSubstring(String s) {
        if (s.length()==0) return 0;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max=0;
        for (int i=0, j=0; i<s.length(); ++i){
            if (map.containsKey(s.charAt(i))){
                j = Math.max(j,map.get(s.charAt(i))+1); //这个很巧妙，更新子串的起始位置j，取最大的位置。
            }
            map.put(s.charAt(i),i);
            max = Math.max(max,i-j+1);
        }
        return max;
    }
```
## 84 Letter Combinations of a Phone Number
#### _medium_
#### 描述：给定一串数字，按照老版手机键盘的顺序，返回可能的字母组合。
Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
#### 思路：
#### 代码：
```
    public List<String> letterCombinations(String digits) {
    LinkedList<String> ans = new LinkedList<String>();
    if(digits.isEmpty()) return ans;
    String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    ans.add("");
    for(int i =0; i<digits.length();i++){
        int x = Character.getNumericValue(digits.charAt(i));
        while(ans.peek().length()==i){
            String t = ans.remove();
            for(char s : mapping[x].toCharArray())
                ans.add(t+s);
        }
    }
    return ans;
}
```
## 85 Basic Calculator II
#### _medium_
#### 描述：给定一个字符串，代表一个加减乘除式子。求该式子的结果
#### 思路：用stack存储结果，如果是乘除法，就直接计算出结果，否则就进入栈中。详细看代码。我之前做的是用两个栈来做，一个存数字，一个存操作。比较复杂，我还忘了多位数字的情况（有十位百位的情况）。太菜了。
#### 代码：
```
public int calculate(String s) {
    int len;
    if(s==null || (len = s.length())==0) return 0;
    Stack<Integer> stack = new Stack<Integer>();
    int num = 0;
    char sign = '+';
    for(int i=0;i<len;i++){
        if(Character.isDigit(s.charAt(i))){
            num = num*10+s.charAt(i)-'0';
        }
        if((!Character.isDigit(s.charAt(i)) &&' '!=s.charAt(i)) || i==len-1){
            if(sign=='-'){
                stack.push(-num);
            }
            if(sign=='+'){
                stack.push(num);
            }
            if(sign=='*'){
                stack.push(stack.pop()*num);
            }
            if(sign=='/'){
                stack.push(stack.pop()/num);
            }
            sign = s.charAt(i);
            num = 0;
        }
    }

    int re = 0;
    for(int i:stack){
        re += i;
    }
    return re;
}
```
## 86 Integer to English Words
#### _hard_
#### 描述：给定一个数字，返回英文的表达式
#### 思路：和之前在阿里测试做的返回中文的类似，都是按照一块一块来处理的。中文是一万以下，英文是一千以下。其他的类似。
#### 代码：
```
private final String[] LESS_THAN_20 = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
private final String[] TENS = {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
private final String[] THOUSANDS = {"", "Thousand", "Million", "Billion"};

public String numberToWords(int num) {
    if (num == 0) return "Zero";

    int i = 0;
    String words = "";
    
    while (num > 0) {
        if (num % 1000 != 0)
    	    words = helper(num % 1000) +THOUSANDS[i] + " " + words;
    	num /= 1000;
    	i++;
    }
    
    return words.trim();
}

private String helper(int num) {
    if (num == 0)
        return "";
    else if (num < 20)
        return LESS_THAN_20[num] + " ";
    else if (num < 100)
        return TENS[num / 10] + " " + helper(num % 10);
    else
        return LESS_THAN_20[num / 100] + " Hundred " + helper(num % 100);
}
```
## 87 Valid Number
#### _hard_
#### 描述：给定一个字符串，判断是否是合理的数字
#### 思路：要判断是否为小数，整数，指数（2e10）。
#### 代码：
```
s = s.trim();
    
    boolean pointSeen = false;
    boolean eSeen = false;
    boolean numberSeen = false;
    boolean numberAfterE = true;
    for(int i=0; i<s.length(); i++) {
        if('0' <= s.charAt(i) && s.charAt(i) <= '9') {
            numberSeen = true;
            numberAfterE = true;
        } else if(s.charAt(i) == '.') {
            if(eSeen || pointSeen) {
                return false;
            }
            pointSeen = true;
        } else if(s.charAt(i) == 'e') {
            if(eSeen || !numberSeen) {
                return false;
            }
            numberAfterE = false;
            eSeen = true;
        } else if(s.charAt(i) == '-' || s.charAt(i) == '+') {
            if(i != 0 && s.charAt(i-1) != 'e') {
                return false;
            }
        } else {
            return false;
        }
    }
    
    return numberSeen && numberAfterE;
```
## 88 Substring with Concatenation of All Words
#### _hard_
#### 描述：给定一个字符串s，和一组长度相同的字符串list。问list全部元素组成的字符串，是否是s的子串，如果是返回各个起始位置的集合
#### 思路：因为每个list的元素长度都相同，那么就从s的i(i 从0到字符串末尾)位置开始遍历，并且看s(i+j* len,i+(j+1) * len）(j从0开始)。是否是list元素之一。知道把list里面的元素全部遍历完。返回i的值。其中可以建立一个关于list的hashMap和遍历之后的hashMap，这样就可以快速查明是list的某元素是否已经遍历过了。还可以用滑动窗口来做,将每个list的元素当做一个字母，就和Minimum Window Substring一样了，不过为了加速，就采用了写提前处理。
#### 代码：
#### solution1
```
public List<Integer> findSubstring(String s, String[] words) {
        final Map<String, Integer> counts = new HashMap<>();
        for (final String word : words) {
            counts.put(word, counts.getOrDefault(word, 0) + 1);
        }
        final List<Integer> indexes = new ArrayList<>();
        final int n = s.length(), num = words.length, len = words[0].length();
        for (int i = 0; i < n - num * len + 1; i++) {
            final Map<String, Integer> seen = new HashMap<>();
            int j = 0;
            while (j < num) {
                final String word = s.substring(i + j * len, i + (j + 1) * len);
                if (counts.containsKey(word)) {
                    seen.put(word, seen.getOrDefault(word, 0) + 1);
                    if (seen.get(word) > counts.getOrDefault(word, 0)) {
                        break;
                    }
                } else {
                    break;
                }
                j++;
            }
            if (j == num) {
                indexes.add(i);
            }
        }
        return indexes;
    }
```
#### solution2
```
public List<Integer> findSubstring(String s, String[] words) {
	int N = s.length();
	List<Integer> indexes = new ArrayList<Integer>(s.length());
	if (words.length == 0) {
		return indexes;
	}
	int M = words[0].length();
	if (N < M * words.length) {
		return indexes;
	}
	int last = N - M + 1;
	
	//map each string in words array to some index and compute target counters
	Map<String, Integer> mapping = new HashMap<String, Integer>(words.length);
	int [][] table = new int[2][words.length];
	int failures = 0, index = 0;
	for (int i = 0; i < words.length; ++i) {
		Integer mapped = mapping.get(words[i]);
		if (mapped == null) {
			++failures;
			mapping.put(words[i], index);
			mapped = index++;
		}
		++table[0][mapped];
	}
	
	//find all occurrences at string S and map them to their current integer, -1 means no such string is in words array
	int [] smapping = new int[last];
	for (int i = 0; i < last; ++i) {
		String section = s.substring(i, i + M);
		Integer mapped = mapping.get(section);
		if (mapped == null) {
			smapping[i] = -1;
		} else {
			smapping[i] = mapped;
		}
	}
	
	//fix the number of linear scans
	for (int i = 0; i < M; ++i) {
		//reset scan variables
		int currentFailures = failures; //number of current mismatches
		int left = i, right = i;
		Arrays.fill(table[1], 0);
		//here, simple solve the minimum-window-substring problem
		while (right < last) {
			while (currentFailures > 0 && right < last) {
				int target = smapping[right];
				if (target != -1 && ++table[1][target] == table[0][target]) {
					--currentFailures;
				}
				right += M;
			}
			while (currentFailures == 0 && left < right) {
				int target = smapping[left];
				if (target != -1 && --table[1][target] == table[0][target] - 1) {
					int length = right - left;
					//instead of checking every window, we know exactly the length we want
					if ((length / M) ==  words.length) {
						indexes.add(left);
					}
					++currentFailures;
				}
				left += M;
			}
		}
		
	}
	return indexes;
}
```
## 89 Palindrome Pairs
#### _hard_
#### 描述：给定一组字符串，问任意两个字符串是否能组成回文串。返回所有结果（不允许重复结果）
 Example 1:
Given words = ["bat", "tab", "cat"]
Return [[0, 1], [1, 0]]
The palindromes are ["battab", "tabbat"]

Example 2:
Given words = ["abcd", "dcba", "lls", "s", "sssll"]
Return [[0, 1], [1, 0], [3, 2], [2, 4]]
The palindromes are ["dcbaabcd", "abcddcba", "slls", "llssssll"]
#### 思路：有两种方法。一种最简单的暴力，从i个元素开始，遍历i+1到最后一个元素。看组成的元素是否是回文子串。第二种是利用回文串的特性，如果字符串s.substring(0, j)是回文子串的话，加上s.substring(j)和s.substring(j)的反序就能组成一个回文串。对比两种方法，复杂度都差不多，第一种适合字符串长度大，字符组小的情况，第二种尤其适合字符串长度小，组内元素多的情况。下面是第二种方法的代码
#### 代码：
```
public List<List<Integer>> palindromePairs(String[] words) {
    List<List<Integer>> res = new ArrayList<>(); 
    if (words == null || words.length < 2) {
        return res;
    }
    Map<String, Integer> map = new HashMap<String, Integer>();
    for (int i = 0; i < words.length; i++) {
        map.put(words[i], i);
    }
    for (int i = 0; i < words.length; i++) {
        for (int j = 0; j <= words[i].length(); j++) {
            String str1 = words[i].substring(0, j);
            String str2 = words[i].substring(j);
            addPair(map, res, str1, str2, i, false);
            if(str2.length() != 0) {
                addPair(map, res, str2, str1, i, true);
            }
        }
    }
    return res;
}
private void addPair(Map<String, Integer> map, List<List<Integer>> res, String str1, String str2, int index, boolean reverse) {
    if (isPalindrome(str1)) {
        String str2rev = new StringBuilder(str2).reverse().toString();
        if (map.containsKey(str2rev) && map.get(str2rev) != index) {
            List<Integer> list = new ArrayList<>();
            if(!reverse) {
                list.add(map.get(str2rev));
                list.add(index);
            } else {
                list.add(index);
                list.add(map.get(str2rev));
            }
            res.add(list);
        }
    }
}
```
## 90 Shortest Palindrome
#### _hard_
#### 描述：给定一个字符串，只能往头部添加字符，返回最小的回文子串
For example:
Given "aacecaaa", return "aaacecaaa".
Given "abcd", return "dcbabcd".
#### 思路：两种思路：第一种利用kmp的next数组。对于字符串s，扩建成ss= s+“#”+s.reverse.对ss求next数组，得到next最后一位的数据i，就向头添加s.substring(i).reverser()。这个和next的特性有关。第二种思路就有点nb了，通过一个for循环，得到一个j。这个j的不可能是s的中点，并且j之后的i不存在s[j] == s[i]。这样就让j后面的字符插入到头部，再对s.substring(0,j)进行相同的处理。详细见代码
#### 代码：
#### solution1:
```
public String shortestPalindrome(String s) {
        int j = 0;
        for(int i = s.length()-1;i >= 0;i--)
            if(s.charAt(i) == s.charAt(j)) j++;
        if(j == s.length()) return s;
        return new StringBuilder(s.substring(j)).reverse().toString() + shortestPalindrome(s.substring(0,j))+s.substring(j);
    }
```
#### solution2:
```
    public String shortestPalindrome(String s) {
        String temp = s + "#" + new StringBuilder(s).reverse().toString();
        int r = getTable(temp)+1;
    
    //get the maximum palin part in s starts from 0
    return new StringBuilder(s.substring(r)).reverse().toString() + s;
    }
    public int getTable(String p){
        int[] next = new int[p.length()];
        next[0] = -1;
        int j = -1;
        int i = 0;
        while(i < p.length()-1){
            if(j== -1 || p.charAt(i) == p.charAt(j)){
                i++;
                j++;
                next[i] = j;
            }
            else
                j = next[j];
        }
        return next[p.length()-1];
    }
```
## 91 Minimum Window Substring
#### _hard_
#### 描述：给定两个字符串s,t。返回s包含t所有元素的最短子串。
For example,
S = "ADOBECODEBANC"
T = "ABC"

Minimum window is "BANC".
#### 思路：利用滑动窗口来做，首先创立一个长度128的数组，用于快速读取或者修改数据。接下来用两个变量lo,hi表示当前窗口的左右边界。首先将t中的字符存储到数组中，然后遍历字符串s，数组中相应位置的值自减一。如果将counter（t的长度） == 0 说明当前字符串包含了t的所有字符元素。然后通过一个while循环来找到当前字符的真正头位置（第一个属于t的元素）。然后再遍历s字符串，这时map中大于0的就是第一个属于t的元素的位置，所以如果再次counter == 0 就表示遇到了等于头结点的字符了。然后比较得到最短长度，同时更新头结点（s中接下来属于t的元素）。最后得到结果。
#### 代码：
```
public String minWindow(String s, String t) {
        int[] map = new int[128];
        for(char c : t.toCharArray())
            map[c]++;
        int lo = 0,hi = 0,head = 0,counter = t.length(),len = Integer.MAX_VALUE;
        while(hi < s.length()){
            if(map[s.charAt(hi++)]-- > 0 )
                counter--;
            while(counter == 0){
                if(hi - lo < len)
                len = hi - (head = lo);
                if(map[s.charAt(lo++)]++ == 0) counter ++;
            }
        }
        return len == Integer.MAX_VALUE ? "" : s.substring(head,head+len);
    }
```
## 92 Palindrome Number
#### _easy_
#### 描述：给定一个整数，判断该整数是否是回文的。（不能将整数转换为string）
#### 思路：正常思路是获得给定整数i的长度，然后利用除法和求余数来得到一头一尾,如果相等就接着遍历，如果不是就返回false。但是下面思路却不一样，创建一个整数，代表i的右半部，然后利用除法得到i的左半部，最后判断左右半边是否相等（厉害了）。（ps：正常思路有个值得注意的问题，就是判断是否溢出，有两种方法，第一种就是将int转换成long，第二种就是和integer的最大值比较：result > Integer.MAX_VALUE / 10  || ((result == Integer.MAX_VALUE / 10) && (n % 10 > Integer.MAX_VALUE % 10))）
#### 代码：
```
public boolean isPalindrome(int x) {
    if (x<0 || (x!=0 && x%10==0)) return false;
    int rev = 0;
    while (x>rev){
    	rev = rev*10 + x%10;
    	x = x/10;
    }
    return (x==rev || x==rev/10);
}
```
## 93 Excel Sheet Column Title
#### _easy_
#### 描述：给定一种编码方式：1 -> A，2 -> B，3 -> C...26 -> Z，27 -> AA，28 -> AB。给定一个整数，返回正确的string 
#### 思路：26进制，不过要注意的是每次循环的时候都得减去1。因为它没有0的编码。（减去1真的卡了我好久。。。麻蛋）
#### 代码：
```
return n == 0 ? "" : convertToTitle(--n / 26) + (char)('A' + (n % 26));

```
## 94 Factorial Trailing Zeroes
#### _easy_
#### 描述：给定一个整数n，求n!尾部有多少个零
#### 思路：因为得到0,一定是2 * 5,并且由2组成的数一定是比由5组成的数多，那么该问题就转换为n里面有多少个数，求得5的个数。应为像25,125等数提供了2,3个5。则问题就转换为n/5 + n/25 + n/125.....就变成了一下的代码。（这个难度真的是easy吗？？！！！）
#### 代码：
```
public int trailingZeroes(int n) {
        return n == 0 ? 0 : n / 5 + trailingZeroes(n / 5);
    }
```
## 95 Sqrt(x)
#### _easy_
#### 描述：给定一个数，求该数开方，如果数不是整数，就切除小数部分。
#### 思路：一开始用遍历，从零开始，找到第一个数平方大于等于n的数。但是复杂度过高。后来就用二叉查找的方法。
#### 代码：
```
public int sqrt(int x) {
    if (x == 0)
        return 0;
    int left = 1, right = Integer.MAX_VALUE;
    while (true) {
        int mid = left + (right - left)/2; 
        if (mid > x/mid) {
            right = mid - 1;
        } else {
            if (mid + 1 > x/(mid + 1))
                return mid;
            left = mid + 1;
        }
    }
}
```
## 96 Excel Sheet Column Number
#### _easy_
#### 描述：和之前的Excel Sheet Column Title题正好相反，给定一个string，返回相应的编码整数
#### 思路：26进制。这才是简单题啊。
#### 代码：
```
public int titleToNumber(String s) {
        if(s.equals("")) return 0;
        int res = 0;
        for(char c : s.toCharArray()){
            int t = c - 'A' + 1;
            res = res * 26 + t;
        }
        return res;
    }
```
## 97 Happy Number
#### _easy_
#### 描述：给定一个数，将他每个位上的数平方相加，最后看是否为1,如果不是再次操作，问最后是否为1。
#### 思路：我一开始以为不会出现操作后等于操作前的数的，结果是有的。那有种思路就是和判断链表中是否有环的思路一样。还有一种是利用hashset，来破除无限循环。
#### 代码：
```
int digitSquareSum(int n) {
    int sum = 0, tmp;
    while (n > 0) {
        tmp = n % 10;
        sum += tmp * tmp;
        n /= 10;
    }
    return sum;
}

boolean isHappy(int n) {
    int slow, fast;
    slow = fast = n;
    do {
        slow = digitSquareSum(slow);
        fast = digitSquareSum(fast);
        fast = digitSquareSum(fast);
    } while(slow != fast);
    if (slow == 1) return true;
    else return false;
}
```
## 97 Divide Two Integers
#### _medium_
#### 描述：给定两个数字，求这两个数的商。结果为整数（不允许用除法，乘法，求余操作）
Input: dividend = 10, divisor = 3
Output: 3
#### 思路：首先判断符号位，然后求商，比较快的方法是用二进制的方法求，先求二进制最高位，在利用递归依次求下面的最高位。不过要注意的一点是，被除数有可能是负数的最大值，直接求绝对值不行，所以得转成long型再求。
#### 代码：
```
public int divide(int dividend, int divisor) {
	//Reduce the problem to positive long integer to make it easier.
	//Use long to avoid integer overflow cases.
	int sign = 1;
	if ((dividend > 0 && divisor < 0) || (dividend < 0 && divisor > 0))
		sign = -1;
	long ldividend = Math.abs((long) dividend);
	long ldivisor = Math.abs((long) divisor);
	
	//Take care the edge cases.
	if (ldivisor == 0) return Integer.MAX_VALUE;
	if ((ldividend == 0) || (ldividend < ldivisor))	return 0;
	
	long lans = ldivide(ldividend, ldivisor);
	
	int ans;
	if (lans > Integer.MAX_VALUE){ //Handle overflow.
		ans = (sign == 1)? Integer.MAX_VALUE : Integer.MIN_VALUE;
	} else {
		ans = (int) (sign * lans);
	}
	return ans;
}

private long ldivide(long ldividend, long ldivisor) {
	// Recursion exit condition
	if (ldividend < ldivisor) return 0;
	
	//  Find the largest multiple so that (divisor * multiple <= dividend), 
	//  whereas we are moving with stride 1, 2, 4, 8, 16...2^n for performance reason.
	//  Think this as a binary search.
	long sum = ldivisor;
	long multiple = 1;
	while ((sum+sum) <= ldividend) {
		sum += sum;
		multiple += multiple;
	}
	//Look for additional value for the multiple from the reminder (dividend - sum) recursively.
	return multiple + ldivide(ldividend - sum, ldivisor);
}
```
## 98 Fraction to Recurring Decimal
#### _medium_
#### 描述：给定两个整数，用string表示出他们的商，如果小数部分有重复的地方，用括号把他们括起来。
For example,
 Given numerator = 1, denominator = 2, return "0.5".
 Given numerator = 2, denominator = 1, return "2".
 Given numerator = 2, denominator = 3, return "0.(6)".
#### 思路：首先还是利用除法求得整数部分，然后将余数乘10，等大于除数的时候就再利用除法得到小数位的值，对于重复的小数，我们用hashmap来存储小数部分和该小数部分对应的位置，当再次碰到这个小数部分时，就知道有重复了，在对应位置加上括号即可（这个太nb了）
#### 代码：
```
public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        // "+" or "-"
        res.append(((numerator > 0) ^ (denominator > 0)) ? "-" : "");
        long num = Math.abs((long)numerator);
        long den = Math.abs((long)denominator);
        
        // integral part
        res.append(num / den);
        num %= den;
        if (num == 0) {
            return res.toString();
        }
        
        // fractional part
        res.append(".");
        HashMap<Long, Integer> map = new HashMap<Long, Integer>();
        map.put(num, res.length());
        while (num != 0) {
            num *= 10;
            res.append(num / den);
            num %= den;
            if (map.containsKey(num)) {
                int index = map.get(num);
                res.insert(index, "(");
                res.append(")");
                break;
            }
            else {
                map.put(num, res.length());
            }
        }
        return res.toString();
    }
```
## 99 Rectangle Area
#### _medium_
#### 描述：给定四个二元组，代表两个矩形的左下角和右上角，求两个矩形组成的面积（面积不会超出integer的最大值）
#### 思路：要求的两个矩阵重叠组成的小矩形的四个顶点。（我之前做的是判断各种情况，是否重叠，哪个顶点重叠，太麻烦了）
#### 代码：
```
int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
    int left = max(A,E), right = max(min(C,G), left);
    int bottom = max(B,F), top = max(min(D,H), bottom);
    return (C-A)*(D-B) - (right-left)*(top-bottom) + (G-E)*(H-F);
}
```
## 100 Pow(x, n)
#### _medium_
#### 描述：给定一个double和一个int，求pow(double,int)
#### 思路：利用x ^ n = x ^(n/2) & x^(n/2)这样就可以用递归来实现了，但是要注意负数和n为负数的最小值的情况
#### 代码：
```
public double myPow(double x, int n) {
        if(n == 0) return 1.;
        double res = myPow(x, n / 2); //不直接将n转换为整数，防止n为最小值的情况
        return n % 2 == 0 ? res * res : n < 0 ? res * res * (1 / x) : res * res * x;
    }
```
## 101 Permutation Sequence
#### _medium_
#### 描述：给定两个整数n,k。给出从1到n的组成的序列中第k个序列，序列按字典顺序排序
#### 思路：先求最高位，然后按照接下来的每一位来求。
#### 代码：
```
public String getPermutation(int n, int k) {
    List<Integer> numbers = new ArrayList<>();
    int[] factorial = new int[n+1];
    StringBuilder sb = new StringBuilder();
    
    // create an array of factorial lookup
    int sum = 1;
    factorial[0] = 1;
    for(int i=1; i<=n; i++){
        sum *= i;
        factorial[i] = sum;
    }
    // factorial[] = {1, 1, 2, 6, 24, ... n!}
    
    // create a list of numbers to get indices
    for(int i=1; i<=n; i++){
        numbers.add(i);
    }
    // numbers = {1, 2, 3, 4}
    
    k--;
    
    for(int i = 1; i <= n; i++){
        int index = k/factorial[n-i];
        sb.append(String.valueOf(numbers.get(index)));
        numbers.remove(index); //移除这一位数字
        k-=index*factorial[n-i];
    }
    
    return String.valueOf(sb);
}
```
## 102 Add Two Numbers
#### _medium_
#### 描述：给定两个链表，模拟整数相加。
#### 思路：简单的链表操作，但是得注意进位问题。
#### 代码：
```
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode c1 = l1;
        ListNode c2 = l2;
        ListNode sentinel = new ListNode(0);
        ListNode d = sentinel;
        int sum = 0;
        while (c1 != null || c2 != null) {
            sum /= 10;
            if (c1 != null) {
                sum += c1.val;
                c1 = c1.next;
            }
            if (c2 != null) {
                sum += c2.val;
                c2 = c2.next;
            }
            d.next = new ListNode(sum % 10);
            d = d.next;
        }
        if (sum / 10 == 1)
            d.next = new ListNode(1);
        return sentinel.next;
    }
```
## 103 Max Points on a Line
#### _hard_
#### 描述：给出一连串平面坐标，问最多有多少个点在同一条线上
#### 思路：暴力法了，不过有一些优化在里面，比如利用辗转相除法求最大公约数，利用hashMap存储结果。
#### 代码：
```
        public int maxPoints(Point[] points) {
        	if (points==null) return 0;
        	if (points.length<=2) return points.length;
        	
        	Map<Integer,Map<Integer,Integer>> map = new HashMap<Integer,Map<Integer,Integer>>();
        	int result=0;
        	for (int i=0;i<points.length;i++){ 
        		map.clear();
        		int overlap=0,max=0;
        		for (int j=i+1;j<points.length;j++){
        			int x=points[j].x-points[i].x;
        			int y=points[j].y-points[i].y;
        			if (x==0&&y==0){
        				overlap++;
        				continue;
        			}
        			int gcd=generateGCD(x,y);
        			if (gcd!=0){
        				x/=gcd;
        				y/=gcd;
        			}
        			
        			if (map.containsKey(x)){
        				if (map.get(x).containsKey(y)){
        					map.get(x).put(y, map.get(x).get(y)+1);
        				}else{
        					map.get(x).put(y, 1);
        				}   					
        			}else{
        				Map<Integer,Integer> m = new HashMap<Integer,Integer>();
        				m.put(y, 1);
        				map.put(x, m);
        			}
        			max=Math.max(max, map.get(x).get(y));
        		}
        		result=Math.max(result, max+overlap+1);
        	}
        	return result;
        	
        	
        }
        private int generateGCD(int a,int b){
    
        	if (b==0) return a;
        	else return generateGCD(b,a%b);
        	
        }
```
## 104 Binary Tree Inorder Traversal
#### _medium_
#### 描述：二叉树中序遍历（用迭代不要用递归）
#### 思路：利用栈，详细见代码。
#### 代码：
```
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<Integer>();

    Stack<TreeNode> stack = new Stack<TreeNode>();
    TreeNode cur = root;

    while(cur!=null || !stack.empty()){
        while(cur!=null){
            stack.add(cur);
            cur = cur.left;
        }
        cur = stack.pop();
        list.add(cur.val);
        cur = cur.right;
    }

    return list;
}
```
## 105 Repeated DNA Sequences
#### _medium_
#### 描述：有一串由字符'A','C','G','T'组成的字符串，求所有出现两次及其以上长度为10的子串。比如十一个‘AAAAAAAAAAA’，返回结果‘AAAAAAAAA’。
#### 思路：没啥好说的，就是截取长度为十的，然后依次比较。这个是暴力的，下面用了hashset和位运算来节省时间。真的思想巧妙。
#### 代码：
```
public List<String> findRepeatedDnaSequences(String s) {
    Set<Integer> words = new HashSet<>();
    Set<Integer> doubleWords = new HashSet<>();
    List<String> rv = new ArrayList<>();
    char[] map = new char[26];
    //map['A' - 'A'] = 0;
    map['C' - 'A'] = 1;
    map['G' - 'A'] = 2;
    map['T' - 'A'] = 3;

    for(int i = 0; i < s.length() - 9; i++) {
        int v = 0;
        for(int j = i; j < i + 10; j++) {
            v <<= 2;
            v |= map[s.charAt(j) - 'A'];
        }
        if(!words.add(v) && doubleWords.add(v)) {
            rv.add(s.substring(i, i + 10));
        }
    }
    return rv;
}
```
## 106 Count Primes
#### _medium_
#### 描述：给定一个整数n，问小于n的数内，有多少质数
#### 思路：利用n数组来标记n以内非质数的数。不过要记得的是质数不包括1。
#### 代码：
```
public int countPrimes(int n) {
        boolean[] dp = new boolean[n+1];
        int count = 0;
        for(int i = 2;i < n;i++){
            if(dp[i]==false){
                count++;
                for(int j = 2;j*i < n;j++)
                    dp[j*i]=true;
            }
        }
        return count;
    }
```
## 107 Valid Sudoku
#### _medium_
#### 描述：给定一个9 * 9的二维数组。判断这个二维数组是不是sudoku的。（Sudoku数组由1到9数字组成，可以允许为空。在每一行，每一列都不包括重复数字。并且将二维数组分成9个3 * 3 的矩形，每个矩形内也不包括相同数字）
#### 思路：类似于暴力法，但是还是利用了hashset的特征，用string来标注每个元素的位置和值，（4）1表面在第二行有一个4,3（4）表示在第四列有个4.0(2)2，表示在最上面中间的小矩形中有个2。这样就将这些信息放入到hashset中，如果有重复就表明违反规则。
#### 代码：
```
public boolean isValidSudoku(char[][] board) {
    Set seen = new HashSet();
    for (int i=0; i<9; ++i) {
        for (int j=0; j<9; ++j) {
            if (board[i][j] != '.') {
                String b = "(" + board[i][j] + ")";
                if (!seen.add(b + i) || !seen.add(j + b) || !seen.add(i/3 + b + j/3))
                    return false;
            }
        }
    }
    return true;
}
```
## 108 Copy List with Random Pointer
#### _medium_
#### 描述：给定一个链表，这链表中的元素除了有指向下一个元素的指针外，还有个指向链表中任意元素（可以为null）的指针。
#### 思路：首先想到的是暴力法，先复制完整个链表，并保存新节点和旧节点的映射，和旧节点中random的list。这样在通过一个循环就可以得到新链表中random的值。详细代码见solution2.还有一种想法是先复制链表，但是不是新创立一个链表，而是在复制后的新节点在旧节点后，这样random就可以直接定位到新的节点了，最后在将两个链表解构就行了。厉害了（详细代码见solution1）。
#### 代码：
#### solution1
```
public RandomListNode copyRandomList(RandomListNode head) {
	RandomListNode iter = head, next;

	// First round: make copy of each node,
	// and link them together side-by-side in a single list.
	while (iter != null) {
		next = iter.next;

		RandomListNode copy = new RandomListNode(iter.label);
		iter.next = copy;
		copy.next = next;

		iter = next;
	}

	// Second round: assign random pointers for the copy nodes.
	iter = head;
	while (iter != null) {
		if (iter.random != null) {
			iter.next.random = iter.random.next;
		}
		iter = iter.next.next;
	}

	// Third round: restore the original list, and extract the copy list.
	iter = head;
	RandomListNode pseudoHead = new RandomListNode(0);
	RandomListNode copy, copyIter = pseudoHead;

	while (iter != null) {
		next = iter.next.next;

		// extract the copy
		copy = iter.next;
		copyIter.next = copy;
		copyIter = copy;

		// restore the original list
		iter.next = next;

		iter = next;
	}

	return pseudoHead.next;
}
```
#### solution2
```
    public RandomListNode copyRandomList(RandomListNode head) {
        HashMap<RandomListNode,RandomListNode> oldMapToNew = new HashMap();
        HashMap<RandomListNode,LinkedList<RandomListNode>> randomList = new HashMap();
        
        RandomListNode nowNode = head,preNode = null,newHead = null,newNode = null,randomNode= null;
        while(nowNode != null){
            newNode = new RandomListNode(nowNode.label);
            if(newHead == null){
                preNode = newNode;
                newHead = newNode;
            }
            else{
                preNode.next = newNode;
                preNode = newNode;
            }
            oldMapToNew.put(nowNode,newNode);
            randomNode = nowNode.random;
            if(randomNode != null){
                if(randomList.containsKey(randomNode))
                    randomList.get(randomNode).add(nowNode);
                else{
                    LinkedList<RandomListNode> l = new LinkedList();
                    l.add(nowNode);
                    randomList.put(randomNode,l);
                }
            }
            nowNode = nowNode.next;
        }
        for(RandomListNode r :randomList.keySet()){
            newNode = oldMapToNew.get(r);
            for(RandomListNode rr :randomList.get(r))
                oldMapToNew.get(rr).random = newNode;
        }
        return newHead;
    }
```
## 109 Single Number
#### _medium_
#### 描述：有一个数组，里面除了一个元素出现了一次以外，其他元素全部出现两次。求只出现一次的元素
#### 思路：利用两次异或的原理。
#### 代码：
```
    public int singleNumber(int[] nums) {
        int res = 0;
        for(int i :nums)
            res = res ^ i;
        return res;
    }
```
## 110 Sudoku Solver
#### _hard_
#### 描述：给定一个数独二维列表，求填充这个数独列表
#### 思路：利用回溯法，每次尝试，直到成功
#### 代码：
```
public void solveSudoku(char[][] board) {
        if(board == null || board.length == 0)
            return;
        solve(board);
    }
    
    public boolean solve(char[][] board){
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(board[i][j] == '.'){
                    for(char c = '1'; c <= '9'; c++){//trial. Try 1 through 9
                        if(isValid(board, i, j, c)){
                            board[i][j] = c; //Put c for this cell
                            
                            if(solve(board))
                                return true; //If it's the solution return true
                            else
                                board[i][j] = '.'; //Otherwise go back
                        }
                    }
                    
                    return false;
                }
            }
        }
        return true;
    }
    
    private boolean isValid(char[][] board, int row, int col, char c){
        for(int i = 0; i < 9; i++) {
            if(board[i][col] != '.' && board[i][col] == c) return false; //check row
            if(board[row][i] != '.' && board[row][i] == c) return false; //check column
            if(board[3 * (row / 3) + i / 3][ 3 * (col / 3) + i % 3] != '.' && 
board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) return false; //check 3*3 block
        }
        return true;
    }
```
