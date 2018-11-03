# https://takuti.me/note/levenshtein-distance/

def levenshtein(s1, s2):
    """
    >>> levenshtein("kitten", "sitting")
    3
    >>> levenshtein("あいうえお", "あいうえお")
    0
    >>> levenshtein("あいうえお", "かきくけこ")
    5
    """
    n, m = len(s1), len(s2)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,         # insertion
                           dp[i][j - 1] + 1,         # deletion
                           dp[i - 1][j - 1] + cost)  # replacement

    return dp[n][m]


if __name__ == "__main__":
    print(levenshtein("kitten", "sitting"))
