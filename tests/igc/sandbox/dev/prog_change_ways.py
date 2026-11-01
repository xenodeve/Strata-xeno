def ways(amount, coins):
    dp = [1] + [0] * amount
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]
print(ways(100, [1, 5, 10, 25, 50]), ways(10, [2, 5, 3, 6]), ways(0, [1]), ways(3, []))
memo = {}
def climb(n):
    if n <= 1:
        return 1
    if n not in memo:
        memo[n] = climb(n - 1) + climb(n - 2)
    return memo[n]
print(climb(90), len(memo))
