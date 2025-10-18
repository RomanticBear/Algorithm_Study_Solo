
import sys
input = sys.stdin.readline

N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]
ans = 1e9


# 팀 전력 차 구하기
def power(t1, t2):
    p1, p2 = 0, 0
    for i in range(len(t1)):
        for j in range(len(t1)):
            p1 += arr[t1[i]][t1[j]]
            p2 += arr[t2[i]][t2[j]]

    return abs(p1 - p2)


# 편 가르기
def dfs(n,A,B):
    global ans

    if len(A) == N // 2:
        if len(A)==len(B):
            ans=min(ans,power(A,B))
        return

    if n == N:
        return

    dfs(n + 1, A+[n], B)
    dfs(n + 1, A, B+[n])


dfs(0, [], [])
print(ans)