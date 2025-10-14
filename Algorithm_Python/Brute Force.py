# 14889 스타트와 링크
# 내 코드 -> 시간 초과

'''
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


# https://www.youtube.com/watch?v=vOqtJotB5Ps

'''

# 1065 한수
'''
N=int(input())
cnt=0

for i in range(1,N+1):
    lst=list(map(int,str(i)))
    if len(lst)<3:
        cnt+=1
    else:
        gap=lst[0]-lst[1]
        for j in range(1,len(lst)-1):
            if lst[j]-lst[j+1]!=gap:
                break
        else:
            cnt+=1

print(cnt)

'''

# 14501 퇴사

'''
마지막 날부터 그리디 하게 생각 -> 변수가 많음 

마지막 날짜 부터 접근

아무 일도 아직 x때
- 퇴사 전까지 가능 하다면: P누적, day기록

뒤에서 무슨일을 할때
- 마감전에 끝낼 수 있는 일이라면
1. 무슨 일을 하는 날짜(들)를 포함할 때
- 무슨 일을 했을 때까지 이득과 지금 하려는 일 중 큰 P누적, day기록

2. 포함하지 않을 때
- 마지막 P에 현재 P누적

3. 조금만 포함할 때
- 현재일 P + 가능한 날짜일 까지

'''

# 방법1 백트래킹
'''
모든 경우의 수를 따져봄
'''
'''
N = int(input())
arr = []
ans=0
for _ in range(N):
    t, p = map(int, input().split())
    arr.append((t, p))

def dfs(n,sm):
    global ans

    if n>N:
        return
    else:
        ans = max(ans, sm)
        if n!=N:
            dfs(n+arr[n][0],sm+arr[n][1])
            dfs(n+1,sm)

dfs(0,0)


print(ans)

'''

# 방법2 DP
'''
N = int(input())
arr = []
dp=[0]*(N+1)
for _ in range(N):
    t, p = map(int, input().split())
    arr.append((t, p))

for i in range(len(arr)-1,-1,-1):
    t,p=arr[i][0],arr[i][1]

    if i+t<=N:  # 상담이 가능하다면
        dp[i]=max(dp[i+t]+p,dp[i+1])
    else:
        dp[i]=dp[i+1]

print(max(dp))

'''

# 14502 연구소
# 내 풀이 -> 실패
'''
N,M=map(int,input().split())
arr=[]
ans=1e9

for _ in range(N):
    arr.append(list(map(int,input().split())))


def dfs(i, j):
    global cnt
    v=arr

    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]

    for k in range(4):
        ni = i + dx[k]
        nj = j + dy[k]
        if 0 <= ni < N and 0 <= nj < M:
            if v[ni][nj] == 0:
                cnt += 1
                v[ni][nj] = 1
                dfs(ni, nj)

    return


# 백트랙킹 칸막이 설치, DFS 수행
def fun(n):
    global ans

    if n==3:
        # dfs 수행
        cnt=0
        for i in range(N):
            for j in range(M):
                if arr[i][j] == 2:
                    dfs(i, j)
        return

    for i in range(N):
        for j in range(M):
            if arr[i][j] == 0:
                arr[i][j]=1
                fun(n+1)
                arr[i][j]=0


# main
fun(0)
print(ans)

'''

# 풀이

'''
TOP-DOWN 방식으로 풀이 진행, 큰 틀을 세우고 필요시 함수 구현

[방법1] 백트랙킹

1. 별도의 빈 공간 좌표를 저장하는 리스트와 바이러스 좌표를 저장하는 리스트 사용(덜 복잡하게 접근)

2.빈공간 리스트에서 백트랙킹을 사용해서 벽을 세울 수 있는 좌표 체크 (방문 리스트1)

3. 빈공간 3개를 만들었을 때 BFS를 통해 바이러스 확산 시키고 빈공간 체크 (방문 리스트2)

DFS 사용할 경우
- 전역 변수를 사용해야함
- 이중 for문을 통해 바이러스 위치를 확인하고 진행해야함

BFS 사용할 경우
- 바이러스 좌표를 큐에 넣고 바로 수행하면 됨
- 따라서 해당 문제는 BFS 사용이 적합

[방법2] 루프
- 3중 루프를 통해 가능한 조합 모두 살피기
- 3개를 뽑는다는 조건이 있어서 가능, K개라고 할 경우 사용X, 무조건 백트랙킹

'''

# 14502 연구소
# 방법 1(백트랙킹) 풀이
'''
from collections import deque
import sys
input=sys.stdin.readline

N,M=map(int,input().split())
arr=[]
ans=0 # 오염 되지 않은 공간 개수

for _ in range(N):
    arr.append(list(map(int,input().split())))

lst=[] # 빈공간 리스트
vir=[] # 바이러스 리스트


for i in range(N):
    for j in range(M):
        if arr[i][j]==0:
            lst.append((i,j))
        elif arr[i][j]==2:
            vir.append((i,j))


v=[0]*(len(lst)) # 벽 세우는 좌표 방문 리스트(방문 리스트1) - 백트랙킹에 사용

# BFS 함수
def bfs(tlst):
    cnt = len(lst) - 3  # 빈공간 개수에서 벽개수(3)빼준 개수: 초기 개수

    # 벽만들기
    for i, j in tlst:
        arr[i][j] = 1

    # BFS
    q = deque()
    w = [[0]*M for _ in range(N)]  # BFS 방문리스트(방문리스트2)
    for i, j in vir:
        q.append((i, j))  # 바이러스 위치(시작점) 삽입
        w[i][j]=1

    while q:
        ci,cj = q.popleft()

        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni = ci + di
            nj = cj + dj
            if 0 <= ni < N and 0 <= nj < M:
                if arr[ni][nj] == 0 and w[ni][nj] != 1:
                    q.append((ni,nj))
                    w[ni][nj] = 1
                    cnt -= 1

    # 벽해체 -> Point!!!!!
    for i, j in tlst:
        arr[i][j] = 0

    return cnt


# 백트랙킹 wiht BFS
def dfs(n,tlst):
    global ans

    if n==3:
        ans=max(ans,bfs(tlst))
        return

    for j in range(len(lst)):
        if v[j]!=1:
            v[j]=1
            dfs(n+1,tlst+[lst[j]])
            v[j]=0


dfs(0,[])
print(ans)

'''

# 방법2
# 루프
'''
from collections import deque
import sys
input=sys.stdin.readline

N,M=map(int,input().split())
arr=[]
ans=0 # 오염 되지 않은 공간 개수

for _ in range(N):
    arr.append(list(map(int,input().split())))

lst=[] # 빈공간 리스트
vir=[] # 바이러스 리스트


for i in range(N):
    for j in range(M):
        if arr[i][j]==0:
            lst.append((i,j))
        elif arr[i][j]==2:
            vir.append((i,j))


v=[0]*(len(lst)) # 벽 세우는 좌표 방문 리스트(방문 리스트1) - 백트랙킹에 사용

# BFS 함수
def bfs(tlst):
    cnt = len(lst) - 3  # 빈공간 개수에서 벽개수(3)빼준 개수: 초기 개수

    # 벽만들기
    for i, j in tlst:
        arr[i][j] = 1

    # BFS
    q = deque()
    w = [[0]*M for _ in range(N)]  # BFS 방문리스트(방문리스트2)
    for i, j in vir:
        q.append((i, j))  # 바이러스 위치(시작점) 삽입
        w[i][j]=1

    while q:
        ci,cj = q.popleft()

        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni = ci + di
            nj = cj + dj
            if 0 <= ni < N and 0 <= nj < M:
                if arr[ni][nj] == 0 and w[ni][nj] != 1:
                    q.append((ni,nj))
                    w[ni][nj] = 1
                    cnt -= 1

    # 벽해체 -> Point!!!!!
    for i, j in tlst:
        arr[i][j] = 0

    return cnt


for i in range(len(lst)-2):
    for j in range(len(lst)-1):
        for k in range(len(lst)):
            ans=max(ans,bfs([lst[i],lst[j],lst[k]]))


print(ans)

'''

# 16439 치킨치킨치킨
# 풀이1 루프
'''
import sys
input=sys.stdin.readline

N,M=map(int,input().split())
arr=[list(map(int,input().split())) for _ in range(N)]
ans=0

# 3개의 열을 선택하는 모든 경우의 수 돌리고, 각 선택된 열에 행 최대값 누적
for i in range(M-2):
    for j in range(M-1):
        for k in range(M):
            sm=0
            for l in range(N):
                sm+=max(arr[l][i],arr[l][j],arr[l][k])
            if ans<sm:
                ans=sm

print(ans)

'''

# 14888 연산자 끼워넣기
'''
import sys
input=sys.stdin.readline

N=int(input())
lst=list(map(int,input().split()))
add,sub,mul,mod=map(int,input().split())
mn=int(1e9)
mx=int(-1e9)

def dfs(n,sm,add,sub,mul,mod):
    global mn,mx

    if sm<int(-1e9) or sm>int(1e9):
        return
    if n==len(lst):
        mx=max(mx,sm)
        mn=min(mn,sm)
        return

    if add>0:
        dfs(n+1,sm+lst[n],add-1,sub,mul,mod)
    if sub>0:
        dfs(n+1,sm-lst[n],add,sub-1,mul,mod)
    if mul>0:
        dfs(n+1,sm*lst[n],add,sub,mul-1,mod)
    if mod>0:
        dfs(n+1,int(sm/lst[n]),add,sub,mul,mod-1)

dfs(1,lst[0],add,sub,mul,mod)
print(mx,mn,sep='\n')

'''

# 1182 부분수열의 합
'''
N,S=map(int,input().split())
lst=list(map(int,input().split()))
ans=0

def dfs(n,sm):
    global ans

    if n==N: # 마지막 인덱스일때
        if len(sm)>0 and sum(sm)==S: # 누적합이 S와 같다면
            ans+=1 # 개수 1추가
        return

    dfs(n+1,sm+[lst[n]]) # 포함시켰을 때
    dfs(n+1,sm) # 포함시키지 않았을 때

dfs(0,[])
print(ans)

'''
