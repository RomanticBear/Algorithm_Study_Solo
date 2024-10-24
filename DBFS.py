# DFS, BFS Youtube(이코테 예제)

# DFS 연습예제

'''
gragh=[
    [], #0번(없는 노드)
    [2,3,8], #1번 노드
    [1,7],
    [4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
    ]

visited=[False]*9

def dfs(gragh,v,visited):

    visited[v]=True #방문 표시
    print(v,end=' ')

    for i in gragh[v]:
        if visited[i]!=True:
            dfs(gragh,i,visited)

dfs(gragh,1,visited)

'''
'''
# 코드설명

1. 첫 노드 방문 -> 방문 표시(True)

2. 인접 노드 전부 접근 -> 1과정 반복(재귀)

+ gragh 위에서 부터 아래까지 노드 번호 순
+ 노드 번호 안헷갈리게 접근하기 위해 첫번째 원소 -> []

'''


# BFS 연습문제
'''
from collections import deque

gragh=[
    [], 
    [2,3,8], 
    [1,7],
    [4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
    ]

visited=[False]*9


def bfs(gragh,start,visited):
    # 시작 노드값 담은 자료형 선언
    queue=deque([start])
    visited[start]=True

    # 큐가 빌 때까지 반복
    while queue:
        v=queue.popleft()
        print(v,end=' ')

        for i in gragh[v]:
            if visited[i] != True:
                queue.append(i)
                visited[i]=True

bfs(gragh,1,visited)
'''
'''
# 코드설명

1. 시작 노드 -> 큐에 담기, 방문 표시
2. 큐에 제일 앞에 담긴 값(v) pop, 인접 노드 중 방문 하지 않은 노드 -> 큐에 추가
3. 2번 과정 반복 -> 큐가 빌 때까지 반복(while문)
'''


# DFS 실전문제 _ 음료수 얼려 먹기

# 내 풀이

'''
# gragh생성
n,m=map(int,input().split())
gragh=[]

for i in range(n):
    gragh.append(list(map(int,input())))


# 상,하,좌,우
dx=[-1,1,0,0]
dy=[0,0,-1,1]


result=0

# DSF
def dsf(gragh,x,y):
    global result
    
    if gragh[x][y]==0:
        gragh[x][y]=True # 방문처리

        for k in range(4):
            nx=x+dx[k]
            ny=y+dy[k]

            if nx<0 or nx>n-1 or ny<0 or ny>m-1:
                continue

            if gragh[nx][ny]==0:
                dsf(gragh,nx,ny)  # 문제점 -> for문 처음 부터 시작, for문 ->함수 밖
            else:
                pass
    

# result 위치 어디하면 될까..?

for i in range(n):
    for j in range(m):
        if gragh[i][j]==0:
            result+=1
            dsf(gragh,i,j)

print(result) 

'''


'''
# Youtube 풀이

def dfs(x,y):
    if x<=-1 or x>=n or y<=-1 or y>=m:
        return False
    # 현재 노드를 아직 방문하지 않았다면
    if gragh[x][y]==0:
        # 방문처리
        gragh[x][y]=1
        # 상,하,좌,우의 위치들도 모두 재귀적으로 호출
        dfs(x-1,y)
        dfs(x,y-1)
        dfs(x+1,y)
        dfs(x,y+1)  # 나머지는 인접 노드 -> 방문 처리
        return True
    else:
        return False 
        

n, m = map(int,input().split())

gragh=[]
for i in range(n):
    gragh.append(list(map(int,input())))

result=0
for i in range(n):
    for j in range(m):
        if dfs(i,j)==True:
            result+=1 # 처음 방문 한 노드 -> result 1추가

print(result)

'''


# BFS 실전문제 _ 미로 탈출

'''

from collections import deque

n,m = map(int,input().split())
gragh=[]

for i in range(n):
    gragh.append(list(map(int,input())))

# 상하좌우
dx=[-1,1,0,0]
dy=[0,0,-1,1]


def bsf(x,y):
    queue=deque()
    queue.append((x,y)) # 튜플 형태로 좌표값 입력
    
    while queue:
        x,y=queue.popleft()
        
        for i in range(4):
            nx=x+dx[i]
            ny=y+dy[i]

            # 벽
            if nx<0 or nx>n-1 or ny<0 or ny>m-1:
                continue  # 재귀x -> return 반환 X / pass, continue O
            # 몬스터
            if gragh[nx][ny]==0:
                continue
            
            # 방문하지 않은 노드
            if gragh[nx][ny]==1:
                gragh[nx][ny]=gragh[x][y]+1
                queue.append((nx,ny))

    return gragh[n-1][m-1]

print(bsf(0,0))


'''

# 1303 전투

'''
import sys
sys.setrecursionlimit(10**6)

N,M=map(int,input().split())

gragh=[]
for i in range(N):
    gragh.append(list(input()))


dx=[-1,1,0,0]
dy=[0,0,-1,1]


def dsf(x,y,cnt,col):
    
    gragh[x][y]==0 # 방문처리

    for k in range(4):
        nx=x+dx[k]
        ny=y+dy[k]

        # 벽
        if 0<=nx<N and 0<=ny<M:
            if gragh[nx][ny]==col:
                cnt=dsf(nx,ny,cnt+1,col)
    return cnt
    

white=0
black=0

for i in range(N):
    for j in range(M):
        if gragh[i][j]=='W':
            white+=(dsf(i,j,1,'W'))**2
        elif gragh[i][j]=='B':
            black+=(dsf(i,j,1,'B'))**2
            
print(white,black)

# [참고]
# https://hseungyeon.tistory.com/232

'''


# 24463 미로

# bfs 구현 시도 -> 지나온 좌표를 저장해야할 방법 모르겠음,, dfs시도
'''
from collections import deque

N,M=map(int,input().split())
arr=[]
for _ in range(N):
    arr.append(list(map(str,input())))

v=[[False for _ in range(M)] for _ in range(N)]
print(v)
est=[]  # 시작점, 끝점 좌표 담을 변수
for i in range(N):
    for j in range(M):
        if i==0 or j==0 or i==N-1 or j==M-1:
            if arr[i][j]=='.':
                est.append((i,j))

ax,ay=est[0][0],est[0][1]
bx,by=est[1][0],est[1][1]

def bfs(x,y):
    q=deque()
    v[x][y]=True # 방문 표시
    q.append((x,y))
    
    dx=[-1,0,1,0]
    dy=[0,-1,0,1]
    
    while q:
        x,y=q.popleft()
        
        for k in range(4):
            nx=x+dx[k]
            ny=y+dy[k]
            
            if v[nx][ny]!=True and arr[nx][ny]=='.':
                q.append((nx,ny))
                
'''



# dfs 도전, 런타임에러

'''
N,M=map(int,input().split())
arr=[]
for _ in range(N):
    arr.append(list(map(str,input())))

v=[[False for _ in range(M)] for _ in range(N)]

est=[]  # 시작점, 끝점 좌표 담을 변수
for i in range(N):
    for j in range(M):
        if i==0 or j==0 or i==N-1 or j==M-1:
            if arr[i][j]=='.':
                est.append((i,j))

ax,ay=est[0][0],est[0][1]
bx,by=est[1][0],est[1][1]
dx = [-1, 0, 1, 0]
dy = [0, -1, 0, 1]
ans=[]

def dfs(x,y,path):
    if x==bx and y==by:
        path+=[(x,y)]
        ans.append(path)
        return

    for k in range(4):
        nx = x + dx[k]
        ny = y + dy[k]

        if 0<=nx<N and 0<=ny<M:
            if v[nx][ny] != True and arr[nx][ny] == '.':
                v[nx][ny]=True
                dfs(nx,ny,path+[(x,y)])

v[ax][ay]=True # 시작노드 방문처리
dfs(ax,ay,[(ax,by)])


for i in range(N):
    for j in range(M):
        if arr[i][j]=='.' and (i,j) not in ans[0]:
            arr[i][j]='@'

for i in range(N):
    print(''.join(arr[i]))

'''


# 7562 나이트의 이동

# dfs
# ?? What PBL?  >>  시간 초과 issue

'''
def dfs(i,j,cnt):
    global ans

    #print(i,j)


    # 도착점 도착
    if lst[i][j]==2:
        ans=min(ans,cnt)
        # print('check')
        return
    
    di = [-1, 1, 2, 2, 1, -1, -2, -2]
    dj = [2, 2, 1, -1, -2, -2, -1, 1]

    for k in range(8):
        ni,nj=i+di[k],j+dj[k]
        
        # 체스판 out
        if 0<=ni<N and 0<=nj<N:
            if visited[ni][nj]!=1: # 방문하지 않았다면
                visited[ni][nj]=1 # 방췍
                dfs(ni,nj,cnt+1) 
                visited[ni][nj]=0


# main
N=int(input())
lst=[[0 for _ in range(N)] for _ in range(N)] # 체스판
visited=[[0 for _ in range(N)] for _ in range(N)] # 방문체크 배열

si,sj=map(int,input().split())
ei,ej=map(int,input().split())

lst[ei][ej]=2 # 도착점

ans=1e9
cnt=0 
dfs(si,sj,cnt)

print(ans)

'''



# bfs
# python3 말고 pypy 돌려야 시초 안뜸 
# What PBL ..?  방췍 리스트 이슈

'''
bfs를 사용해야 하는 이유

- 4방향 탐색이 아닌 8방향 탐색 -> 시간 복잡도 훨씬 큼

- bfs로 한번 이동한 좌표 -> 현좌표에서 한번 이동했을 때 갈 수 있는 최선의 좌표가 보장 됨

'''

'''
from collections import deque
import sys
input = sys.stdin.readline

# main
TC=int(input())

for _ in range(TC):
    
    N=int(input()) # 체스판 가로 길이
    si,sj=map(int,input().split()) # 시작점
    ei,ej=map(int,input().split()) # 목표점

    lst=[[0 for _ in range(N)] for _ in range(N)] # 체스판
    visited=[[0 for _ in range(N)] for _ in range(N)] # 방췍 배열
    
    q=deque()
    q.append((si,sj))
    visited[si][sj]=True # 시작점 방췍

    # bfs
    while q:
        i,j=q.popleft()

        # 나이트 이동방향
        di = [-1, 1, 2, 2, 1, -1, -2, -2]
        dj = [2, 2, 1, -1, -2, -2, -1, 1]

        for k in range(8):
            ni,nj=i+di[k],j+dj[k]

            if 0<=ni<N and 0<=nj<N:
                if visited[ni][nj]!=True:
                    q.append((ni,nj))
                    visited[ni][nj]=True
                    lst[ni][nj]=lst[i][j]+1

    # print(lst)
    print(lst[ei][ej])

'''


# BFS

# 개선
# 1. while문에서 목표점 도달 시 탈출  
# 2. 방문 체크 안사용해도 됨 -> BFS의 한번 이동은 최적해가 보장 되기 때문에 방문한 좌표에 대해서 더 이상 갱신이 일어나지 않음 -> 원본 테이블만 활용해서 사용하자

# Python3 통과

'''
from collections import deque
import sys
input = sys.stdin.readline


# bfs
def bfs():

    q=deque()
    q.append((si,sj))
    lst[si][sj]=1 # 시작점 방췍

    # bfs
    while q:
        i,j=q.popleft()

        if i==ei and j==ej:
            return lst[i][j]-1  # 처음 좌표(이동안했기 때문에 원래 0인데 방문처리 때문에 1로 시작함)를 1로 넣었으므로 이동횟수 1감소


        # 나이트 이동방향
        di = [-1, 1, 2, 2, 1, -1, -2, -2]
        dj = [2, 2, 1, -1, -2, -2, -1, 1]

        for k in range(8):
            ni,nj=i+di[k],j+dj[k]

            if 0<=ni<N and 0<=nj<N:
                if lst[ni][nj]==0:
                    q.append((ni,nj))
                    lst[ni][nj]=lst[i][j]+1

    return lst[ei][ej]


# main
TC=int(input())

for _ in range(TC):
    
    N=int(input()) # 체스판 가로 길이
    si,sj=map(int,input().split()) # 시작점
    ei,ej=map(int,input().split()) # 목표점
    lst=[[0 for _ in range(N)] for _ in range(N)] # 체스판
    ans=bfs()  # bfs 실행
    
    print(ans) 

'''






# 13023 ABCDE

'''
# DFS
def dfs(nod,lst):
    global ans

    # print(lst)

    if len(lst)==5:    
        ans=True
        return
    
    for nex_nod in graph[nod]:
        if visited[nex_nod] == False:
            visited[nex_nod]=True
            dfs(nex_nod,lst+[nex_nod])
            visited[nex_nod]=False


# Main
N,M=map(int,input().split())
graph=[[] for _ in range(N+1)]
# visited=[False for _ in range(N+1)]
ans=False

for _ in range(M):
    nod1,nod2=map(int,input().split())
    graph[nod1].append(nod2)

# print(graph)

for nod in range(N+1):

    dfs(nod,[nod])


if ans:
    print(1)
else:
    print(0)

'''


# 사이클 형성 안하는 원소 개수 세기
# BFS + DFS 느낌 ?
# MST ? 


# 두번째 try
# 서로소 집합 알고리즘


'''
# find 함수
def find_parent(parent, x):
    # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출 -> 루트 노드는 자신의 번호를 초기화 과정에서 자신의 번호를 가지고 있음
    if parent[x]!=x:
        return find_parent(parent, parent[x])
    return x

# union 함수
def union_parent(parent, a, b):
    a_root=find_parent(parent, a) # a 루트 노드 탐색
    b_root=find_parent(parent, b) # b 루트 노드 탐색

    
    # 이 부분이 가장 헷갈렸음 #####
    # 연결된 A,B의 합이 아니라 A와 B의 루트노드에 관해 합해줌
		# union2,4 단계 그림 참고
    if a_root<b_root:
        parent[b_root]=a_root # B'가 A'를 향하게 함 (B'->A')
    else:
        parent[a_root]=b_root


# v:루트노드, e:간선개수
v,e=map(int,input().split())
parent=[0]*(v+1) # 부모 테이블 초기화

# 부모 테이블에서 부모를 자기 자신으로 초기화 -> V개의 트리 형태
for i in range(1,v+1):
    parent[i]=i

# union 연산 모두 수행
for i in range(e):
    a,b=map(int,input().split())
    union_parent(parent,a,b)


for i in range(len(parent)):
    if parent.count(i)>=5:
        ans=1
        break

print(ans)


# 각 원소가 속한 집합 출력 -> 루트노드 출력 -> 루트 노드가 같다면 같은 집합, 다르다면 다른 집합
print('각 원소가 속한 집합(루트노드): ', end='')
for i in range(1,v+1):
    print(find_parent(parent,i), end=' ')


# 각각의 부모 노드 출력 -> parent 테이블 = 부모 노드 정보 테이블
print('부모 테이블: ', end='')
for i in range(1, v+1):
    print(parent[i],end=' ')
'''





# 13023 BOJ ABCDE

# 문제 설명 더러움 
# 결론: 한붓그리기 5개 노드 okay면 True 출력하기

'''
# 1. 그래프 초기화

# DFS
def dfs(nod,lst):
    global ans

    if len(lst)==5:    
        ans=True
        return
    
    for nex_nod in graph[nod]:
        if visited[nex_nod] == False:
            visited[nex_nod]=True
            dfs(nex_nod,lst+[nex_nod])
            visited[nex_nod]=False


# Main
N,M=map(int,input().split())
graph=[[] for _ in range(N+1)]
visited=[False for _ in range(N+1)]
ans=False

for _ in range(M):
    nod1,nod2=map(int,input().split())

    # 양방향 간선
    graph[nod1].append(nod2)
    graph[nod2].append(nod1)

for nod in range(N+1):
    visited[nod]=True
    dfs(nod,[nod])
    visited[nod]=False
    
    # magic code -> 안해주면 시간초과 나는 마술 
    # 순차적으로 노드를 탐색하면서 True값을 뽑았다면, 이후 노드들에 대해 탐색 x
    if ans:
        break

if ans:
    print(1)
else:
    print(0)



# 2. defaultdic

from collections import defaultdict

# DFS
def dfs(nod,lst):
    global ans

    if len(lst)==5:    
        ans=True
        return
    
    for nex_nod in dic[nod]:
        if visited[nex_nod] == False:
            visited[nex_nod]=True
            dfs(nex_nod,lst+[nex_nod])
            visited[nex_nod]=False


# MAIN
N,M=map(int,input().split())
dic=defaultdict(list)
visited=[False for _ in range(N+1)]
ans=False

# 그래프 형성
for _ in range(M):
    n1,n2=map(int,input().split())
    dic[n1].append(n2)
    dic[n2].append(n1)

# 각 지점에서 DFS 수행
for nod in range(N+1):
    visited[nod]=True
    dfs(nod,[nod])
    visited[nod]=False

    # magic method
    if ans:
        break

if ans:
    print(1)
else:
    print(0)

# 1, 2 시간 비슷

'''


# BOJ_2667_단지 번호 붙이기

'''
풀이법

1. DFS로 인접한 아파트를 묶어주고 번호로(apt_num) 그룹화 한다.

2. 단지 개수를 별도의 리스트(ans_lst)에 담고 그룹화 한다.

3. 출력한다.


# DFS
# i,j: 좌표 / cnt: 가구 개수
def dfs(i,j):

    for di,dj in [(-1,0),(0,-1),(1,0),(0,1)]:
        ni,nj=i+di,j+dj

        # 범위 안
        if 0<=ni<len(lst) and 0<=nj<len(lst[0]):
            if lst[ni][nj]==1:
                lst[ni][nj]=apt_num
                dfs(ni,nj)

# Main
N=int(input())
lst=[list(map(int,input())) for _ in range(N)]
ans_lst=[]

apt_num=2 # 아파트 단지 번호, 기본 값 1 방문 처리 회피
for i in range(len(lst)):
    for j in range(len(lst[0])):
        if lst[i][j]==1:
            lst[i][j]=apt_num
            dfs(i,j) # 모두 방문처리 해줌
            apt_num+=1


# masic method : sum
# https://wellsw.tistory.com/210 : 2 mtx -> 1 mtx

lst_flat = list(sum(lst, []))
for apt_num in range(2,apt_num):
    ans_lst.append(lst_flat.count(apt_num))

print(len(ans_lst))
for num in sorted(ans_lst):
    print(num)

'''

            

# 16929_BJ_Two Dots

'''
# DFS
def dfs(i,j,target_x,target_y,dist):
    global ans 

    # 이유3 : 사기 코드
    if ans:
        return 

    if i==target_x and j==target_y and dist>=3: # 시작점과 동일하고, 거리가 3이상이라면 
        ans=True
        return

    for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni,nj=i+di,j+dj

        # 이동 기준
        # 1. 시작점과 같은 단어야함
        # 2. 방문처리가 되지 않았고, 같은 단어인 경우 
        # 3. 방문처리가 되었지만, 거리가 3이상인 같은 단어인 경우 (즉, 사이클을 형성한 후의 출발점이라면)
        
        if 0<=ni<N and 0<=nj<M: # 범위 조건 만족 
            if lst[ni][nj]==lst[target_x][target_y]: # 1

                # 이유1: 인덱스 조건을 포함 안시켜주면, 방문처리된 3이상 이동거리 좌표 전부 방문 -> 메모라 초과
                # 이유2(오답으로 이어짐, 시간초과 X): 방문 해제를 안해주면, 시작점이 다른 경우에 대한 처리가 안됨 testcase3
                if (visited[ni][nj]==True and dist>=3 and ni==target_x and nj==target_y) or (visited[ni][nj]==False):  # 2, 3
                    visited[ni][nj]=True # 방문처리
                    dfs(ni,nj,target_x,target_y,dist+1)
                    visited[ni][nj]=False


# Main 
N,M=map(int,input().split())
lst=[list(input()) for _ in range(N)]
visited=[[False for _ in range(M)] for _ in range(N)]
ans=False


for i in range(len(lst)):
    for j in range(len(lst[0])):
        si,sj=i,j  # 시작 좌표 저장
        visited[i][j]=True
        dfs(i,j,si,sj,0)
        visited[i][j]=False

print("Yes" if ans else "No")




    '''

'''

** 풀이법 **

문제: 도시의 거리가 최소가 되게 만들어라 / 도시의 거리: 모든 가구에서 가장 가까운 치킨 집까지의 거리 총 합

핵심: 폐업을 시키지 않을 치킨집은 어떤 치킨집일까?


Case 1. 치킨 집을 기준으로 가구를 바라본다.

- 치킨집 한 곳과 모든 가구의 거리를 비교할 수는 없음 / 모든 가구가 그 해당 치킨집과 가까운 것은 아니기 때문에

>> 그렇다면, 어떻게 함 ,,?



Case 2. 모든 가구에서 모든 치킨집을 바라본다. 

1. 각 가구에서 모든 치킨 집까지의 거리 정보를 저장함 
2. 정해진 개수 만큼, 조합으로 폐업할 가구 완전탐색 -> 각 경우에서의 도시의 합 구해서 최솟값 반환 

>> 시간초과 날려나 ,,?


[필요정보]
- 치킨집 위치 
- 가구 위치
- 가구당 모든 치킨집 거리 저장할 배열


폐업하지 않을 치킨집을 선택  -> DFS 종료조건

'''

# BJ_15868_치킨 배달

'''
M,N=map(int,input().split())
lst=[list(map(int,input().split())) for _ in range(M)]

# 치킨집 거리 정보 
ch_lst=[]
home_num=0 # 가구 수
ans=1e9 # 정답

for i in range(len(lst)):
    for j in range(len(lst[0])):
        if lst[i][j]==2:
            ch_lst.append((i,j))
        
        # 가구면 가구 숫자 +1
        if lst[i][j]==1:
            home_num+=1



# 가구-치킨 거리정보 테이블
dist_lst=[[] for _ in range(home_num)]


# 거리 계산
row=0
for i in range(len(lst)):
    for j in range(len(lst[0])):
        if lst[i][j]==1:
            for k in range(len(ch_lst)):
                ch_i,ch_j=ch_lst[k][0],ch_lst[k][1]
                dist=abs(i-ch_i)+abs(j-ch_j)
                dist_lst[row].append(dist)
            row+=1
                
# print('집 to 치킨집 거리', dist_lst)

############ 여기까지 ok


# DFS
def dfs(check,idx):
    global ans
     
    # 종료조건 1 / N만큼 가게 선택
    if len(check)==N:
        
        ### 완탐으로 어떤거 삭제할지 구현
        ### 제외한 컬럼 빼고 최단 거리구하는 코드 작성
        total_dist=0

        for i in range(len(dist_lst)):
            dist=1e9
            for j in range(len(dist_lst[0])): 
                if j in check:
                    dist=min(dist,dist_lst[i][j])
            
            total_dist+=dist

        # print('#########',check)
        # print(total_dist)

        ans=min(ans,total_dist)
        # print(check)
        return

    # 종료조건 2 / 마지막 인덱스 
    if idx==len(dist_lst[0]):
        return

    dfs(check,idx+1)
    dfs(check+[idx],idx+1)


dfs([],0)
print(ans)

'''