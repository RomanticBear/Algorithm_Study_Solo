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

'''
# 각 원소가 속한 집합 출력 -> 루트노드 출력 -> 루트 노드가 같다면 같은 집합, 다르다면 다른 집합
print('각 원소가 속한 집합(루트노드): ', end='')
for i in range(1,v+1):
    print(find_parent(parent,i), end=' ')


# 각각의 부모 노드 출력 -> parent 테이블 = 부모 노드 정보 테이블
print('부모 테이블: ', end='')
for i in range(1, v+1):
    print(parent[i],end=' ')
'''

for i in range(len(parent)):
    if parent.count(i)>=5:
        ans=1
        break

print(ans)





# 한붓그리기네 ,, 
#  https://sonsh0824.tistory.com/entry/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EA%B3%B5%EB%B6%804-%ED%95%9C%EB%B6%93%EA%B7%B8%EB%A6%AC%EA%B8%B0Eulerian-circuit