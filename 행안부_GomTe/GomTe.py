# Greedy

# 1. 거스름돈

'''
N=int(input())

coin_lst=[500,100,50,10]
ans=0

for coin in coin_lst:
    if N//coin>=1:
        ans+=(N//coin)
        N%=coin
        print(coin, ans, N)
    

print(ans)

'''


# 2. 큰수의 법칙

'''
n,m,k=map(int,input().split())
arr=list(map(int,input().split()))

mok=m//k
na=m%k

arr.sort(reverse=True)
ans=(arr[0]*mok*k)+(arr[1]*na)

print(ans)

'''


# 곱하기 혹은 더하기

'''
arr=list(map(int,input()))

ans=0
for i in range(len(arr)):
    if arr[i]<=1 or ans<=1:
        ans+=arr[i]
    else:
        ans*=arr[i]
    
    print(arr[i],ans)

print(ans)

'''

# 모험가 길드

'''
길드 정렬

차례로 접근 -> 공포도 확인 -> 해당 공포도의 인원만큼 패스

'''

'''
N=int(input())
arr=list(map(int,input().split()))

arr.sort()
flag=True # 그룹 조건(True: 새 그룹 / False: 그룹 만드는 과정)
ans=0 # 총 그룹 수 
num=0 # 각 그룹의 인원수 


for i in range(len(arr)):
    if flag==True: # 그룹 만들 수 있으면
        flag=False  # 그룹 만든 과정으로 상태 변환
        goal=arr[i] # 그룹 형성을 위한 공포도 조건 받고,
        num+=1 # 인원 한명 추가
        
        
    else:
        num+=1 # 그룹 만드는 과정이라면 한명 추가
        goal=arr[i]
    
    if  num>=goal: # 그룹이 형성되었다면
        flag=True # 새 그룹 가능으로 조건 변환
        ans+=1 # 현재 그룹 개수 추가
        num=0
    

print(ans)

'''



# AVATA

# 상하좌우

'''
N=int(input()) # 지도 크기

lst=list(input().split())  # 이동 방향

dir={'R':(0,1),'L':(0,-1),'U':(-1,0),'D':(1,0)} 

x,y=0,0  # 시작점 

for d in lst:
    dx,dy=dir[d]

    nx=x+dx
    ny=y+dy

    if (0<=nx<N) and (0<=ny<N):
        x,y=nx,ny

ans_x,ans_y=x+1,y+1
# print(ans_x,ans_y)

'''

# 시각

'''
3이 총 몇번 들어갔는지 확인하기

- 3이 하나라도 포함되어 있으면 카운트, 하나라도 없으면 노카운트

00시 00분 00초 ~ N시 59분 59초

'''

'''
H=int(input())
cnt=0

for i in range(H+1):
    for j in range(60):
        for k in range(60):
            if ('3' in str(i)) or ('3' in str(j)) or ('3' in str(k)):
                cnt+=1


print(cnt)

'''


# 왕실의 나이트

'''
idx=input()
lst=list(idx)

row_idx={'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8}

x=row_idx[lst[0]]
y=int(lst[1])


dxy=[(-2,1),(-2,-1),(2,1),(2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]

ans=0
for dir in dxy:
    dx,dy=dir[0],dir[1]
    nx,ny=x+dx,y+dy

    if (1<=nx<9) and (1<=ny<9):
        ans+=1


print(ans)

    
'''


# 문자열 재정렬

'''
lst=list(input())
lst.sort()

print(lst)

num_lst=[]
ch_lst=[]

for i in range(len(lst)):
    if ord('1')<=ord(lst[i])<=ord('9'):
        num_lst.append(lst[i])
    else:
        ch_lst.append(lst[i])


num_ans=sum(list(map(int,num_lst)))
ch_ans=''.join(ch_lst)

ans=ch_ans+str(num_ans)

print(ans)

'''


# BFS, DFS

# 음료수 얼려먹기

'''
N,M=map(int,input().split())
arr=[] # 얼음판 
cnt=0 # 정답 

# 얼을판 
for _ in range(N):
    line=list(map(int,list(input())))
    arr.append(line)


# 방문체크
visited=[[False] * M for _ in range(N)]


# dfs 함수
def dfs(i,j,visited):
    visited[i][j]=True
    dij=[(1,0),(0,1),(-1,0),(0,-1)] # 방향 함수
    
    for di,dj in dij:
        ni=i+di
        nj=j+dj

        if 0<=ni<N and 0<=nj<M and arr[ni][nj]==0 and visited[ni][nj]!=True:
            dfs(ni,nj,visited)

    return 


# main
for i in range(N):
    for j in range(M):
        if arr[i][j]==0 and visited[i][j]!=True:
            visited[i][j]=True # 방문 체크
            dfs(i,j,visited)
            cnt+=1


print(cnt)
            
            



from collections import deque


# 미로 탈출

N,M=map(int,input().split())
arr=[]

# 미로 입력
for _ in range(N):
    line=list(map(int,input()))
    arr.append(line)


# 방문 체크 배열
visited=[[False]*M for _ in range(N)]
dij=[(-1,0),(1,0),(0,-1),(0,1)]

# bfs 함수 
def bfs(i,j):
    q=deque()
    visited[i][j]=True # 방문 체크
    q.append((i,j))

    while q:
        i,j=q.popleft() # 가장 처음 들어간 노드 빼기
        for di,dj in dij:
            ni,nj=i+di,j+dj
            if 0<=ni<N and 0<=nj<M and arr[ni][nj]==1 and visited[ni][nj]!=True:
                arr[ni][nj]+=arr[i][j]  # 거리 누적 
                q.append((ni,nj))  # 큐에 담기 
            


# bfs 탐색
bfs(0,0)

print(arr[N-1][M-1])


'''

# 정렬

# 1. 선택정렬

'''
arr=[7,5,9,0,3,1,6,2,8,]

for i in range(len(arr)):
    min_num=arr[i]
    idx=i # i 이후 가장 작은 수의 인덱스
    for j in range(i+1,len(arr)):
        if arr[j]<min_num:
            min_num=arr[j]
            idx=j
    
    arr[i],arr[idx]=arr[idx],arr[i]

print(arr)

'''

# 삽입정렬

# 작은 순으로 정렬
'''
arr=[7,5,9,0,3,1,6,2,4,8]


for i in range(1,len(arr)):
    for j in range(i,0,-1):
        if arr[j]<arr[j-1]:
            arr[j],arr[j-1]=arr[j-1],arr[j]
        else:
            break

print(arr)

'''

# 큰 순으로 정렬

'''
arr=[7,5,9,0,3,1,6,2,4,8]

for i in range(1,len(arr)):
    for j in range(i,0,-1):
        if arr[j]>arr[j-1]:
            arr[j],arr[j-1]=arr[j-1],arr[j]
        else:
            break

print(arr)

'''

#  왼쪽이 고정되어 있다고 생각하고, 오름차순 정렬

'''
arr=[7,5,9,0,3,1,6,2,4,8]

for i in range(len(arr)-1,-1,-1):
    for j in range(i,len(arr)-1):
        if arr[j]>arr[j+1]:
            arr[j],arr[j+1]=arr[j+1],arr[j]
        else:
            break

print(arr)

'''


'''

1. 순차적으로 피봇(pivot)을 설정한다.

2. 왼쪽에서는 피봇보다 큰수, 오른쪽에서부터 피봇보다 작은 수를 선택한다.

-> 선택되면 바꾼다.

-> 왼쪽을 가르키는 인덱스가, 오른쪽을 가르키는 인덱스와 엇갈리면, 피봇과 왼쪽을 가르키는 인덱스를 바꾼다.


3. 분할된 묶음을 재귀로 같은 과정을 반복한다.



# 재귀를 어떻게 써야할까 ,,?  dfs로 간다면 종료조건은 어떻게 걸어야하지 ,,?

# 정렬할 배열을 전역 변수로 접근해야겠지 ,,? 

'''


# 퀵 소트 정렬 -> 1차 실패 


arr=[1,4,2,0,3,5,6,9,7,8]
flog=False # 스왑여부

pivot_idx=0 # 우선 가장 첫번째 인덱스를 피봇으로 잡음 

for i in range(len(arr)): # 왼쪽 인덱스
    for j in range(len(arr),-1,-1):  # 오른쪽 인덱스
        if i>j: # 좌, 우 인덱스가 엇갈린다면
            arr[pivot_idx],arr[i]=arr[i],arr[pivot_idx] # 피봇과 좌 인덱스 스왑 


def dfs(arr):
    pivot=arr[0]
    
    for i in range(len(arr)):
        for j in range(len(arr),-1,-1):
            if i>j:
                arr[pivot_idx],arr[i]=arr[i],arr[pivot_idx]
                flag=True # 스왑 체크
                right_arr=arr[0:i]
                left_arr=arr[len(arr):i:-1]
                dfs(right_arr)
                dfs(left_arr)
                break
        if flag: # 만약 스왑이 되었다면, 바로 빠져나가고 아니면, 다음으로 이동  

            break


# 수정해야함: 현재는 왼쪽이 한칸 갈 때, 오른쪽을 전부 보냄 -> 왼쪽에서 피봇보다 큰 것을 찾고, 오른쪽에서 피봇보다 작은 것을 찾아서 스왑해야함 


# arr=[1,2,3,4]

# print(arr[-1,1,-1])