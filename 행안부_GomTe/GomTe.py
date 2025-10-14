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


# 퀵 소트 정렬 -> 2차 실패 
# for문 접근 -> i가 j랑 스왑하고, 1증가하면서 나머지 작은 수에 대해서 스왑이 불가 / i도 j 처럼 처음부터 접근할시 결국 O(N^2) 형태 -> while 문으로 가는게 맞음

'''
arr=[7,5,9,0,3,1,6,2,4,8]

def dfs(arr):        
    pivot_idx=0 # 우선 가장 첫번째 인덱스를 피봇으로 잡음 

    for i in range(pivot_idx+1, len(arr)): # 왼쪽 인덱스
        for j in range(len(arr)-1,pivot_idx,-1):  # 오른쪽 인덱스

            if ((arr[pivot_idx]<arr[i]) and (arr[pivot_idx]>arr[j]) and (i<j)):
                arr[i],arr[j]=arr[j],arr[i] # 피봇과 좌 인덱스 스왑
                break
                
            if i>j:
                arr[pivot_idx],arr[j]=arr[j],arr[pivot_idx]
                right_arr=arr[0:j]
                left_arr=arr[j+1:len(arr)-1]
                dfs(right_arr)
                dfs(left_arr)
            


dfs(arr)

print(arr)


'''



# while문 도전 퀵 정렬
'''
arr=[7,5,9,0,3,1,6,2,4,8]


def quick_sort(arr):


    if len(arr)<=1:
        return arr

    pivot=0
    l_idx=1
    r_idx=len(arr)-1

    while True:

        while l_idx <= r_idx and arr[l_idx] < arr[pivot]:
            l_idx += 1
        # 오른쪽에서 피벗보다 작은 값 찾기
        while l_idx <= r_idx and arr[r_idx] > arr[pivot]:
            r_idx -= 1

        # 교차 해버렸을 때, 탈출 
        if l_idx > r_idx:
            break

        # 교차 전, 스왑 조건 만족 시
        arr[r_idx],arr[l_idx]=arr[l_idx],arr[r_idx]
        l_idx+=1
        r_idx-=1


    # 교차 후, r_idx와 pivot 스왑 
    arr[pivot], arr[r_idx]=arr[r_idx],arr[pivot]

    return quick_sort(arr[:r_idx])+[arr[r_idx]]+quick_sort(arr[r_idx+1:])


print(quick_sort(arr))

'''

# 1316 그룹 단어 체커

'''
def check_word(lst):

    flag=True
    stack=[]

    for word in lst:
        if word not in stack:
            stack.append(word)
        else:
            if word != stack[-1]:
                flag=False
                break
    
    return flag


N=int(input())
cnt=0
for _ in range(N):
    lst=list(input())
    cnt+=check_word(lst)

print(cnt)


'''

# 2941 크로아티아 알파벳


# dz=, dz 의 경우 count 함수 접근 불가 -> dz 2개로 카운트 됨 

'''

cro_lst=['c=','c-','dz=','d-','lj','nj','s=','z=']

str=input()

cnt=0 # cro_lst 단어 개수 
word_sum=0 # cro_lst 글자 개수 


for word in cro_lst:

    if word=='z=':
        if 'dz=' in str:
            cnt+=(str.count('z=')-str.count('dz='))
            word_sum+=(str.count('z=')-str.count('dz='))*len(word)
            continue  # 'dz='이 없다면 아래로 처리     

    cnt+=str.count(word)
    word_sum+=len(word)*str.count(word)

left_word=len(str)-word_sum


print(cnt+left_word)


'''

# 2960 에라토스테네스의 체

'''
N,K=map(int,input().split())

lst=[i for i in range(2,N+1)]

flag=False # 이중 for문 탈출 
ans=0
check_lst=[]

for i in range(len(lst)):
    if lst[i] not in check_lst and K!=0:
        check_lst.append(lst[i])
        K-=1

        # print('#',lst[i],check_lst,K)
        if K==0:
            ans=lst[i]
            break

        for j in range(i+1,len(lst)):
            if lst[j]%lst[i]==0:
                if lst[j] not in check_lst:
                    check_lst.append(lst[j])
                    K-=1
                if K==0:
                    ans=lst[j]
                    flag=True
                    break
            
            # print(lst[i],lst[j],check_lst,K)

    if flag:
        break
    
    # print(lst[i],lst,K)

print(ans)

'''

# 2167 2차원 배열의 합 -> 시간 초과


'''
import sys
input = sys.stdin.readline

def sq_sum(lst,i,j,x,y):
    
    sq_sum=0
    r_len=x-i
    c_len=y-j

    for row in range(i,i+r_len+1):
        for col in range(j,j+c_len+1):           
            sq_sum+=lst[row][col]
    return sq_sum


lst=[]
N,M=map(int,input().split())

for _ in range(N):
    lst.append(list(map(int,input().split())))

K=int(input())
ans=0
for _ in range(K):
    i,j,x,y=map(int,input().split())
    print(sq_sum(lst,i-1,j-1,x-1,y-1))


'''

# 1543 문서 검색

'''

str=input()
sub_str=input()

cur_idx=0
cnt=0

while True:
    if cur_idx+len(sub_str)>len(str):
        break

    # 여부 확인
    
    cur_str=str[cur_idx:cur_idx+len(sub_str)]

    if cur_str==sub_str:
        cnt+=1
        cur_idx+=len(sub_str)

    else:
        cur_idx+=1

print(cnt)

'''


# 1459 걷기

# 좌표나 시간에 따른 기준으로 분기하기에 복잡 
'''
X,Y,W,S=map(int,input().split()) # W: 직선 시간 / S: 대각선 시간
ans=0

if W>=S*2:
    ans=(X+Y)*S
else:
    L_D,S_D=max(X,Y),min(X,Y)

    if (X+Y)%2==0:
        # CASE1: 대각선 이동 
        CASE1=L_D*W

        # CASE2: 대각선 + 직선 이동  
        CASE2=S_D*W+(L_D-S_D)*S
        ans=min(CASE1,CASE2)

    else:
        pass
print(ans)

'''

# 경로에 따라 분기하자. 

# 1. 직선으로만 조진다.

# 2. 대각선으로만 조진다.
# 2-1. X+Y가 짝수라면 대각선으로만 조진다.
# 2-2. X+Y가 홀수라면 한번 직선으로 가고, 대각선으로 조진다. (최대한 대각선으로 조지는 방법)

# 앞의 방법에서 고려하지 못한 문제
# 3. 대각선으로 초기 이동하되, 지나치지는 말고, 나머지는 직선으로 가자.
# 왜냐하면, 대각선으로만 조지는 케이스는 다른 경우의 수로 계산함

'''
X,Y,S,W=map(int,input().split()) # S: 직선 시간 / W: 대각선 시간

DIST1=(X+Y)*S

if (X+Y)%2==0:
    DIST2=max(X,Y)*W
else:
    DIST2=(max(X,Y)-1)*W+S

DIST3=min(X,Y)*W+(abs(X-Y)*S)

print(min(DIST1,DIST2,DIST3))

'''


# 15649 N과 M(1)

# 실패 -> 중복을 형성하지 않고, 뒤로 조합만 구성하는 형태 / 오름차순만 가능하고 (2,1), (3,1) 와 같은 형태 불가 / 백트랙킹 x 

"""
N,M=map(int,input().split())

lst=[i for i in range(1,N+1)]
cur_idx=0

def back(cur_idx,sub_lst):

    if len(sub_lst)==M:
        print(*sub_lst)
        return 

    else:
        if cur_idx>=len(lst):
            return
        else:
            back(cur_idx+1,sub_lst+[lst[cur_idx]])
            back(cur_idx+1,sub_lst+[])


back(0,[])

"""

# 2차

'''
N,M=map(int,input().split())
lst=[]
v=[True]*(N+1)

def dfs(sub_lst):
    if len(sub_lst)==M:
        lst.append(sub_lst[:])
        return
    
    for num in range(1,N+1):
        if v[num]:
            v[num]=False
            sub_lst.append(num)
            dfs(sub_lst)
            v[num]=True
            sub_lst.pop()

dfs([])

for seq in lst:
    print(*seq)

'''


# 15650 N과 M(2)

# 방법 1
'''
N, M=map(int,input().split())

def dfs(idx,sub_lst):
    if len(sub_lst)==M:
        print(*sub_lst)
        return

    for num in range(idx,N+1):
        dfs(num+1,sub_lst+[num])

dfs(1,[])
'''


# 방법 2
'''
N, M=map(int,input().split())

def dfs(num,sub_lst):
    if num>N:    
        if len(sub_lst)==M:
            print(*sub_lst)
        
        return 
    
    dfs(num+1,sub_lst+[num])
    dfs(num+1,sub_lst)

dfs(1,[])

'''

# 15651 N과 M (4)
'''
N,M=map(int,input().split())

def dfs(idx,sub_lst):
    
    if len(sub_lst)==M:
        print(*sub_lst)
        return

    for num in range(idx,N+1):
        dfs(num,sub_lst+[num])

dfs(1,[])

'''


# 15655 N과 M (6)

'''
N,M=map(int,input().split())
lst=sorted(list(map(int,input().split())))
v=[True]*(N+1)


def dfs(sub_lst):

    if len(sub_lst)==M:
        print(*sub_lst)
        return
    
    for num in range(len(lst)):
        if v[num]:
            v[num]=False
            dfs(sub_lst+[lst[num]])
            v[num]=True

dfs([])

'''

# 15655 N과 M (6)
'''
N,M=map(int,input().split())
lst=sorted(list(map(int,input().split())))

def dfs(idx,sub_lst):

    if idx>=N:
        if len(sub_lst)==M:
            print(*sub_lst)
        return 
        
    dfs(idx+1,sub_lst+[lst[idx]])
    dfs(idx+1,sub_lst)

dfs(0,[])
'''




# 14502 연구소


'''
함수 1. 백트랙킹으로 벽 세우기

함수 2. DFS로 바이러스 확산, 안전영역 개수 카운트


[발생한 문제점]

1. arr을 모든 dfs 함수에서 공유하면서, 감염 시 문제 발생 
>> 벽을 3개 세웠을 때, 깊은 복사로 복제본을 만들고 확산시키자.

2. DFS로 벽을 세울 시, 인접한 4방향으로 접근하면 인접칸에 대해서만 벽이 세워짐 == 떨어진 빈 칸에는 벽을 절대 세울 수 없음.
>> 해결하지 못함 


'''


N,M=map(int,input().split())

arr=[] # 연구소
ans=-1e9 # 최소 안전 영역 

for _ in range(N):
    arr.append(list(map(int,input().split())))


# 함수 1. 벽 세우기
def wall_dfs(arr,i,j,cnt):

    global ans

    # 벽 다세웠다면,
    if cnt==0:

        # 복제본 만들기
        copy=[row[:] for row in arr]

        # 바이러스 확산
        for i in range(N):
            for j in range(M):
                if copy[i][j]==2:
                    virus_dfs(copy,i,j)


        # *** 궁금증 *** virus_dfs 과정을 거친 매트릭스가 반영되나? ㅇㅇ 

        # 안전영역 개수 확인, 최적값 갱신  
        safe_cnt=sum(row.count(0) for row in copy)
        ans=max(ans,safe_cnt) 
        return


    # 벽 세우기
    for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni,nj=i+di,j+dj

        if 0<=ni<N and 0<=nj<M:
            if arr[ni][nj]==0:
                arr[ni][nj]=1   # 벽 설치
                wall_dfs(arr,ni,nj,cnt-1) #  설치 횟수 감소 시키고, dfs 호출
                arr[ni][nj]=0   # 벽 제거    


# 함수 2. 바이러스 확산 
def virus_dfs(arr,i,j):
    for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni,nj=i+di,j+dj

        if 0<=ni<N and 0<=nj<M:
            if arr[ni][nj]==0:
                arr[ni][nj]=2
                virus_dfs(arr,ni,nj)


# main 문 
for i in range(N):
    for j in range(M):
        if arr[i][j]==0:
            wall_dfs(0,arr,i,j,3)


print(ans)

