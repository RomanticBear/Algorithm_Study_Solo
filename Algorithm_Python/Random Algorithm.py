# 그리디 
# 11047 동전0

'''
import sys
input=sys.stdin.readline

n,k=map(int,input().split())
lst=[]

for i in range(n):
    num=int(input())
    lst.append(num)

lst.reverse()

cnt=0

for money in lst:
    if money<=k:
        cnt+=k//money
        k%=money

        # print(k, money)
    
        if k==0:
            break
    else:
        pass

print(cnt)

'''


# 그리디
# 카드 정렬하기 1715

'''
import sys
import heapq
input=sys.stdin.readline

n=int(input())
q=[]

for _ in range(n):
    num=int(input())
    heapq.heappush(q,num)

# 비교 횟수 저장 리스트
cmp=[]

while q:

    # 두 덱을 비교한 횟수
    sum=0
    
    if len(q)==1:
        break
    else:
        item1=heapq.heappop(q)
        item2=heapq.heappop(q)
    
        sum+=item1+item2
        # print('it1:',item1,' item2:',item2,' sum:',sum)
        cmp.append(sum)
        heapq.heappush(q,sum)



result=0


for cnt in cmp:
    result+=cnt

print(result)

'''


# 그리디
# 1744 수 묶기

'''
풀이법

음수+0, 양수일 때 나누기

1. 음수+0

개수가 짝수 -> 절댓값이 큰 것부터 짝지어서 두개씩 곱함(마지막이 0이라면 곱해서 0)

개수가 홀수 -> 절댓값이 큰 것 부터 짝지어서 두개씩 곱하고 남은거 더해줌


2. 양수

개수가 짝수 -> 큰 것부터 짝지어서 곱함
개수가 홀수 -> 큰 것부터 짝지어서 곱하고 남은 하나 더함

** 짝수의 상황에서 1의 경우: + 가 더 큰 계산값 도출




개선점

1일 때를 따로 나누어주어 3분할해서 접근하면 더 코드가 깔끔

'''

'''
import sys
input=sys.stdin.readline


n=int(input())
lst=[]
neg_lst=[]
pos_lst=[]


for _ in range(n):
    lst.append(int(input()))


for i in range(len(lst)):
    if lst[i]<=0:
        neg_lst.append(lst[i])
    else:
        pos_lst.append(lst[i])


neg_lst.sort()  # 오름차순 정렬 -> 절댓값이 큰 순서
neg_sum=0
sub_sum=0
nL=len(neg_lst)

if nL%2 != 0: # 홀수  
    for i in range(nL-1):
        if (i+1)%2 != 0:
            sub_sum=neg_lst[i]

        else:
            sub_sum*=neg_lst[i]
            neg_sum+=sub_sum
 
            
    neg_sum+=neg_lst[nL-1]


else:
    for i in range(nL):
        if (i+1)%2 != 0:
            sub_sum=neg_lst[i]

        else:
            sub_sum*=neg_lst[i]
            neg_sum+=sub_sum



# 짝수의 경우 -> 1은 더해주는게 유리  

pos_lst=sorted(pos_lst,reverse=True)
pL=len(pos_lst)
pos_sum=0


if pL%2 != 0: # 홀수  
    for i in range(pL-1):
        if (i+1)%2 != 0:
            sub_sum=pos_lst[i]

        else:
            if pos_lst[i]==1:
                sub_sum+=pos_lst[i]
                pos_sum+=sub_sum
            else:    
                sub_sum*=pos_lst[i]
                pos_sum+=sub_sum
            
            
    pos_sum+=pos_lst[pL-1]


else:
    for i in range(pL):
        if (i+1)%2 != 0:
            sub_sum=pos_lst[i]

        else:
            if pos_lst[i]==1:
                sub_sum+=pos_lst[i]
                pos_sum+=sub_sum
 
            else:    
                sub_sum*=pos_lst[i]
                pos_sum+=sub_sum
        

print(neg_sum+pos_sum)

'''



# 그리디
# 1946 신입사원

'''

1. 서류 심사 순위 오름차순 정렬
2. 서류 심사 순위 두번째부터 마지막 지원자 중 자신보다 위의 사람들보다 면접점수가 높은 경우 cnt+=1


ex1)
1 4  -> o
2 3  -> o
3 2  -> o
4 1  -> o
5 5

-------------

ex2)
1 4  -> o
2 5  
3 6   
4 2  -> o
5 7  
6 1  -> o
7 3  

'''


'''

test_num=int(input())

for _ in range(test_num):
        
    n=int(input())
    lst=[[] for _ in range(n+1)]


    for _ in range(n):
        scr1,scr2=map(int,input().split())
        lst[scr1].append(scr2)


    cnt=1   # 서류 성적 1위 지원자

    
    face_scr=lst[1] # 면접점수
    
    for i in range(2,n+1):
        if lst[i]<face_scr:
            # print(lst[i],face_scr)
            cnt+=1
            face_scr=lst[i]
            
           
    print(cnt)

'''




# 두가지 풀이 유형 존
# 그리디
# 16953 A->B

'''
풀이법

Top-Down 방식

b의 일의 자리숫자가 1이라면 제거 -> 짝수라면 나누기, 1을 제외한 홀수라면 -1



a,b=map(int,input().split())
cnt=0

while a!=b:
    temp=b
    
    if b%10 == 1: # b의 일의자리 숫자가 1이라면 한자리수 제거
        b//=10
        cnt+=1
        
    elif b%2 == 0: # 2의 배수일 때,
        b//=2
        cnt+=1
       
    if b==temp: # b의 변호가 없을때, 1x 2의배수x
        print(-1)
        break

else:
    print(cnt+1)
        
'''



# BFS
# 16953 A->B
# 큐 라이브러리 정리하기

'''
풀이법

Bottom UP 방식

1. 큐에 a와 t(연산횟수) 넣기

2. 추출 후, b보다 클때는 continue하고 작은 경우, 1을 더했을 때와 2를 곱했을 경우 두가지 모두 삽입
   (물론, a==b인 경우 break)
   
3. 큐가 빌 때까지 값이 안나온 경우 print(-1)



from collections import deque

a,b=map(int,input().split())
t=0

q=deque()
q.append((a,t))

while q:
    now,t=q.popleft()
    t+=1

    if now==b:
        print(t)
        break

    elif now>b:
        continue

    else:
        q.append((int(str(now)+"1"),t))
        q.append((now*2,t))

else:
    print(-1)
    

'''

# 그리디
# 1049 기타줄

'''

(개별로 구매했을 때 가장 저가 상품, 패키지로 구매했을 때 가장 저가 상품) -> 둘 중 하나

case1. 개수를 만족시키는 패키지 2개가 더 싼경우

case2. 패키지 + 개별

-> 둘 중 하나 고르면 됨

'''

'''

import sys
input=sys.stdin.readline


n,m=map(int,input().split())

low_pak=1e9     # 가장 싼 패키지 가격
low_pri=1e9     # 가장 싼 상품 가격


for _ in range(m):
    k,r=map(int,input().split())

    if k<low_pak:
        low_pak=k
        
    if r<low_pri:
        low_pri=r


# case1 패키지 구매

# 패키지 개수로 나누어 떨어지는 경우
if n%6==0:
    money=min(n//6*low_pak,n*low_pri)

# 안나누어 떨어질 때
else:
    
    # 개별 구매가 패키지 가격보다 싼경우
    if low_pak>low_pri*6:
        money=n*low_pri

    # 패키지가 더 싼경우
    else:
        left_num=n%6
        money=min(n//6*low_pak+left_num*low_pri,(n//6+1)*low_pak)

print(money)
        
        
'''


# 그리디
# 10610 30

'''
n=int(input())

lst=[]

for i in map(int,str(n)):
    lst.append(i)

sum_thr=0
lst=sorted(lst,reverse=True)

for num in lst:
    sum_thr+=num

if sum_thr%3 !=0 or lst[len(lst)-1]!=0:
    print(-1)
else:
    result=''.join(map(str,lst))
    print(int(result))
    
'''

# 그리디
# 1449 수리공 항승

'''
풀이법

- 오름 차순 정렬

- 각 구멍 인덱스에 순차 접근

- 테이프 길이 범위 설정 -> 그 안에 들어가 있는 원소 다음 부터 다시 반복 (cnt+=1) , 길이 설정 초기화

'''

'''
n,l=map(int,input().split())

lst=list(map(int,input().split()))
lst.sort()

cnt=1

min_dis=lst[0]-0.5
max_dis=lst[0]-0.5+l

for i in range(1,len(lst)):
    if lst[i]>=min_dis and lst[i]<=max_dis:
        continue

    else:
        min_dis=lst[i]-0.5
        max_dis=lst[i]-0.5+l
        cnt+=1

print(cnt)

'''
    

# 2차원 배열 선언 정리 / lambda함수 정리 / 파이썬 min, max heap 사용 정리
# 그리디
# 13904 과제

'''
풀이법

1. 점수가 가장 높은 순, 점수가 같다면 기간이 짧게 남은 순서로 정렬

2. 최대 날짜 크기만큼의 리스트 초기화

3. 정렬된 리스트에 순차 접근해 날짜에 해당하는 인덱스 부터 첫째날까지 할당되지 않은 공간이 있다면 할당시켜줌


**
답지 풀이
-> max heap을 이용한 구현
-> 궁금점: max heap을 사용하면 기간이 짧게 남은 순서로 추출이 가능한가 ..?

'''

'''
import sys
input=sys.stdin.readline

lst=[]
n=int(input())

max_date=0

for _ in range(n):
    
    d,w=map(int,input().split())

    if d>max_date:
        max_date=d

    lst.append((d,w))



lst.sort(key=lambda x :(-x[1],x[0]))


res=[0]*(max_date+1)


for i in range(n):
    for j in range(lst[i][0],0,-1):

        if res[j]!=0:
            continue
        else:
            res[j]=lst[i][1]
            break
      

print(sum(res))

'''



# 구현
# 14503 구현

# 실패코드 1

'''
1. while문 반복 조건 -> (r,c) 좌표가 graph 영역 안 + 벽이 뒤에 없을때까지

2. while문
    a) 현재 칸 방문처리 후, 상하좌우(d방향으로 우선시 이동) 벽이 아닌 공간 중 벽이 아닌공간이 있다면 청소
    b) 상하좌우에 이동할 공간이 없다면 d방향의 반대 방향으로 이동
       -> 반복조건 통과 시 a) 과정 다시 진행
       -> 반복조건 걸릴 시 while문 탈출, 청소한 방 개수 출력


import sys
input=sys.stdin.readline

# n:행, m:열
n,m=map(int,input().split())

# (r,c):좌표, d:방향
r,c,d=map(int,input().split())

# 방 입력
graph=[]

for _ in range(n):
    graph.append(list(map(int,input().split())))
    


# 이동 방향 (북->서->남->동, 반시계)

if d==0: # 북쪽일 때
    dn=[-1,0,1,0]
    dm=[0,-1,0,1]

elif d==1:  # 서쪽일 때
    dn=[0,1,0,-1]
    dm=[-1,0,1,0]

elif d==1:  # 남쪽일 때
    dn=[1,0,-1,0]
    dm=[0,1,0,-1]

else:   # 동쪽일 때
    dn=[0,-1,0,1]
    dm=[1,0,-1,-0]

    
def moving(graph,r,c,cnt):
    if graph[r][c]==0:  # 청소하지 않은 공간이라면
        cnt+=1
        graph[r][c]=True  # 청소하고 방문처리

        temp_r=r
        temp_c=c
        
        for i in range(4):  
            r+=dn[i]
            c+=dm[i]

            if graph[r][c]==0 and r>=0 and r<=n and c>=0 and c<=m:  # 조건을 만족한다면 함수 재귀 호출
                moving(graph,r,c,cnt)
            else:   # 아니라면 현위치로 다시 유지
                r=temp_r
                c=temp_c

    if r==temp_r and c==temp_c: # 상하좌우 중 어디로 이동하지 못했다면 
        # 후진진행
        r+=dn[3]
        c+=dm[3]

        # 조건 만족 시
        # 청소는 했지만 이동하는 경우 구현 못함 
        if graph[r][c]==0 and r>=0 and r<=n and c>=0 and c<=m:
            moving(graph,r,c,cnt)
        else:
            return cnt

result=moving(graph,r,c,0)

print(result)
                
'''            



# 실패코드 2
# 맞게 푼거 같은데 뭔 개소린지 모르겠다 ,, 욕나오네 

# 풀이법
'''
1. while문 반복 조건 -> (r,c) 좌표가 graph 영역 안 + 벽이 뒤에 없을때까지

2. while문
    a) 현재 칸 방문처리 후, 상하좌우(d방향으로 우선시 이동) 벽이 아닌 공간 중 벽이 아닌공간이 있다면 청소
       
    b) 상하좌우에 이동할 공간이 없다면 진행 반대 방향으로 이동
       -> b조건에서 방문여부는 좌표값 이동에 영향을 미치지 않음. 즉, 인덱스가 밖이거나 벽만 아니면 됨
          : 방문처리 그래프 별도 선언
       -> 반목 조건 통과 다시 a단계 진행
       -> 반복조건 걸릴 시 while문 탈출, 청소한 방 개수 출력


- d가 현재 방향이고, r이 현재 위치 일때(rx,ry)

시계 반대 방향으로 돌리는 법
: d=(d+3)%4, moving_r=r+d

진행 방향의 반대로 한칸 이동하는 법
: moving_r=r-d

'''

'''
import sys
input=sys.stdin.readline

# n:행, m:열
n,m=map(int,input().split())

# (r,c):좌표, d:방향
r,c,d=map(int,input().split())

# 방 입력
graph=[]
for _ in range(n):
    graph.append(list(map(int,input().split())))

# 방문 여부 확인 리스트
visited=[[False]*m for _ in range(n)]


# 이동 방향 (북,동,남,서 -> 시계 방향)
dx=[-1,0,1,0]
dy=[0,1,0,-1]


# 출발지 방문 체크
visited[r][c]=1
cnt=1

    
while True:
    flag=0   # 갱신이 이루어졌는지 체크하기 위한 변수
    
    for _ in range(4): 
        d=(d+3)%4    # 반시계 방향으로 한칸 돌림 *point
        nr=r+dx[d]
        nc=c+dy[d]

        if 0<=nr<n and 0<=nc<m and graph[nr][nc]==0:  # 범위 안 조건 만족, 벽이 아니고
            if visited[nr][nc]==False:   # 방문하지 않은 곳이라면
                visited[nr][nc]=True    # 방문처리
                print("(",nr,",",nc,")")
                cnt+=1
                r=nr
                c=nc
                flag==1 # 상하좌우 중 이동해서 청소했다는 뜻
                break    # for문 탈출 / cotinue: 다음 for문 실행  *point


    # 상하좌우로 좌표가 이동하지 못하는 경우, 반대방향으로 한칸 후진
    if flag==0:
        nr=r-dx[d]
        nc=c-dy[d]

        if nr<0 or nr>=n or nc<0 or nc>=m or graph[nr][nc]==1:  # 방문여부는 상관 없음
            print(cnt)
            break
        else:
            r=nr
            c=nc
        
'''                  
                    
# 정답 코드

'''
import sys
input = sys.stdin.readline
from collections import deque

n,m = map(int,input().split())
graph = []
visited = [[0] * m for _ in range(n)]
r,c,d = map(int,input().split())

# d => 0,3,2,1 순서로 돌아야한다.
dx = [-1,0,1,0]
dy = [0,1,0,-1]

for _ in range(n):
    graph.append(list(map(int,input().split())))

# 처음 시작하는 곳 방문 처리
visited[r][c] = 1
cnt = 1

while 1:
    flag = 0
    # 4방향 확인
    for _ in range(4):
        # 0,3,2,1 순서 만들어주기위한 식
        nx = r + dx[(d+3)%4]
        ny = c + dy[(d+3)%4]
        # 한번 돌았으면 그 방향으로 작업시작
        d = (d+3)%4
        if 0 <= nx < n and 0 <= ny < m and graph[nx][ny] == 0:
            if visited[nx][ny] == 0:
                print(nx,ny)
                visited[nx][ny] = 1
                cnt += 1
                r = nx
                c = ny
                #청소 한 방향이라도 했으면 다음으로 넘어감
                flag = 1
                break
    if flag == 0: # 4방향 모두 청소가 되어 있을 때,
        if graph[r-dx[d]][c-dy[d]] == 1: #후진했는데 벽
            print(cnt)
            break
        else:
            r,c = r-dx[d],c-dy[d]

'''

# 1474 방 번호

'''
num=int(input())
lst=[0]*10

for i in map(int,str(num)):
    if i==6 or i==9:
        if lst[6]>=lst[9]:
            lst[9]+=1
        else:
            lst[6]+=1
    else:
        lst[i]+=1

print(max(lst))

'''


# 실패
# 구현
# 15668 치킨 배달

'''
각각의 집에서 가장 가까운 치킨 집까지의 거리 추출 완료

Q1. 삭제해야할 치킨 집을 고르는 문제
-> 치킨 집을 삭제하면 가장 가까운 치킨 집 까지의 거리에도 문제가 발생함 ,, !
-> How to ,,?

어떤 기준으로 지점을 삭제해야 가장 큰 이득을 취할 수 있을 것인가 ,.?
-> 각각의 가게까지 모든 집으로부터 거리 누적 합 구하기
-> 거리가 가장 작은 것부터 M개까지 도시의 거리값을 구하되, 다음 좌표를 추가한 값이 이전 보다 커지는 경우 -> 전 단계 출력


Q2. 누적값이 가장 작은 도시부터 거리값을 구할 때, 해당되는 치킨 집의 좌표값 어떻게 일치시킬 것인가 ,,?



'''

'''
import sys
input=sys.stdin.readline

n,m=map(int,input().split())
lst=[]

for _ in range(n):
    lst.append(list(map(int,input().split())))


# 치킨 좌표
c_idx=[]

# 집 좌표
h_idx=[]

# 치킨, 집 좌표 각각의 리스트에 삽입
for i in range(n):
    for j in range(n):
        if lst[i][j]==1:
            h_idx.append((i,j))
        elif lst[i][j]==2:
            c_idx.append((i,j))
        else:
            continue


# 각 집마다 가장 가까운 치킨 집 거리
dist_lst=[]

for i in range(len(c_idx)):
    cx,cy=c_idx[i]
    dist=0
    
    for hx,hy in h_idx:
        dist+=abs(hx-cx)+abs(hy-cy)

    dist_lst.append(dist)



print(dist_lst)
            
'''

# dfs, bfs
# 1260  DFS와 BFS

'''
핵심
-> 인접 리스트 형태 변환하기 + 정렬 수행

'''

'''
import sys
from collections import deque
input=sys.stdin.readline


n,m,start=map(int,input().split())
graph=[[] for i in range(n+1)]


# 정점 연결 정보 -> 인접 리스트 방식 변경
for _ in range(m):
    s,e=map(int,input().split())
    graph[s].append(e)
    graph[e].append(s)

# 번호가 낮은 순서대로 방문해야하므로 올림차순 정렬
for i in range(n+1):
    graph[i].sort()



visited1=[False]*(n+1)
visited2=[False]*(n+1)

# dfs
def dfs(graph,start,visited1):
    visited1[start]=True
    print(start,end=' ')

    for i in graph[start]:
        if visited1[i]!=True:
            visited1[i]=True
            dfs(graph,i,visited1)


# bfs
def bfs(graph,start,visited2):
    q=deque([start])
    visited2[start]=True

    while q:
        node=q.popleft()
        print(node,end=' ')

        for i in graph[node]:
            if visited2[i]!=True:
                q.append(i)
                visited2[i]=True



dfs(graph,start,visited1)
print()
bfs(graph,start,visited2)

'''

# bfs
# 2606 바이러스

'''
import sys
from collections import deque
input=sys.stdin.readline

n=int(input())
m=int(input())
start=1

# 인접 리스트 형태 변환
graph=[[] for _ in range(n+1)]
for _ in range(m):
    v1,v2=map(int,input().split())
    graph[v1].append(v2)
    graph[v2].append(v1)

visited=[False]*(n+1)
num=0

def bfs(graph,start,visited,num):
    q=deque([start])
    visited[start]=True

    while q:
        node=q.popleft()
        for i in graph[node]:
            if visited[i]!=True:
                q.append(i)
                num+=1
                visited[i]=True

    return num


result=bfs(graph,start,visited,num)
print(result)
    
'''


# 11724 연결 요소의 개수

'''
import sys
sys.setrecursionlimit(10**6)
input=sys.stdin.readline

n,m=map(int,input().split())

graph=[[] for _ in range(n+1)]
for _ in range(m):
    v1,v2=map(int,input().split())
    graph[v1].append(v2)
    graph[v2].append(v1)


visited=[False]*(n+1)

def dfs(graph,start,visited):
    visited[start]=True

    # 인접노드에 대해 전부 방문 처리
    for i in graph[start]:
        if visited[i]!=True:
            dfs(graph,i,visited)


result=0
for i in range(1,n+1):
    # 방문하지 않은 곳이 있다면 결과 값 1증가, 인접 노드 방문 처리 -> dfs 함수 역할
    if visited[i]!=True:
        dfs(graph,i,visited)
        result+=1

print(result)

'''
        
 

# BFS
# 7576 토마토

'''
Point
1. 걸리는 기간 구하는 문제 -> BFS 이용
2. Queue 사용

고민점: 토마토가 있는 곳에 대해 어떻게 각각의 날짜를 증가 동시다발적으로 확산 할 것인가

-> 큐에 토마토가 있는 곳의 위치를 먼저 넣음.
    큐에서 차레대로 꺼내면서 확인하고, 조건에 부합하는 곳의 인덱스 값을 변경하고 큐에 인덱스 차례로 추가
   
-> 각 토마토에 대해 차례대로 수행될 수 밖에 없음(Queue:First In First Out)


break: 반복문 탈출
exit: 프로그램 종료

'''

'''
import sys
from collections import deque
input=sys.stdin.readline

m,n=map(int,input().split())

# 토마토 밭
graph=[]
for _ in range(n):
    graph.append(list(map(int,input().split())))


# 토마토가 있는 위치
q=deque([])
for i in range(n):
    for j in range(m):
        if graph[i][j]==1:
            q.append([i,j])


dx=[-1,1,0,0]
dy=[0,0,-1,1]

def bfs():
    while q:
        x,y=q.popleft()

        for k in range(4):
            nx=x+dx[k]
            ny=y+dy[k]

            if 0<=nx<n and 0<=ny<m:
                if graph[nx][ny]==0:
                    graph[nx][ny]=graph[x][y]+1
                    q.append([nx,ny])


bfs()   # bfs 알고리즘 수행
res=0    # day
flag=True  # 모든 감염되었는지 확인 하기위한 변수

for i in graph:
    for j in i:
        if j==0:
            flag=False
            break
    res=max(res,max(i))


if flag==True:
    print(res-1) # 감염까지 걸리는 날짜를 구함으로 1빼주기 
else:
    print(-1)

'''

# 인터넷 답 넣고  넘어감 
# 7562 나이트의 이동

'''
import sys
from collections import deque
input=sys.stdin.readline

# 나이트의 움직임
dx = [-1, 1, 2, 2, 1, -1, -2, -2]
dy = [2, 2, 1, -1, -2, -2, -1, 1]

def bfs():

    q=deque()
    q.append((sx,sy))

    while q:
        x,y=q.popleft()
        print(x,y)
        
        if x==vx and y==vy:
            print('yes')
            return graph[x][y]-1

        for k in range(8):
            nx=x+dx[k]
            ny=x+dy[k]

            if 0<=nx<n and 0<=ny<n and graph[nx][ny]==0:
                graph[nx][ny]=graph[x][y]+1
                q.append((nx,ny))


n=int(input())
sx,sy=map(int,input().split())
vx,vy=map(int,input().split())
graph=[[0]*n for _ in range(n)]
graph[sx][sy]=1           
print(bfs())

-> 왜 안되는지 모르겠음 ,, 

'''

# 인터넷 답 - 뭐가 다르냐고

'''
from collections import deque
import sys
input = sys.stdin.readline

t = int(input().rstrip())


def bfs() :
    dx = [-1, 1, 2, 2, 1, -1, -2, -2]
    dy = [2, 2, 1, -1, -2, -2, -1, 1]

    q = deque()
    q.append((startX, startY))
    while q :
        x, y = q.popleft()
        if x == endX and y == endY :
            return matrix[x][y] -1 
        for i in range(8) :
            nx = x + dx[i]
            ny = y + dy[i]
            if 0<=nx<l and 0<=ny<l and matrix[nx][ny] == 0 :
                matrix[nx][ny] = matrix[x][y] + 1
                q.append((nx,ny))
                
            
        
for _ in range(t) :
    l = int(input().rstrip())
    startX, startY = map(int, input().rstrip().split())
    endX, endY = map(int, input().rstrip().split())
    matrix = [[0]*l for _ in range(l)]
    matrix[startX][startY] = 1
    print(bfs())

'''

# DP
# 2579 계단 오르기 (복습)

'''
import sys
input=sys.stdin.readline

n=int(input())

# n의 개수가 3보다 작을 때 고려해서, lst의 초기값을 0으로 초기화
lst=[0]*(n+1)
for i in range(n):
    lst[i+1]=int(input())

# n의 개수가 3보다 작을 때를 대비해서 0으로 조건값 크기만큼 초기화
d=[0]*301 # 각 계단을 밟았을 때 최고점

d[1]=lst[1]
d[2]=lst[1]+lst[2]


d[3]=max(lst[1]+lst[3],lst[2]+lst[3])

for i in range(4,n+1):
    d[i]=max(d[i-3]+lst[i-1]+lst[i],d[i-2]+lst[i])

print(d[n])

'''


# DP
# 파도반 수열

'''
t=int(input())

for _ in range(t):
    n=int(input())

    d=[0]*101

    d[1]=1
    d[2]=1
    d[3]=1
    d[4]=2
    d[5]=2

    for i in range(5,n+1):
        d[i]=d[i-5]+d[i-1]

    print(d[n])

'''

# DP
# 부녀회장이 될테야

'''
t=int(input())

for _ in range(t):
    # k:층, n:호
    k=int(input())
    n=int(input())

    lst=[[0]*15 for _ in range(k+1)]

    for i in range(1,15):
        lst[0][i]=i


    for i in range(1,k+1):
        for j in range(1,15):
            for h in range(1,j+1):
                lst[i][j]+=lst[i-1][h]

    print(lst[k][n])
   
'''

# DP
# 1932 정수 삼각형

'''
n=int(input())

lst=[[0]*10 for _ in range(n+1)]

for i in range(10):
    lst[0][i]=1

for i in range(n+1):
    for j in range(10):
        if j==0:    # 수는 0으로 시작 가능함
            lst[i][j]=1
        else:
            lst[i][j]=lst[i-1][j]+lst[i][j-1]

cnt=0

for i in range(10):
    cnt+=lst[n-1][i]

print(cnt%10007)

'''


# DP
# 10844 쉬운 계단 수

'''
풀이법

- len 길이에 따라 각자리 숫자가 가질 수 있는 계단 개수를 누적해서 구함
  -> len 길이 - 각자리 숫자 : 2차원 배열로 접근

'''

'''
n=int(input())

dp=[[0]*10 for _ in range(n)]

for j in range(1,10):
    dp[0][j]=1


for i in range(1,n):
    for j in range(0,10):
        if j==0:
            dp[i][j]=dp[i-1][j+1]   # 한 자리수 낮을 때, 1이 갖는 개수와 동일 (0)
        elif j==9:
            dp[i][j]=dp[i-1][j-1]   # 한 자리수 낮을 때, 8이 갖는 개수와 동일 (9)

        else:
            dp[i][j]=dp[i-1][j-1]+dp[i-1][j+1]  # 양 사이드 숫자에서 갖는 개수의 합과 동일 (1~8)

cnt=0
for j in range(10):
    cnt+=dp[n-1][j]

print(cnt%1000000000)
    
'''

# DFS, BFS
# 1260 DFS와 BFS (복습)

'''

* 방문할 수 있는 정점이 여러개인 경우 번호가 작은것 부터 수행 -> 리스트 각 행에 대해 오름차순 정렬

[깊이 우선 탐색]
1. 시작 노드 방문 처리
2. 인접 노드 중 방문하지 않은 노드가 있다면 방문 처리 -> DFS 호출
3. 모든 노드 방문 시 탈출


[너비 우선 탐색]
1. 시작 노드 큐에 삽입 후, 방문 처
2. while(큐가 빌때 까지 반복)
-> 노드 추출 후 인접노드에 방문하지 않은 노드가 있다면 큐에 삽입ㄷ

'''

'''
import sys
from collections import deque
input=sys.stdin.readline

n,m,v=map(int,input().split())

# 방문 여부 리스트
visited=[False] * (n+1)

# 간선 정보 그래프
graph=[[] for _ in range(n+1)]

for _ in range(m):
    v1,v2=map(int,input().split())
    graph[v1].append(v2)
    graph[v2].append(v1)




# 그래프 오름차순 정렬
for i in range(1,n+1):
    graph[i].sort()


# DFS
def dfs(graph,start):
    visited[start]=True
    print(start,end=' ')
    

    for nex_nod in graph[start]:
        if visited[nex_nod]!= True:
            dfs(graph,nex_nod)



dfs(graph,v)
print()
visited=[False] * (n+1)


def bfs(graph,start):
    q=deque()
    q.append(start)
    visited[start]=True

    while(q):
        cur_nod=q.popleft()
        print(cur_nod,end=' ')

        for nex_nod in graph[cur_nod]:
            if visited[nex_nod]!=True:
                q.append(nex_nod)
                visited[nex_nod]=True

bfs(graph,v)

'''              
        
    
# DFS, BFS 
# 2606 바이러스 (복습)

'''
# DFS 구현 시 -> num 변수가 로컬 변수이기 때문에 함수에서 카운트를 증가 시켜주려면 glbal 선언 후 접근 해야함

n=int(input())
m=int(input())

# 간선 그래프
graph=[[] for _ in range(n+1)]

for _ in range(m):
    v1,v2=map(int,input().split())
    graph[v1].append(v2)
    graph[v2].append(v1)

# 방문 정보
visited=[False]*(n+1)


# DFS 
num=0
def dfs(graph,start):
    global num
    visited[start]=True
    
    
    
    for nex_nod in graph[start]:
        if visited[nex_nod]!=True:
            num+=1
            dfs(graph,nex_nod)

dfs(graph,1)

print(num)




# BFS
from collections import deque

def bfs(graph,start,num):
    visited[start]=True # 방문 처리
    q=deque()
    q.append(start)

    while q:
        cur=q.popleft()
        num+=1

        for nex in graph[cur]:
            if visited[nex]!=True:
                q.append(nex)
                visited[nex]=True

    return num

res=bfs(graph,1,0)
print(res-1)
    
'''

# DFS, BFS
# 2178 미로 탐색 (복습)

'''
# BFS 알고리즘 - 최단 경로
from collections import deque

n,m=map(int,input().split())


# 간선 정보 입력
graph=[[] for _ in range(n)]

for i in range(n):
    str=input()
    for j in map(int,str):
        graph[i].append(j)
 

def bfs(graph,x,y):
    q=deque()
    q.append((x,y)) # 큐 라이브러리 생성 후 첫번째 인덱스 넣기

    dx=[-1,1,0,0]
    dy=[0,0,-1,1]

    while q:
        x,y=q.popleft()
        
        for k in range(4):
            nx=x+dx[k]
            ny=y+dy[k]

            if 0<=nx<n and 0<=ny<m and graph[nx][ny]==1:
                graph[nx][ny]=graph[x][y]+1
                q.append((nx,ny))


bfs(graph,0,0)


print(graph[n-1][m-1])

'''


# DP 
# 2839 설탕 배달(복습)

'''
1. DP 테이블 INF 초기화
2. 3일때, DP 테이블 한번 돌려서 최솟값 갱신
3. 5일때, DP 테이블 돌려서 최솟갑 갱신
4. 구하고자하는 인덱스 출력

'''

'''
INF=1e9

n=int(input())

dp=[INF]*5001

weight=[3,5]

dp[3]=1
dp[5]=1


for i in range(6,5001):
    for wet in weight:
        dp[i]=min(dp[i-wet]+1,dp[i])

if dp[n]==INF:
    print(-1)
else:
    print(dp[n])

'''


# 2293 동전 1

'''
n,k=map(int,input().split())

lst=[]
for _ in range(n):
    money=int(input())
    lst.append(money)


dp=[0]*(k+1)

# 동전을 1개 사용하는 경우를 위해 dp[0]값 1 초기화
dp[0]=1

for coin in lst:
    for i in range(coin,k+1):
        if dp[i-coin]!=0:
            dp[i]+=dp[i-coin]

print(dp[k])
                
'''

# DFS 
# 2667 단지번호붙이기(복습)

'''
n=int(input())
graph=[]

for _ in range(n):
    graph.append(list(map(int,input())))


def dfs(graph,x,y,num):
    graph[x][y]=0

    dx=[-1,1,0,0]
    dy=[0,0,-1,1]

    for i in range(4):
        nx=x+dx[i]
        ny=y+dy[i]

        if 0<=nx<n and 0<=ny<n and graph[nx][ny]==1:
            num=dfs(graph,nx,ny,num+1)

    return num


lst=[]

for i in range(n):
    for j in range(n):
        if graph[i][j]==1:
            lst.append(dfs(graph,i,j,1))

lst.sort()
print(len(lst))
for i in range(len(lst)):
    print(lst[i])

'''


# 프로그래머스 공원산책

'''
graph=[]

park=['SOOO','OOOO','OOOO']
routes=["E 2","E 1","S 2","W 1"]

print(len(park))
print(len(park[0]))

for i in range(len(park)):
    for j in range(len(park[i])):
        if park[i][j]=='S':
            # 현재 좌표
            row=i
            col=j

# 방향 인덱스 딕셔너리
dir_dict={'N':0,'S':1,'W':2,'E':3}

# 방향 정보 리스트(N,S,W,E)
dx=[0,0,-1,1]
dy=[1,-1,0,0]


# 좌표 이동
for i in routes:
    dir,num=i.split()
    num=int(num)
    idx=dir_dict[dir]
    print('idx',dir)
        
    # num 만큼 좌표 이동
    for _ in range(num):
        nx=row+dx[idx]
        ny=col+dy[idx]

        if 0<=ny<len(park[0]) and 0<=nx<len(park):  # 범위 내에 있으면서 이동할 수 있는 통로에 있다면
            if park[nx][ny]=='O':
                row=nx
                col=ny
        
        
        print(row,col)
    print('###')

     
'''

# 프로그래머스 바탕화면 정리

'''
wallpaper = [".#...", "..#..", "...#."]
lst=[]

for i in range(len(wallpaper)):
    for j in range(len(wallpaper[i])):
        if wallpaper[i][j]=='#':
            lst.append([i,j])

# 시작점 -> 가장 위쪽 row값, 가장 왼쪽 col값
lst.sort(key=lambda x:x[0])
lux=lst[0][0]
print(lux)


lst.sort(key=lambda x:x[1])
luy=lst[0][1]
print(luy)


# 끝점 -> 가장 아래쪽 row값, 가장 오른쪽 col값
lst.sort(key=lambda x:-x[0])
rdx=lst[0][0]+1
print(rdx)

lst.sort(key=lambda x:-x[1])
rdy=lst[0][1]+1
print(rdy)

'''

# 프로그래머스 덧칠하기

'''
- section 차례로 팝하고 cnt+1
  >> 페인트 칠한 길이: 팝 원소 + 롤러 거리

- 순서대로 팝하면 페인트 칠한 길이보다 작다면 단순히 팝만 시켜주기
 >> 크다면 페인트 칠한 길이 갱신
 >> 만약 구역 길이보다 길다면 탈출 
 
 - cnt 출력
 
'''

# 프로그래머스 카드뭉치

'''
def solution(cards1, cards2, goal):
    answer = 'Yes'
    
    for i in range(len(goal)):
        letter=goal[i]
        
        if len(cards1)!=0 and cards1[0]==letter:
            print(cards1[0])
            cards1.pop(0)
            print('after cards1',cards1)
            pass
        
        elif len(cards2)!=0 and cards2[0]==letter:
            print(cards2[0])
            cards2.pop(0)
            print('after cards2',cards2)
            pass
            
        else:
            return 'No'

    return answer
    
    


cards1=["i", "drink", "water"]
cards2=["want", "to"]
goal=["i", "want", "to", "drink", "water"]

solution(cards1,cards2,goal)

'''

# 프로그래머스 둘만의 암호

'''
def solution(s, skip, index):
    answer = ''
    
    black_lst=list(skip)
    lst=list(s)
    
    for i in range(len(lst)):
        num=ord(s[i])
        cnt=0
        while True:
            num+=1
            if num>ord('z'):
                num-=26
            str=chr(num)

            if str in black_lst:
                print('pass',str)
                pass
            
            else:
                cnt+=1
            
            if cnt==index:
                answer+=chr(num)
                break

    return answer


s="aukks"
skip="wbqd"
index=5


solution(s,skip,index)

'''


# 프로그래머스 크기가 작은 부분문자열

'''
def solution(t, p):
    answer = 0
    
    t=list(t)
    p=list(p)
   
    # p 숫자 변환
    p_num=''
    for i in p:
        p_num+=i
    p_num=int(p_num)

    ## 자료형 변환하면 됨 p_num=int(p)

        
    lst=[]
    for i in range(len(t)-len(p)+1):
        lst.append(list(t[i:i+len(str(p_num))]))

    
    for i in lst:
        num=''
        for j in i:
            num+=j
        if int(num)<=p_num:
            answer+=1     
    
    
   return answer


t="3141592"
p="271"

solution(t,p)

'''

# 프로그래머스 숫자 짝꿍

'''
def solution(X, Y):
    answer = ''

    for i in range(9,-1,-1):
        answer+=(str(i)*min(X.count(str(i)),Y.count(str(i))))
    
    if answer=='':
        answer='-1'
    
    if len(answer) >=2 and answer[1] =='0':
        answer='0'
    
    return answer

X="12321"
Y="42531"
print(solution(X, Y))

'''


# 프로그래머스 대충 만든 자판

'''
answer=[]


keymap=['ABACD','BCEFD']
targets=['ABCD','AABB']

dic={}

for x in keymap:
    for i in range(len(x)):
        if x[i] in dic:
            dic[x[i]]=min(dic[x[i]],x.index(x[i]))
        else:
            dic[x[i]]=x.index(x[i])

            
for x in targets:
    num=0
    for j in range(len(x)):
        if x[j] not in dic:
            return [-1]
        
        else:
            num+=(dic[x[j]]+1)
        
    answer.append(num)

'''

# 프로그래머스 공원 산책 -> 실패

'''
park=["SOO","OXX","OOO"]
routes=["E 2","S 2","W 1"]	
# E,W,N,S
dx=[1,-1,0,0]
dy=[0,0,1,-1]

for i in range(len(park)):
    for j in range(len(park[0])):
        if park[i][j]=='S':
            row=i
            col=j

print('start',row,col)
for rot in routes:
    dir,num=rot.split()
    num=int(num)
    nx=row
    ny=col
    cnt=0

    if dir=='E':
        for _ in range(num):
            nx+=dx[0]
            ny+=dy[0]

            if park[nx][ny]!='X' and 0<=nx<len(park) and 0<=ny<len(park[0]):
                cnt+=1
            
            else:
                break

            if cnt==num:
                row=nx
                col=ny
  


    elif dir=='W':
        for _ in range(num):
            nx=row+dx[1]
            ny=row+dy[1]
    

            if park[nx][ny]!='X' and 0<=nx<len(park) and 0<=ny<len(park[0]):
                cnt+=1
            
            else:
                break

            if cnt==num:
                row=nx
                col=ny


    elif dir=='N':
        for _ in range(num):
            nx=row+dx[2]
            ny=row+dy[2]
            
            if park[nx][ny]!='X' and 0<=nx<len(park) and 0<=ny<len(park[0]):
                cnt+=1
                
            else:
                break

            if cnt==num:
                row=nx
                col=ny


    elif dir=='S':
        for _ in range(num):
            nx=row+dx[3]
            ny=row+dy[3]
            

            if park[nx][ny]!='X' and 0<=nx<len(park) and 0<=ny<len(park[0]):
                cnt+=1
                
            else:
                break

            if cnt==num:
                row=nx
                col=ny


    print((dir,num),row,col)
    
print('last',row,col)

 
'''


# 순열 모듈

'''
from itertools import permutations

lst=['A','B','C','D']
print(''.join(list(permutations(lst))))

'''


# 프로그래머스 문자열 나누기

'''
def solution(s):
    answer = 0
    s=list(s)
    s.reverse()
    
    while True:
        cnt1=0
        cnt2=0
        x=s[len(s)-1]
        print(x)
        
        for _ in range(len(s)):
            y=s.pop()
            
            if x==y:
                cnt1+=1
                
            else:
                cnt2+=1
        
            if cnt1==cnt2:
                answer+=1
                
                if len(s)==0:
                    return answer
                
                break
            

        if len(s)==0:
            if cnt1>0:
                answer+=1
            break
        
    return answer

'''

# 큐 사용해서 구현 해보기

'''
from collections import deque

def solutions(s):
    answer=0
    q=deque(s)

    while q:
        x=q.popleft()
        cnt1=1
        cnt2=0

        while q:
            y=q.popleft()

            if x==y:
                cnt1+=1
            else:
                cnt2+=1

            if cnt1==cnt2:
                answer+=1
                break

        if cnt1!=cnt2:
            answer+=1

    return answer

'''

# 프로그래머스 성격유형

'''
def solution(survey, choices):
    answer=''
    
    dic={'R':0,'T':0,'C':0,'F':0,'J':0,'M':0,'A':0,'N':0}
    
    for s,c in zip(survey,choices):
        if c>4:
            dic[s[1]]+=c-4
        elif c<4:
            dic[s[0]]+=4-c
        
    lst=list(dic.items())
    
    
    for i in range(0,8,2): # 0에서 7까지 두칸 간격
        if lst[i][1]<lst[i+1][1]:
            answer+=lst[i+1][0]
        else:
            answer+=lst[i][0]
    
    return answer

'''


# 내 코드 -> 런타임 에러(일부 테스트 케이스)

'''
# 2진수 변환
def binary(n):
    lst=[]
    while True:
        if n<2:
            if n==1:
                lst.append(str(1))
            break
    
        else:
            lst.append(str(n%2))
            n//=2

    lst.reverse()
    
    return ''.join(lst)



def solution(n,arr1,arr2):
    answer=[]
    lst=[]
    for i in range(len(arr1)):
        sum=int(binary(arr1[i]))+int(binary(arr2[i]))

        lst.append(sum)
    
    for x in lst:
        ans=''
        for i in range(len(str(x))):
            if int(str(x)[i])>0:
                ans+='#'
            else:
                ans+=' '

        # arr1과 arr2를 더한 값의 자리수가 n보다 작을때, 가장 앞 0에 대한 공백처리가 불가능함 -> 모자란 만큼 공백 넣어주기
        if len(str(ans))<n:
            cnt=n-len(ans)
            ans=list(ans)
            ans.reverse()
            
            for _ in range(cnt):
                ans+=' '
            ans.reverse()
            ans=''.join(ans)
            
        answer.append(ans)

    return answer                

    
n=6
arr1=[46, 33, 33 ,22, 31, 50]
arr2=[27 ,56, 19, 14, 14, 10]

print(solution(n,arr1,arr2))


'''











