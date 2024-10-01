# 피보나치

'''
def fibo(x):
    if x==1 or x==2:
        return 1
    else:
        return fibo(x-1)+fibo(x-2)

print(fibo(5))

'''


# 피보나치(Top-Down방식)

'''
# 한 번 계산된 결과를 Memoization하기 위한 리스트 초기화
d=[0]*100

def fibo(x):

    # 종료 조건
    if x==1 or x==2:
        return 1

    # 계산한적 있다면 그대로 반환
    if d[x]!=0:
        return d[x]

    # 아직 계산하지 않은 문제라면 점화식에 따라 결과 반환, d[x]추가
    else:
        d[x]=fibo(x-1)+fibo(x-2)
        return(d[x])

print(fibo(99))

'''

# 피보나치 (Bottom-Up방식)

'''
d=[0]*100

d[1]=1
d[2]=1
n=99

for i in range(3,n+1):
    d[i]=d[i-1]+d[i-2]


print(d[n])

'''



# Youtube 문제

# 1. 개미 전사

'''
풀이법

n번째 창고에서 최대 이득 -> max( n-2 창고까지의 이득 +  n창고 이득, n-1 창고까지 이득

'''

'''

N=int(input())
lst=list(map(int,input().split()))

# Memoization -> 최대 창고 입력 개수: 100
d=[0]*100


d[0]=lst[0]
d[1]=max(lst[0],lst[1]) # 3

for i in range(2,N):
    d[i]=max(d[i-2]+lst[i],d[i-1])

print(d[N-1])

'''


# 2. 1로 만들기


# 그리디 접근 -> 최소의 연산 만족 x

'''
def cal(x):
    while x!=1:

        if x%5==0:
            x=x//5
            print(x)

        elif x%3==0:
            x=x//3
            print(x)

        elif x%2==0:
            x=x//2
            print(x)
        else:
            x=x-1
            print(x)

print(cal(X))

'''
# 풀이 코드

'''
N=int(input())

# DP 테이블 초기화, N = 0,1 -> 0 초기화
d=[0]*30001


for i in range(2,N+1):

    # 1감소 했을 때의 숫자에서 최소의 경우
    d[i]=d[i-1]+1

    # N이 배수일때 나눈 몫의 숫자에서 최선의 경우의 수 + 1  VS  1감소 했을 때 최소의 경우
    if i%5==0:
        d[i]=min(d[i],d[i//5]+1)


    if i%3==0:
        d[i]=min(d[i],d[i//3]+1)


    if i%2==0:
        d[i]=min(d[i],d[i//2]+1)


print(d[N])

'''

# 3. 효율적인 화폐 구성

'''
풀이법

i번째 금액에서 화폐 단위를 뺀 값(i-array) -> INF가 아니라면 -> min(i-array의 화폐 구성 개수 + 1, i번째 화폐 구성 개수)

'''

'''
N, M = map(int,input().split())

# 화폐단위 저장할 배열
array=[]
for i in range(N):
    num=int(input())
    array.append(num)


d=[100001]*(M+1) # 100001 -> INF(무한) 의미 , d 리스트의 각각의 인덱스 -> 화폐 값 의미
d[0]=0

for i in range(N):
    for j in range(array[i],M+1): # 화폐금액~입력금액
        if d[j-array[i]]!=100001: # 화폐금액에서 단위 금액을 뺀 금액에서의 OPS값이 존재한다면
            d[j]=min(d[j-array[i]]+1,d[j]) #


if d[M]==100001:
    print(-1)
else:
    print(d[M])

'''

# 4. 금광

'''
풀이법

각 칸에서의 OPS -> 현재 위치 + max(위에서 대각선 아래로 왔을 때 이득, 옆에서 왔을때 이득, 아래에서 위로 왔을 때 이듯) -> 각 칸에서 최적해 구한 후 max(마지막열) 찾기

* 첫 행: 위에서 대각선 아래로 올 수 없음 / 마지막 행: 아래에서 대각선 위로 올 수 없음 *

'''
# Youtube 조금 다르게 만듬

'''
N,M=map(int,input().split())  # N: 행길이, M: 열길이
array=list(map(int,input().split()))

dp=[] # DP테이블

# DP테이블 2차원 배열로 초기화
index=0
for i in range(N):
    dp.append(array[index:index+M])
    index+=M

for j in range(1,M):
    for i in range(N):
        # print(dp[i][j])

        if i==0: # 첫번째 행
            val=dp[i][j] # 원래 값
            dp[i][j]=val+max(dp[i][j-1],dp[i+1][j-1])

        elif i==2: # 마지막 행
            val=dp[i][j]
            dp[i][j]=val+max(dp[i][j-1],dp[i-1][j-1])

        else:
            val=dp[i][j]
            dp[i][j]=val+max(dp[i][j-1],dp[i+1][j-1],dp[i-1][j-1])


# 마지막 열 최대값 도출
max_val=0
for j in range(M):
    max_val=max(dp[N-1][j],max_val)

print(max_val)

'''

# 18353 병사 배치하기

'''
풀이법

LIS 알고리즘 이용 (*리스트 거꾸로 뒤집고 연산*)

1. 처음 dp테이블의 모든 값 1초기화 -> 자기 자신으로도 하나 정렬되기 때문

2. 0<=j<i에 대하여 D[i]=max(D[i],D[j]+1) -> if array[j]<array[i] 수행

3. 전체 수 - max(D) -> 열외 시키는 병사 수

'''

'''
N=int(input())
array=list(map(int,input().split()))

# LIS: 작은 숫자 ~ 큰 숫자 -> 좌우 반전 시키고 LIS 적용
array.reverse()

# DP 테이블 생성
D=[1]*N

for i in range(1,N): # 첫 번째 값: 1 생략
    for j in range(0,i): # 0~i-1 인덱스 까지
        if array[j]<array[i]: # 오른쪽에 있는 값이 왼쪽에 있는 값보다 크다면 DP 테이블갱신
            D[i]=max(D[j]+1,D[i]) # 앞에 있었던 DP값 들중 가장 큰 숫자와 현재 인덱스에서의 DP값 비교

# 열외한 병사 수 = 전체 숫자 - 최대 LIS 개수
result=N-max(D)
print(result)

'''


# 1463 1로 만들기

'''
최솟값 도출 -> dp테이블 입력 최대값으로 초기화, dp[1]=0

각각의 dp테이블의 인덱스 값 -> 자연수 값 의미

각각의 인덱스에서 min(1빼기 연산, 2나누기 연산 or 3나누기 연산(배수일때만 해당)) 갱신

'''

'''
X=int(input())


d=[100000]*(X+1)
d[1]=0

for i in range(2,X+1):
    d[i]=d[i-1]+1

    if i%2==0:
        d[i]=min(d[i],d[i//2]+1)

    if i%3==0:
        d[i]=min(d[i],d[i//3]+1)

print(d[X])

'''

# 9095 1,2,3 더하기

'''
방법의 수 = 최대 경우의 수

직관적으로 열거 -> i=>4 일때, d[i]=d[i-1]+d[i-2]+d[i-3]

'''

'''
n = int(input())

arr = [0] * 11
arr[1] = 1
arr[2] = 2
arr[3] = 4

for i in range(4,11):
    arr[i] = arr[i-1] + arr[i-2] + arr[i-3]

for i in range(0,n):
    testNum = int(input())
    print(arr[testNum])

'''


# 11726 2×n 타일링

'''
n=int(input())
d=[0]*1001

d[1]=1
d[2]=2
d[3]=3


for i in range(3,n+1):
    d[i]=(d[i-1]+d[i-2])%10007

print(d[n])

'''

# 11722 가장 긴 감소하는 부분 수열

'''
N=int(input())
array=list(map(int,input().split()))
array.reverse()
dp=[1]*1000

for i in range(1,N):
    for j in range(i):
        if array[j]<array[i]:
            dp[i]=max(dp[j]+1,dp[i])

print(max(dp))

'''

# 2579 계단 오르기(노션 정리)

'''
풀이법

도착했을 때를 기준으로 생각하기

마지막 계단(stair[n])에 오를 수 있는 방법

1. 한 칸으로 -> stair[n]+stair[n-1]+]+dp[n-3]
2. 두 칸으로 -> stair[n]+dp[n-2]

위에서 부터 각 칸에 대해서 max(1,2) 선택하는 방법으로 각계단에서의 OPS 찾아감

첫번째, 두번째, 세번째 계단 -> 초기화 : 점화식에서 n-2까지 정보 필요

# dp테이블
dp[1]: stair[1]
dp[2]: stair[1]+stair[2]
dp[3]: stair[3]+max(dp[1],dp[2])

참고) https://v3.leedo.me/devs/64

'''

'''
import sys
input=sys.stdin.readline

N=int(input())
lst=[0]*301

for i in range(1,N+1):
    lst[i]=int(input())


# 첫번째 계단과 DP테이블 편의상 맞춤
dp=[0]*301

dp[1]=lst[1]
dp[2]=lst[1]+lst[2]


# 3번째 계단부터 선택지 생김
dp[3]=max(lst[1]+lst[3],lst[2]+lst[3])


for i in range(4,N+1):
    dp[i]=max(lst[i]+lst[i-1]+dp[i-3],lst[i]+dp[i-2])

print(dp[N])

'''


# 1965 상자넣기

'''
# LIS문제와 동일

N=int(input())
array=list(map(int,input().split()))


# DP 테이블 생성
D=[1]*N

for i in range(1,N): # 첫 번째 값: 1 생략
    for j in range(0,i): # 0~i-1 인덱스 까지
        if array[j]<array[i]: # 오른쪽에 있는 값이 왼쪽에 있는 값보다 크다면 DP 테이블갱신
            D[i]=max(D[j]+1,D[i]) # 앞에 있었던 DP값 들중 가장 큰 숫자와 현재 인덱스에서의 DP값 비교

result=max(D)
print(result)

'''

# 2565 전깃줄 -> 아이디어 구상하다 포기

'''
풀이법 _ 내 생각

1. 문제 상황

A: ai~aj(인덱스)
B: bi~bj

->합선이 일어날때 : ai<bi 이지만 aj>bj

모든 전깃줄이 서로 교차하기 않게 하기 위해 없애는 최소 방법
-> 많이 겹쳐진것 부터 빼기



2. 해결 방법

첫째,A 전봇대의 각각의 인덱스에 합선 개수 저장

둘째, MAX부터 차례대로 빼주면서 합선이 없는지 검사

'''


# 2193 이친수

'''
1 -> 1

1 0 -> 1

1 0 1
1 0 0  -> 2

1 0 0 1
1 0 0 0
1 0 1 0  -> 3

1 0 0 0 0
1 0 0 0 1
1 0 0 1 0
1 0 1 0 0
1 0 1 0 1 -> 5


1 0 0 0 0 1
1 0 0 0 1 0
1 0 0 1 0 0
1 0 1 0 0 0
1 0 1 0 1 0
1 0 0 1 0 1
1 0 1 0 0 1
1 0 1 0 1 0 -> 8


'''

'''
N=int(input())

dp=[0]*(N+1)
dp[1]=1



for i in range(2,N+1):
    dp[i]=dp[i-1]+dp[i-2]

print(dp[N])

'''



# 14501 퇴사

'''
현재 날짜 i: 1~7
이전 날짜 j: 0~i-1

if j(이전 날짜) + T[j](걸리는 시간) <= i(현재 날짜)

-> dp[i] = max(dp[j]+P[i], dp[i]) : 이전 날짜에서의 최대 이득 + 현재 날짜 이득, 현재 날짜 이득

'''

'''
import sys
input=sys.stdin.readline

N=int(input())
array=[]

for i in range(N):
    array.append(list(map(int,input().split())))

dp=[0]*(N+1)

for i in range(1,N+1):
    for j in range(i):
        if j+array[j][0]<=i:  # 시작하는 날 + 걸리는 시간이 다음 <= 다음 날짜
            dp[i]=max(dp[j]+array[j][1],dp[i])

print(dp)

'''

# 17626 Four Squares -> 실패

'''
내생각 -> 무한루프


N=int(input())
cnt=0

while N!=0:

    num=0
    FS=0 # 제곱수

    while FS<N:

        num+=1
        FS=num*num


    FS=(num-1)*(num-1)
    cnt+=1
    N-=FS

'''


# 14916 거스름돈

'''
N=int(input())

d=[100000]*(N+1)
d[0]=0

array=[2,5]

for i in range(len(array)):
    for j in range(array[i],N+1):
        if d[j-array[i]]!=100000:
            d[j]=min(d[j-array[i]]+1,d[j])

if d[N]==100000:
    print(-1)
else:
    print(d[N])

'''


# 9657 돌 게임 3

'''

1 상 -> 상
2 상찬 -> 찬
3 상상상 -> 상
4 상상상상 -> 상
5 상상상찬상 -> 상


완벽히 게임을 했다
-> 큰 숫자가 이득이 있지는 않다
-> 규칙성 존재


게임에서 이기는 방법

N-1, N-3, N-4 -> 2 : 상대방은 2개를 못뽑으므로 하나 뽑고 자신에게 기회가 돌아옴


참고 ) https://velog.io/@gkska741/BOJ-9657%EB%8F%8C%EA%B2%8C%EC%9E%84Python

'''

'''
N=int(input())

dp=[0,'W','L','W','W'] + [0]*(N-4)

# L을 보낼 수 있으면 -> W, 보낼 수 없다면 -> L
# 1,3,4 중 가져갔을 때 남은 개수가 2개(L)이여야 이길 수 있음


if N<=4:
    pass
else:
    for i in range(5,N+1):
        # SK기준
        if 'L' in [dp[i-1], dp[i-3], dp[i-4]]:
            dp[i]='W'
        else:
            dp[i]='L'

if dp[N]=='W':
    print('SK')
else:
    print('CY')

'''

# 11048 이동하기

'''
풀이법

이동 방향: 아래, 오른쪽, 오른쪽 아래

(1,1)에서 부터 각각의 좌표 -> d[i][j] = max(d[i-1][j-1],d[i][j-1],d[i-1][j]) -> 최대값 갱신

DP테이블 : d[0][o]=arr[0][0], 가로 누적, 세로 누적

'''

'''
N,M=map(int,input().split())

array=[]

for i in range(N):
    array.append(list(map(int,input().split())))

# DP테이블 초기화

d=[[0]*M for j in range(N)]

d[0][0]=array[0][0]
row_sum=array[0][0]
col_sum=array[0][0]

for i in range(1,M):
    row_sum+=array[0][i]
    d[0][i]=row_sum

for j in range(1,N):
    col_sum+=array[j][0]
    d[j][0]=col_sum


# DP테이블 갱신

for i in range(1,N):
    for j in range(1,M):
        d[i][j]=array[i][j]+max(d[i-1][j-1],d[i][j-1],d[i-1][j])

print(d[N-1][M-1])

'''




# 11052 카드 구매하기

'''

풀이법

각 인덱스에서 OPS 도출

**
dp[i]: 카드 i개를 구매하는 최대 가격, p[j]: k개가 들어있는 카드팩의 가격

카드 i개를 구매하는 최대 비용

-> d[i] = max(p[1]+dp[i-1], p[2]+dp[i-2] ,,,,, p[i]+dp[0], d[i])




참고) https://infinitt.tistory.com/250

'''

'''
N=int(input())
p=[0]+list(map(int,input().split()))

d=[0]*(N+1)
d[1]=p[1]

for i in range(2,N+1):
    for j in range(1,i+1): # j=i일때 -> p[j] + d[0] : N개 통째 가격 + d[0]
        d[i]=max(p[j]+d[i-j],d[i])

print(d[N])


'''



# 16194 카드 구매하기2

'''
N=int(input())
p=[0]+list(map(int,input().split()))

d=[10000]*(N+1)
d[0]=0
d[1]=p[1]

for i in range(2,N+1):
    for j in range(1,i+1): # j=i일때 -> p[j] + d[0] : N개 통째 가격 + d[0]
        d[i]=min(p[j]+d[i-j],d[i])

print(d[N])

'''

# 2156 포도주 시식

'''
풀이법

계단 오르기 문제랑 유사

- 연속해 있는 3잔 마시기 x

**
p: 포도주 양, d: 각 인덱스에서 OPS

연속해서 3개를 못마실 때, i 번째 포도주에서 OPS 구하는 방법 (if, i=5 가정)

(1) d[i-3]+p[i-1]+p[i]   (5,4,(2) , (n):n까지의 최적해)

(2) d[i-2]+p[i] (5,(3))

(3) d[i-1]: i번째 포두주를 안먹는 경우

-> d[i] = max((1), (2), (3))


d 테이블 초기화: i-3까지 초기화

d[1]=p[1]
d[2]=p[1]+p[2]

**
d[3]=max(p[1]+p[3],p[2]+p[3],p[1]+p[2])

-> 계단 오르기 문제와의 차이점: 마지막 계단을 반드시 밟아야 했지만, 해당 문제에서는 안마셔도 됨



'''

'''
**

실패 코드(런타임 에러)

import sys
input=sys.stdin.readline

N=int(input())
p=[0]*(N+1)

for i in range(1,N+1):
    p[i]=int(input())



# dp테이블 초기화
d=[0]*(N+1)
d[1]=p[1]
d[2]=p[1]+p[2]
d[3]=max(p[1]+p[3],p[2]+p[3],p[1]+p[2])


# 점화식
for i in range(4,N+1):
    d[i]=max(d[i-3]+p[i-1]+p[i],d[i-2]+p[i],d[i-1])

print(d[N])



실패한 이유: 런타임에러(INDEX)

-> N이 1,2일때 dp테이블 따로 처리

참고) https://velog.io/@yj_lee/%EB%B0%B1%EC%A4%80-2156%EB%B2%88-%ED%8F%AC%EB%8F%84%EC%A3%BC-%EC%8B%9C%EC%8B%9D-%ED%8C%8C%EC%9D%B4%EC%8D%AC

-> N=1,2일때 따로 IF문으로 처리

'''


'''
import sys
input=sys.stdin.readline

N=int(input())
p=[0]*(N+1)

for i in range(1,N+1):
    p[i]=int(input())



# dp테이블 초기화
d=[0,p[1]]

if N==1:
    print(d[1])

if N==2:
    d.append(p[1]+p[2])
    print(d[2])

if N>=3:
    # d[2]
    d.append(p[1]+p[2])
    # d[3]
    d.append(max(p[1]+p[3],p[2]+p[3],p[1]+p[2]))


    for i in range(4,N+1):
        d.append(max(d[i-3]+p[i-1]+p[i],d[i-2]+p[i],d[i-1]))

    print(d[N])

'''

# 9251 LCS
# 재귀 -> 깊이 초과
'''
def lcs(n,m,cnt):
    if n<0 or m<0:
        return cnt

    if lst1[n]==lst2[m]:
        return lcs(n-1,m-1,cnt+1)
    else:
        return max(lcs(n-1,m,cnt),lcs(n,m-1,cnt))

lst1=input()
lst2=input()

print(lcs(len(lst1)-1,len(lst2)-1,0))

'''
# dp
'''
lst1=list(map(str,input()))
lst2=list(map(str,input()))

def lcs(x,y):
    x,y=[' ']+x, [' ']+y
    m,n=len(x),len(y)
    c=[[0 for _ in range(n)] for _ in range(m)]

    for i in range(1,m):
        for j in range(1,n):
            if x[i]==y[j]:
                c[i][j]=c[i-1][j-1]+1
            else:
                c[i][j]=max(c[i-1][j],c[i][j-1])

    return c[m-1][n-1]

print(lcs(lst1,lst2))

'''
# 2579 계단 오르기
'''
n번째 계단을 밟을 때 최대 점수
- dp[2칸전]+현재 점수
- dp[3칸전]+한칸전 점수+현재 점수

lst, dp 테이블 크기 설정이 중요함, n이 3보다 작을 경우, initial value 설정에설 인덱스 에러
'''
'''
N=int(input())
lst=[0]*(301)
for i in range(N):
    lst[i+1]=int(input())

# dp initial value
dp=[0]*301
dp[1]=lst[1]
dp[2]=lst[1]+lst[2]
dp[3]=max(lst[1]+lst[3],lst[2]+lst[3])

# dp테이블 갱신
for i in range(4,N+1):
    dp[i]=max(dp[i-2]+lst[i],dp[i-3]+lst[i-1]+lst[i])

print(dp[N])

'''

# 1149 RGB거리
'''
세로 길이:3 고정 (N아님)

N=int(input())
arr=[list(map(int,input().split())) for _ in range(N)]

dp=[[0]*3 for _ in range(N)]
dp[0]=arr[0]

for i in range(1,N):
        dp[i][0] = arr[i][0] + min(dp[i - 1][1], dp[i - 1][2])
        dp[i][1] = arr[i][1] + min(dp[i - 1][0], dp[i - 1][2])
        dp[i][2] = arr[i][2] + min(dp[i - 1][0], dp[i - 1][1])


print(min(dp[-1]))

'''

# 2839 설탕 배달
'''
N=int(input())

# 최솟값을 구해야하므로 비교 연산 MIN 사용 -> 시작 테이블 max_val 초기화
INF=1e9
dp=[INF]*(N+1)

# 0을 만드는 방법 - 0
dp[0]=0

# 설탕 봉투 사이즈
pocket=[3,5]

# 봉투 사이즈 차례로 순회
for p in pocket:
    # 사이즈 값 MIN 갱신
    for i in range(p,N+1):
        if dp[i-p]!=INF:
            dp[i]=min(dp[i-p]+1,dp[i])

# N번째 인덱스가 INF면 -1 아니면 값 출력
ans=-1 if dp[N]==INF else dp[N]
print(ans)


'''



# 14501 퇴사

'''
dp풀이 1

- 해당 날짜와 이전 날짜들을 매번 비교
- 첫번째 날짜부터 접근

dp 테이블 갱신 조건
조건1: 현재 날짜에서 날짜를 추가했을 때(일을 한다면) 퇴사날 전까지 마무리 되어야 함   ... 현재 날짜에 대한 조건
조건2: 과거 날짜랑 비교할 때, 과거 날짜에서 날짜를 추가한 값이 현재날짜 이하여야 일이 가능함   ... 과거 날짜와 현재 날짜의 관계 조건


문제점 : for문 2회 사용

'''

'''
N=int(input())

lst=[(0,0)] 
for _ in range(N):
    T,P=map(int,input().split())
    lst.append((T,P))

dp=[0]*(N+1)

for i in range(1,N+1):
    cur_t,cur_p=lst[i]

    for j in range(i):
        pre_t,pre_p=lst[j]

        # 현재 하려고 하는 일을 진행했을 때 마감기간이 아웃되지 않고, ㅁ
        if i+cur_t<=N+1:

            # 과거의 일을 진행했을 때, 겹쳐지지 않는다면 dp갱신
            if j+pre_t<=i:
                dp[i]=max(dp[i],dp[j]+cur_p)

print(max(dp))
'''

'''
dp풀이 2

- dp테이블 end_day(i+cur_p)에 대해서 갱신 수행(과거와 비교할 필요 x)
- 현재 날짜에 대한 dp테이블 값만 초기화 (단, 다음날로 현재 날짜까지의 dp테이블 최고값을 넘겨주어야 함  >> 안넘겨 준다면 매번 dp테이블이 갱신됨)

'''

'''
N=int(input())

lst=[(0,0)] 
for _ in range(N):
    T,P=map(int,input().split())
    lst.append((T,P))

dp=[0]*(N+2) # 인덱스 에러 회피 (dp[i+1] = max(dp[i+1],dp[i]))

for i in range(1,N+1):
    cur_t,cur_p=lst[i]

    end_day=i+cur_t

    # end_day에 해당되는 dp테이블 값과 현재에서 이득을 취했을 때 값 비교하여 더 높은 값으로 갱신
    if end_day<=N+1:
        dp[end_day]=max(dp[end_day],dp[i]+cur_p)

    if i+1<=N+1:
        # 현재까지의 최대 이익을 다음 날에 전달 (무조건, i+1<=end_day 만족)
        dp[i+1] = max(dp[i+1],dp[i])  

print(max(dp))
'''


'''
dp풀이 3

- 뒤에서 부터 갱신
>> 선택하는 경우와 안하는 경우로 구분하여 더 큰 값으로 dp갱신 (이때, 일을 수행했을 때 날짜가 마감일 이하인지 확인해야함)  ... 약간 재귀호출이랑 dfs랑 비슷한 느낌임 
>> 이점 : 1중 for문 + 간단함 

'''
'''
N=int(input())

lst=[] 
for _ in range(N):
    T,P=map(int,input().split())
    lst.append((T,P))

dp=[0]*(N+1)

for i in range(len(lst)-1,-1,-1):
    t,p=lst[i]

    if i+t>N:
        dp[i]=dp[i+1] # 선택 못함 = 다음 날짜 dp값 그대로 받아옴
    
    else:
        dp[i]=max(dp[i+1],dp[i+t]+p)  # dp[i+1]: 선택하지 않았을 때, dp[i+t]+t: 선택했을 때


print(dp[0])  # 뒤에서 부터 접근하면서 최댓값을 갱신 받아오기 때문에, 첫번째 인덱스에 최대값(max(dp))이 담기게 됨
'''


'''
4. DFS 풀이

- 선택한 날과 선택하지 않은 날 완전탐색

'''
'''
# idx -> 날짜, sm -> 현재까지 받아온 이득
def dfs(idx,sm):
    global ans

    # 퇴사일 초과 
    if idx>N:
        return 
    
    # 퇴사일이면
    if idx==N:
        ans=max(ans,sm)
        return
    
    # 마지막 날에서 최대이득이 아닐 수 있음
    ans=max(ans,sm)
    
    dfs(idx+lst[idx][0],sm+lst[idx][1])
    dfs(idx+1,sm)


N=int(input())

lst=[] 
for _ in range(N):
    T,P=map(int,input().split())
    lst.append((T,P))

ans=0
dfs(0,0)
print(ans)
'''

'''
실행 속도: 1>2=3>4

'''