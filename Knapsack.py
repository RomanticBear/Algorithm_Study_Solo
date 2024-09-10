# Kanpsack Algorithm

'''
# i: 물건 인덱스, W: 총 배낭 무게,  w:무게 인덱스, p:이득 인덱스
def kanpsack(i,W,w,p):
    
    # 아이템이 없거나 남은 배낭 무게가 0일 때, 0 반환
    if i<=0 or w<=0:    
        return 0
    
    # 배낭에 물건을 넣을 수 없는 경우 (i번째 물건을 배낭에 물건을 넣었을 때, 남은 무게(W-w[i])가 음수라면 배낭 넣을 수 없으므로 포함시키지 않았을 때 반환)
    if w[i]>W:  
        return knapsack(i-1,W,w,p)

    # 배낭에 물건을 넣을 수 있는 경우 (i번째 물건을 포함 시키고 이득을 더한것과 포함시키지 않았을 때 이득 중 max값 return)
    else:  
        left=knapsack(i-1,W,w,p) # 포함시키지 않았을 때
        right=knapsack(i-1,W-w[i],w,p)  # 포함시켰을 때
        return max(left,p[i]+right)
        
'''


# 12865 평범한 배낭 
# 재귀로 구현 [시간 초과]
'''
import sys
input=sys.stdin.readline

n,W=map(int,input().split())

w=[0]*(n+1)    # 무게 리스트
p=[0]*(n+1)    # 가치 리스트

# 아이템 무게, 가치 입력
for i in range(1,n+1):
    weight,profit=map(int,input().split())
    w[i]=weight
    p[i]=profit


def knapsack(i,W,w,p):
    if i<=0 or W<0:
        return 0

    if w[i]>W:
        return knapsack(i-1,W,w,p)
    else:
        left=knapsack(i-1,W,w,p)
        right=knapsack(i-1,W-w[i],w,p)
        return max(left,right+p[i])

print(knapsack(n,W,w,p))

'''

# DP 테이블 사용 [2차원]
# https://hongcoding.tistory.com/50

'''
N,W=map(int,input().split())    # N: 아이템 개수, W: 배낭 무게 제한

item=[[0,0]]   # 아이템 정보 2차원 리스트
for _ in range(N):
     item.append(list(map(int,input().split())))

# DP 테이블 (행:N+1, 열:W+1) 
dp=[[0 for _ in range(W+1)] for _ in range(N+1)]


for i in range(1,N+1):
    for j in range(1,W+1):  # j: weight 범위(0~W)
        wt=item[i][0]
        val=item[i][1]

        if j<wt:    # (j-wt)가 음수인 경우 넣을 수 없음 -> 넣지 않기
            dp[i][j]=dp[i-1][j]
            
        else:   # 넣을 수 있는 경우 -> 넣고 이득 더한 값과 넣지 않았을 때 값 비교 -> Max값 
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-wt]+val)
        
        

print(dp[N][W])

'''

# DP 테이블 사용 [1차원]
# https://codingmovie.tistory.com/48

'''
N,K=map(int,input().split())
item=[]

for _ in range(N):
    w,v=map(int,input().split())
    item.append((w,v))

dp=[0]*(K+1)

for i in range(N):
    w=item[i][0]
    v=item[i][1]
    for j in range(K,w-1,-1):
        dp[j]=max(dp[j],dp[j-w]+v)

print(dp[-1])

'''

# 17845 수강 과목
'''
import sys
input=sys.stdin.readline

K,N=map(int,input().split())

# 과목 정보
items=[]

for _ in range(N):
     v,w = map(int,input().split())
     items.append((v,w)) 

# dp 테이블
dp=[0]*(K+1)

for item in items:
     v,w=item
     for j in range(K,w-1,-1):
          dp[j]=max(dp[j],dp[j-w]+v)

print(dp[-1])
     
'''



# 9084 동전
'''
t=int(input())

for tc in range(t):
     n=int(input())
     coins=list(map(int,input().split()))
     money=int(input())

     dp=[0]*(money+1)
     dp[0]=1   # 0원 만드는 방법: 한 가지


     for coin in coins:
          for i in range(coin,money+1):
               dp[i]+=dp[i-coin]
                    

     print(dp[money])

'''
     
'''          
   0  1  2  3  4  5  6  7  8  

2  1  0  1  0  1  0  1  0  1

4  1  0  0  0  2  0  2  0  1

'''

# 14728 벼락치기

'''
N,T=map(int,input().split())

items=[]
for _ in range(N):
     w,v=map(int,input().split())
     items.append((w,v))


dp=[0]*(T+1)

for item in items:
     w,v=item
     for j in range(T,w-1,-1):
          dp[j]=max(dp[j],dp[j-w]+v)

print(dp[-1])

'''

# 1106 호텔

'''
ex) 3,1: 비용  5,1: 이득

    0   1   2   3   4   5   6   7   8   9   10  11  12 (가로:인원)

5   0  IN   IN  IN  IN  1   IN  IN  IN  IN   2  IN
                       (3)                  (6)

1  (0) (1)  (2) (3) (4) (5)  (4) (7) (8) (9) (10) (7) (8)
                            (3+1)                (6+1) (6+2) 

(세로: 수용 인원 단위, 가로: 단위당 인원 수) , 값: Weight(최소가 되게하는 것이 목적) -> DP : 초기화 INF -> min 비교


'''
'''
# 최소 C명 이상  -> C명 보다 값이 클 때 비용이 최소가 될 수 있음 -> 리스트 범위 C값만큼 크게 잡아주기

C,N=map(int,input().split())
items=[]

for _ in range(N):
     w,v=map(int,input().split())
     items.append((w,v))


INF=1e7 
dp=[INF]*(C+100)
dp[0]=0


for item in items:
     w,v=item

     for j in range(v,C+100):
          if dp[j-v]!=INF:
               dp[j]=min(dp[j],dp[j-v]+w)
  
     
print(min(dp[C:]))

'''

# 3067 Coins(동전문제와 동일)

'''

    0  1  2  3  4  5  6  7  8

x   1  0  0  0  0  0  0  0  0

2   1  0  1  0  1  0  1  0  1  
 
4   1  0  1  0  2  0  2  0  3

'''
'''
T=int(input())

for _ in range(T):
     n=int(input())
     coins=list(map(int,input().split()))
     money=int(input())

     dp=[0]*(money+1)
     dp[0]=1

     for coin in coins:
          for i in range(coin,money+1):
               if dp[i-coin]!=0:
                    dp[i]=dp[i-coin]+dp[i]


     print(dp[-1])

'''


# 2662 기업투자

'''
Point

- 같은 기업에 여러번 투자 x

가로: 투자 금액(weight)
세로: 투자 단위
값: 이득


   0  1  2  3  4  5  6  7  8  9  10
x  

1

2
   
'''
'''
mon,n=map(int,input().split())
lst1=[]   # A회사
lst2=[]   # B회사

for _ in range(n):
     t,p1,p2=map(int,input().split())
     lst1.append((t,p1))
     lst2.append((t,p2))

max_sm=0  # 금액
ans=[]
for t1,p1 in lst1:
     sm=p1
     for t2,p2 in lst2:
          if t1==t2:
               pass
          else:
               sm+=p2
               if sm>max_sm:
                    max_sm=sm
                    ans.append((t1,t2))
                    

print(max_sm)
print(ans[-1])

'''

# 18427 함께 블록 쌓기

'''
H를 만들 수 있는 숫자 합

* 각각의 학생이 자신의 높이를 더하거나 안더할 수 있음 (일반)
  + 조건: 각각의 학생이 하나가 아닌 여러개의 높이를 가지고 있음   

'''
# 재귀를 이용해서 실패한 코드, 중복 카운트되는데 이유 모르겠음
'''
N,M,H=map(int,input().split())
arr=[]
for _ in range(N):
     arr.append(list(map(int,input().split())))

cnt=0
def dfs(n,sm,v):
     global cnt

     if sm==H:
          print(v)
          cnt+=1
          return

     if n==N:
          return

     for x in arr[n]:
          dfs(n+1,sm+x,v+[(x,n)])
          dfs(n+1,sm,v)

dfs(0,0,[])
print(cnt)

'''

# DP -> 실패
'''
N,M,H=map(int,input().split())
person=[]
for _ in range(N):
     person.append(list(map(int,input().split())))

dp=[[0 for _ in range(H+1)] for _ in range(N+1)]
dp[0][0]=1 # 높이 0을 만드는 방법은 아무것 도 안하는 방법 1개

for i in range(1,N+1):
     p=person[i-1]
     print(p)
     for x in p:
          for j in range(H+1):
               if j==0:
                    dp[i][j]=1
               if j-x>=0:
                    if dp[i-1][j-x]!=0:
                         dp[i][j]=dp[i-1][j]+1
                    else:
                         dp[i][j]=dp[i-1][j]

     print(dp)

print(dp[N][H])

'''






























    
    
