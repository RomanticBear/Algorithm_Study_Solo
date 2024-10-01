# 15650 N과 M(1)

'''
n,m=map(int,input().split())
arr=[]


def func():
    if m==len(arr):
        print(*arr)
        return
    
    for i in range(1,n+1):
        if i not in arr:
            arr.append(i)
            func()
            arr.pop()

func()

'''

# 15650 N과 M(2)

'''
n,m=map(int,input().split())
arr=[]


def func():
    if m==len(arr):
        print(*arr)
        return
    
    for i in range(1,n+1):
        if i not in arr:
            if len(arr)==0:
                arr.append(i)
                func()
                arr.pop()
            else:
                if i>max(arr):
                    arr.append(i)
                    func()
                    arr.pop()
            
func()

'''

# 15651 N과 M(3)

'''
n,m=map(int,input().split())
arr=[]

def func():
    if len(arr)==m:
        print(*arr)
        return

    for i in range(1,n+1):
        arr.append(i)
        func()
        arr.pop()

func()

'''

# 15652 N과 M(4)
'''
n,m=map(int,input().split())
arr=[]

def func():
    if len(arr)==m:
        print(*arr)
        return

    for i in range(1,n+1):
        if len(arr)==0:
            arr.append(i)
            func()
            arr.pop()
        else:
            if i>=max(arr):
                arr.append(i)
                func()
                arr.pop()

func()

'''

# 156555 N과 M(6)

'''
n,m=map(int,input().split())
lst=list(map(int,input().split()))
lst.sort()
arr=[]

def func():
    if len(arr)==m:
        print(*arr)
        return

    for x in lst:
        if len(arr)==0:
            arr.append(x)
            func()
            arr.pop()
        else:
            if x not in arr:
                if max(arr)<x:
                    arr.append(x)
                    func()
                    arr.pop()

func()
'''

# 156556 N과 M(7)

'''
n,m=map(int,input().split())
lst=list(map(int,input().split()))
lst.sort()
arr=[]

def func():
    if len(arr)==m:
        print(*arr)
        return

    for x in lst:
        arr.append(x)
        func()
        arr.pop()

func()

'''

# 156557 N과 M(8)
'''
n,m=map(int,input().split())
lst=list(map(int,input().split()))
lst.sort()
arr=[]


def func():
    if len(arr)==m:
        print(*arr)
        return

    for x in lst:
        if len(arr)==0:
            arr.append(x)
            func()
            arr.pop()
        else:
            if max(arr)<=x:
                arr.append(x)
                func()
                arr.pop()

func()

'''

# 156558 N과 M(9)
# 원소로 접근하면 중복되니 인덱스로 접근 -> visited로 인덱스 방문 표시
# 앞의 숫자를 기억하는 변수를 선언하여 다음 재귀문으로 넘겨주어 조건을 걸어줌

'''
n,m=map(int,input().split())
lst=list(map(int,input().split()))
lst.sort()
visited=[False]*len(lst)
arr=[]



def func():
    remember_num=0
    if len(arr)==m:        
        print(*arr)    
        return

    for i in range(len(lst)):
        if visited[i]!=True and remember_num!=lst[i]:
            arr.append(lst[i])
            visited[i]=True
            remember_num=lst[i] # 다음 재귀문에서의 조건을 나타냄
            func()
            arr.pop()
            visited[i]=False

func()

'''

# 15664 N과 M(10)
'''
n,m=map(int,input().split())
arr=list(map(int,input().split()))
arr.sort()

visited=[False]*n
lst=[]

def func():
    if len(lst)==m:
        print(*lst)
        return
    remember_num=0
    for i in range(n):
        if len(lst)==0 and remember_num!=arr[i]:
            lst.append(arr[i])
            visited[i]=True
            remember_num=arr[i]
            func()
            lst.pop()
            visited[i]=True
            
        if visited[i]!=True and arr[i]!=remember_num and arr[i]>=max(lst):
            lst.append(arr[i])
            visited[i]=True
            remember_num=arr[i]
            func()
            lst.pop()
            visited[i]=False

func()

'''

# 15665 N과 M(11)
'''
n,m=map(int,input().split())
arr=list(map(int,input().split()))
arr.sort()
lst=[]


def func():
    if len(lst)==m:
        print(*lst)
        return
    remember_num=0
    for i in range(n):         
        if arr[i]!=remember_num:
            lst.append(arr[i])
            remember_num=arr[i]
            func()
            lst.pop()


func()
'''

# 15666 N과 M(12)

'''
n,m=map(int,input().split())
arr=list(map(int,input().split()))
arr.sort()


lst=[]

def func():
    if len(lst)==m:
        print(*lst)
        return
    remember_num=0
    
    for i in range(n):
        if len(lst)==0 and remember_num!=arr[i]:
            lst.append(arr[i])
            remember_num=arr[i]
            func()
            lst.pop()

            
        if arr[i]!=remember_num and arr[i]>=max(lst):
            lst.append(arr[i])
            remember_num=arr[i]
            func()
            lst.pop()


func()

'''

# 1759 암호 만들기
'''
N,C=map(int,input().split())
arr=list(input().split())
arr.sort() # 오름 차순 정렬
check=['a','e','i','o','u']
v=[]

def dfs(n,s):
    if len(s)==N:
        mo=0
        ja=0
        if s not in v:
            for i in range(len(s)):
                if s[i] in check:
                    mo+=1
                else:
                    ja+=1

        if mo>=1 and ja>=2:
            v.append(s)
            print(s)

        return

    if n==C:
        return

    dfs(n+1,s+arr[n])
    dfs(n+1,s)


dfs(0,'')

'''
# 15649 N과 M (1)
'''
두 개의 dfs로 각 인덱스를 포함시켰을 때, 포함시키지 않았을 때로 재귀호출 하는 경우
-> 오름차순으로 담길 수 밖에 없음
-> 즉, 포함 여부를 쓸때는 가능하지만, 순서 정보가 저장되어야 하는 경우에는 못 사용함!!!!!!!!!!
for문 이용해야 순서바뀐 경우도 뽑아줄 수 있음

'''
'''
N,M=map(int,input().split())
ans=[]
v = [False] * (N + 1)

def dfs(n,lst):
    global ans

    if n==M: # 개수만큼 골랐을 때
        ans.append(lst)
        return

    for i in range(1,N+1):
        if v[i]!=True:
            v[i]=True
            dfs(n+1,lst+[i])
            v[i]=False

dfs(0,[])

for x in ans:
    print(*x)
    
'''

# 15650 N과 M(2)
# 방법1
'''
N,M=map(int,input().split())
v=[0]*(N+1)
ans=[]

def dfs(n,lst):
    global ans

    if n==M:
        ans.append(lst)
        return

    for i in range(1,N+1):
        if len(lst)==0:
            if v[i]==0:
                v[i]=1
                dfs(n+1,lst+[i])
                v[i]=0
        else:
            if lst[-1]<i and v[i]==0:
                v[i]=1
                dfs(n+1,lst+[i])
                v[i]=0


dfs(0, [])

for x in ans:
    print(*x)

'''

# 방법2
'''
N,M=map(int,input().split())
num=[x for x in range(1,N+1)]
ans=[]

def dfs(n,lst):
    if len(lst)==M:
        ans.append(lst)
        return

    if n==N:
        return

    dfs(n+1,lst+[num[n]])
    dfs(n+1,lst)

dfs(0,[])
for x in ans:
    print(*x)

'''

# 15651 N과 M(3)
'''
N,M=map(int,input().split())
v=[0]*(N+1)
ans=[]

def dfs(n,lst):
    global ans

    if n==M:
        ans.append(lst)
        return

    for i in range(1,N+1):
        if len(lst)==0:
            dfs(n+1,lst+[i])
        else:
            if lst[-1]<=i:
                dfs(n+1,lst+[i])


dfs(0, [])

for x in ans:
    print(*x)
'''

# 15663 N과 M(5)
'''
N,M=map(int,input().split())
arr=list(map(int,input().split()))
arr.sort()
v=[0]*N
ans=[]
check=[]

def dfs(n,lst):
    if n==M:
        ans.append(lst)
        return

    prev=0
    for i in range(N):
        if v[i]!=1 and arr[i]!=prev:
            v[i]=1
            prev=arr[i]
            dfs(n+1,lst+[arr[i]])
            v[i]=0


dfs(0,[])
for x in ans:
    print(*x)

'''


# 2961 도영이가 만든 맛있는 음식
'''
NUM=int(input())
arr=[]
ans=1e9
for _ in range(NUM):
    s,b=map(int,input().split())
    arr.append((s,b))


# DFS
def dfs(n,lst,N):
    global ans

    if len(lst)==N:
        s_sum=lst[0][0]
        b_sum=lst[0][1]

        if len(lst)>1:
            for i in range(1,len(lst)):
                s_sum*=lst[i][0]
                b_sum+=lst[i][1]

        ans=min(ans,abs(s_sum-b_sum))
        return

    if n==NUM:
        return

    dfs(n+1,lst+[arr[n]],N)
    dfs(n+1,lst,N)

for N in range(1,NUM+1): # 1개에서 최대 N개까지 재료 선택했을 때
    dfs(0,[],N)

print(ans)

'''

# 사람의 수
N = 4

# 초기 상태에서 각 사람이 가지고 있는 번호
init = [1, 2, 3, 4]

# 정보 교환 순서
order = [2, 4, 3, 1]

# 정보를 교환할 일 수
days = 3

# 각 사람의 초기 정보를 리스트로 설정
info = [[n] for n in init]

# days일 동안 정보 교환 반복
for day in range(days):
    print(f"Day {day + 1} 변화:")
    new_info = [[] for _ in range(N)]

    for i in range(N):
        cur = order[i] - 1
        nxt = order[(i + 1) % N] - 1
        new_info[cur] = sorted(list(set(info[cur] + info[nxt])))

    info = new_info

    # 현재 상태 출력
    for i, inf in enumerate(info):
        print(f"{i + 1}번: {inf}")

    print()  # 빈 줄 출력하여 각 날짜의 결과를 구분
