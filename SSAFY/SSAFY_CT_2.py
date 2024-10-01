

'''
3
3 7
3 2 3
#1 6
5 6
1 2 3 4 5
#2 3
5 10
3 2 1 1 3
#3 4


def dfs(lst):
    global ans

    if len(lst)==2:
        ans.append(lst)
        return

    for i in range(1,N+1):
        dfs(lst+[arr[i]])

T=int(input())
for tc in range(1,T+1):
    N,K=map(int,input().split())
    arr=[0]+list(map(int,input().split()))
    ans=[]
    dfs([])
    ans.sort()
    print(f'#{tc}',sum(ans[K-1]))

'''

# 2
'''
def dfs(lst):
    global res

    if len(lst)==N:
        res.append(lst)
        return

    for i in range(N):
        if v[i]!=1:
            v[i]=1
            dfs(lst+[arr[i]])
            v[i]=0


T=int(input())
for tc in range(1,T+1):
    N=int(input())
    p=list(map(int,input().split()))
    arr=list(map(int,input().split()))
    res=[]
    ans=0
    v=[0]*N
    dfs([])

    for x in res:
        cnt=0
        for i in range(len(x)):
            if abs(x[i]-p[i])<=3:
                cnt+=1
        ans=max(ans,cnt)

    print(f'#{tc}',ans)
'''
'''
3
3
4 8 2
6 10 7
#1 2
4
1 2 3 4
4 3 2 1
#2 4
6
5 8 3 4 2 1
6 3 7 9 5 2
#3 6

'''
'''
시험 제출 답안

T=int(input())
for tc in range(1,T+1):
    N=int(input())
    p = sorted(list(map(int, input().split())))
    arr = sorted(list(map(int, input().split())))
    ans=0
    INF=1e9

    for i in range(len(p)):
        sub=arr.copy()
        for j in range(len(sub)):
            if abs(p[i]-sub[j])<=3:
                sub[j]=INF
                break

        cnt=sub.count(INF)
        ans=max(cnt,ans)

    print(f'#{tc}',ans)

'''

T=int(input())
for tc in range(1,T+1):
    N=int(input())
    p = sorted(list(map(int, input().split())))
    arr = sorted(list(map(int, input().split())))
    ans=0
    INF=1e9

    for i in range(len(p)):
        for j in range(len(arr)):
            if abs(p[i]-arr[j])<=3:
                arr[j]=INF
                break

        cnt=arr.count(INF)
        ans=max(cnt,ans)

    print(f'#{tc}',ans)