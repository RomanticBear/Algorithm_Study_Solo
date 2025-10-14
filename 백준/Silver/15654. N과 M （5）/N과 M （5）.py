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