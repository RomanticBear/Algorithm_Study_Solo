N,M=map(int,input().split())
lst=[]
v=[True]*(N+1)


def dfs(sub_lst):
    if len(sub_lst)==M:
        # lst.append(sub_lst[:])
        print(*sub_lst)
        return
    
    for num in range(1,N+1):
        if v[num]:
            v[num]=False
            sub_lst.append(num)
            dfs(sub_lst)
            v[num]=True
            sub_lst.pop()

dfs([])