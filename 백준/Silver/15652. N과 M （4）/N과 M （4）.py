N,M=map(int,input().split())

def dfs(idx,sub_lst):
    
    if len(sub_lst)==M:
        print(*sub_lst)
        return

    for num in range(idx,N+1):
        dfs(num,sub_lst+[num])

dfs(1,[])
