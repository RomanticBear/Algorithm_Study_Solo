N,M=map(int,input().split())

def dfs(sub_lst):
    
    if len(sub_lst)==M:
        print(*sub_lst)
        return

    for num in range(1,N+1):
        dfs(sub_lst+[num])

dfs([])