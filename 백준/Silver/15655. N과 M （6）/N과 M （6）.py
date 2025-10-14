
# 15655 N과 M (6)

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
