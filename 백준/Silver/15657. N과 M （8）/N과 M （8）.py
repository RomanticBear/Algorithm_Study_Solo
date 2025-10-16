# 15657 N과 M(8)

def dfs(idx,sub_lst):
    if idx>=N:
        return
    
    if len(sub_lst)==M:
        print(*sub_lst)
        return

    for i in range(idx,len(lst)):
        dfs(i,sub_lst+[lst[i]])

N,M=map(int,input().split())
lst=sorted(list(map(int,input().split())))

dfs(0,[])
