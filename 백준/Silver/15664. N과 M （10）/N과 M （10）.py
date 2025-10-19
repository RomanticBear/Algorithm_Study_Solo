# 15664 N과 M(10)

N,M=map(int,input().split())
lst=sorted(list(map(int,input().split())))
ans=set()

def dfs(idx,sub_lst):
    
    if len(sub_lst)==M:
        ans.add(tuple(sub_lst))
        return
    
    if idx==N:
        return

    dfs(idx+1,sub_lst+[lst[idx]])
    dfs(idx+1,sub_lst)


dfs(0,[])

for row in sorted(ans):
    print(*row)