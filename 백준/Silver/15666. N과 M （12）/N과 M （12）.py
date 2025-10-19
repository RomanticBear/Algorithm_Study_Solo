# 15665 N과 M(12)

N,M=map(int,input().split())
lst=sorted(list(map(int,input().split())))
ans=set()

def dfs(idx,sub_lst):
    
    if len(sub_lst)==M:
        ans.add(tuple(sub_lst))
        return
    
    for i in range(idx,len(lst)):
        dfs(i,sub_lst+[lst[i]])


dfs(0,[])

for row in sorted(ans):
    print(*row)