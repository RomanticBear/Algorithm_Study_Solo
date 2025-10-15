# dfs -> 재귀로 변경 


from collections import deque

def bfs(sub_lst,CNT):

    # 벽 세우기
    for i,j in sub_lst:
        arr[i][j]=1

    CNT-=3 # 벽 3개 세움
    q=deque()
    w=[[0]*M for _ in range(N)] # BFS 방문 체크 

    for i,j in virus:
        q.append((i,j))
    
    while q:
        i,j=q.popleft()

        for di,dj in [(-1,0),(1,0),(0,1),(0,-1)]:
            ni,nj=i+di,j+dj

            if 0<=ni<N and 0<=nj<M:
                if w[ni][nj]==0 and arr[ni][nj]==0:
                    w[ni][nj]=1 # 방문 체크 
                    q.append((ni,nj)) # 큐 삽입
                    CNT-=1

        
    
    # 벽 해체
    for i,j in sub_lst:
        arr[i][j]=0
    
    return CNT



# main 
N,M=map(int,input().split())
arr=[list(map(int,input().split())) for _ in range(N)]

lst=[] # 빈 공간
virus=[] # 바이러스 공간

for i in range(N):
    for j in range(M):
        if arr[i][j]==0:
            lst.append((i,j))
        elif arr[i][j]==2:
            virus.append((i,j))

ans=0 # 정답
CNT=len(lst)
v=[[0]*M for _ in range(N)] # DFS 방문 체크 

for i in range(CNT-2):
    for j in range(i+1,CNT-1):
        for k in range(j+1,CNT):
            ans=max(ans,bfs([lst[i],lst[j],lst[k]],CNT))


print(ans)
