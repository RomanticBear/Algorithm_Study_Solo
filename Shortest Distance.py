# 이코테 Youtube

# 1. 기본 다익스트라 알고리즘

'''
import sys
input=sys.stdin.readline
INF=int(1e9) # 10억 -> 무한 의미


n,m=map(int,input().split())    # 노드의 개수, 간선의 개수 입력

start=int(input())  # 시작 노드 번호 입력

graph=[[] for i in range(n+1)]  # 노드 경로 정보 그래프 입력

visited=[False]*(n+1)   # 방문 여부 체크목적 리스트

distance=[INF]*(n+1)    # 최단 거리 테이블 선언, 무한으로 초기화


# 노드 간선 정보 입력
for _ in range(m):
    a,b,c=map(int,input().split())
    graph[a].append((b,c)) # a노드에서 b노드로 가는 비용이 c


# 방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호 반환 함수
def get_smallest_node():
    min_val=INF
    index=0 # 가장 최단 거리가 짧은 노드 번호
    for i in range(1,n+1):
        if distance[i]<min_val and not visited[i]:
            min_val=distance[i]
            index=i

    return index


# 다익스트라 알고리즘
def dijkstra(start):
    # 시작 노드 초기화
    distance[start]=0
    visited[start]=True

    # 인접 노드 거리 갱신
    for j in graph[start]:
        distance[j[0]]=j[1]//

    # 시작 노드를 제외한 n-1개 노드에 대해 반복
    for i in range(n-1):
        now=get_smallest_node()
        visited[now]=True

        for j in graph[now]:
            cost=distance[now]+j[1] # j[0]->도착 노드번호, j[1]->거리

            # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost<distance[j[0]]:
                distance[j[0]]=cost


# 다익스트라 알고리즘 수행
dijkstra(start)

# 시작노드에서 모든 노드로 가기 위한 최단 거리를 출력
for i in range(1,n+1):
    # 도달할 수 없는 경우
    if distance[i]==INF:
        pirnt("INFINITY")
    # 도달할 수 있는 경우
    else:
        print(distance[i])


'''


# 2. 개선된 다익스트라 알고리즘

'''
import heapq
import sys
input=sys.stdin.readline
INF=int(1e9)


# 노드 및 간선 개수, 시작 노드 번호, 간선 정보 그래프, 거리 정보 리스트 선언
n,m=map(int,input().split())
start=int(input())
graph=[[] for i in range(n+1)]
distance=[INF]*(n+1)


# 노드 간선 정보 입력
for _ in range(m):
    a,b,c=map(int,input().split())
    graph[a].append((b,c))


def dijkstra(start):
    q=[]
    heapq.heappush(q,(0,start)) # (힙리스트, (가치, 노드))
    distance[start]=0

    # 큐가 비어질 때까지 반복
    while q:
        dist, now = heapq.heappop(q) # 가장 짧은 노드 정보 꺼내기

        if distance[now]<dist: # 현재 distance에 저장된 값이 더 작다면 (=처리된 노드라면) -> 무시
            continue

        else:
            for j in graph[now]:
                cost=dist+j[1]
                
                # 해당 노드를 거쳐 가는 비용이 기존에 저장된 비용보다 적은 경우 -> distance 테이블 갱신, heapq에 추가
                if cost<distance[j[0]]:
                    distance[j[0]]=cost
                    heapq.heappush(q,(cost, j[0]))


dijkstra(start)

for i in range(1,n+1):
    if distance[i]==INF:
        print("INFINITY")
    else:
        print(distance[i])

'''

# 플루이드 워셜 알고리즘

'''
INF=int(1e9)

n,m=map(int,input().split())

# 2차원 리스트 생성 및 INF 초기화
graph=[[INF]*(n+1) for i in range(n+1)]

# 자기 자신에서 자기 자신으로 가는 비용 0 초기화
for a in range(1,n+1):
    for b in range(1,n+1):
        if a==b:
            graph[a][b]=0


# 각 간선에 대한 정보를 입력, 초기화
for _ in range(m):
    # A에서 B로 가는 비용 -> C 설정
    a,b,c=map(int,input().split())
    graph[a][b]=c

# 플루이드 워셜 알고리즘 수행
for k in range(1,n+1):
    for a in range(1,n+1):
        for b in range(1,n+1):
            graph[a][b]=min(graph[a][b],graph[a][k]+graph[k][b])


# 수행 결과 출력
for a in range(1,n+1):
    for b in range(1,n+1):
        if graph[a][b]==INF:
            print("INFINITY",end=" ")
        else:
            print(graph[a][b], end=" ")
    print()

'''

########################################################################

# 실전 문제 _ 미래 도시 p.259
# 플루이드 워셜 알고리즘

'''
INF=int(1e9) # 10억 -> 무한 비용
n,m=map(int,input().split()) # n -> 노드개수, m -> 간선개수

graph=[[INF]*(n+1) for _ in range(n+1)]

# 자신 노드 -> 자신 노드 : 0 초기화
for a in range(1,n+1):
    for b in range(1,n+1):
        if a==b:
            graph[a][b]=0


# 간선 정보 입력
for _ in range(m):
    a,b=map(int,input().split())
    graph[a][b]=1
    graph[b][a]=1


# 도착 노드 X, 거쳐 가는 노드 K 입력
x,k=map(int,input().split())


# 플루이드 워셜 알고리즘 실행
for k in range(n+1):
    for a in range(n+1):
        for b in range(n+1):
            graph[a][b]=min(graph[a][b],graph[a][k]+graph[k][b])

# 결과 출력
distance=graph[1][k]+graph[k][x]

if distance>=INF:
    print("-1")
else:
    print(distance)

'''



# 실전 문제 _ 전보 p.262
# 다익스트라 알고리즘 -> 최소 힙 이용

'''
import sys
import heapq
input=sys.stdin.readline


# 무한 비용
INF=1e9

# N:노드개수, M:간선개수, C:시작노드
n,m,c=map(int,input().split())

# 최소 비용 저장 리스트
distance=[INF]*(n+1)

# 간선 정보 저장 그래프
graph=[[] for i in range(n+1)]


# 간선 정보 입력
for i in range(m):
    # X:출발, Y:도착, Z:비용
    x,y,z=map(int,input().split())
    graph[x].append((y,z))


# 다익스트라 알고리즘
def dijkstra(start): # start -> C

    q=[] # 힙
    heapq.heappush(q,(0,start))
    distance[start]=0

    while q:
        dist,now=heapq.heappop(q) # 가장 짧은 노드 정보 추출

        if distance[now]<dist: # 뽑은 노드까지 가는 비용이 distance 테이블에 저장된 노드까지의 비용보다 큰 경우 -> 탈락
            continue

        else:
            for j in graph[now]:
                cost=dist+j[1]

                if cost<distance[j[0]]:
                    distance[j[0]]=cost
                    heapq.heappush(q,(cost,j[0]))


# 시작노드로 부터 다익스트라 알고리즘 수행         
dijkstra(c)

# 도달개수
count=0

# 최대거리
max_dist=0

for i in range(1,n+1):
    if distance[i]!=INF:
        count+=1
        max_dist=max(max_dist,distance[i])

print(count-1,max_dist)
                                  
'''                            
                


########################################################################

# 백준 최단경로

# 경로 찾기 11430

'''
풀이법

플루드 워셜 알고리즘 적용 -> distance table 인덱스 값이 INF->0, 아니라면 ->1 출력
원소의 개수가 100개 미만 -> 시간 복잡도 ok

'''

'''
INF=1e9

# 노드 개수
n=int(input())


# 간선 정보 입력
graph=[]
for i in range(n):
    graph.append(list(map(int,input().split())))


# 간선 정보를 제외한 간선 비용 -> INF
# 문제 -> 자기 자신으로 가는 비용 -> *INF 
for i in range(n):
    for j in range(n):
        if graph[i][j]==0:
            graph[i][j]=INF

# 플루드 워셜 알고리즘
for k in range(n):
    for i in range(n):
        for j in range(n):
            graph[i][j]=min(graph[i][j],graph[i][k]+graph[k][j])

# INF라면->0, INF아니라면->1
for i in range(n):
    for j in range(n):
        if graph[i][j]==INF:
            print(0, end=" ")
        else:
            print(1, end=" ")
    print()

'''


# 1753 최단경로

'''
풀이법
-> 최소 힙을 이용한 다익스트라알고리즘 수행
'''

'''
import sys
import heapq
input=sys.stdin.readline

INF=1e9 # 무한 비용

n,m=map(int,input().split())
start=int(input())

# 그래프 생성
graph=[[] for j in range(n+1)]

# 간선정보 입력
for i in range(m):
    u,v,w=map(int,input().split())
    graph[u].append((w,v)) # 노드 u에서 노드 v만큼 이동하는 비용 -> w

# 최단거리 테이블
distance=[INF]*(n+1)


# 다익스트라
def dijkstra(start):
    q=[]
    heapq.heappush(q,(0,start))
    distance[start]=0

    while q:
        dist,now=heapq.heappop(q) # dist->v(비용), now->v(도착 노드)

        # 최단거리를 계산한 노드 -> dist 비용이 distance테이블에 저장된 비용보다 큼 -> continue
        if dist>distance[now]:
            continue

        # 아니라면 -> 인접 노드의 비용따져서 distance 테이블 갱신 -> heap에 추가
        else:
            for j in graph[now]:
                cost=dist+j[0]
                if cost<distance[j[1]]:
                    distance[j[1]]=cost
                    heapq.heappush(q,(cost,j[1]))

dijkstra(start)


for i in range(1,n+1):
    if distance[i]==INF:
        print("INF")
    else:
        print(distance[i])
    
'''          
            

# 11404 플로이드


'''
풀이법

- 최소힙을 이용한 다익스트라 알고리즘
- 시작노드가 모든 도시 1회전

'''

'''
import sys
import heapq
input=sys.stdin.readline

INF=1e9

# n->노드개수, m->간선개수
n=int(input())
m=int(input())

# 간선 정보 저장을 위한 graph 선언
graph=[[]*(n+1) for _ in range(n+1)]


# 간선 정보 입력
for i in range(m):
    a,b,c=map(int,input().split()) # a노드에서 b노드로 이동하는 비용 -> c
    graph[a].append((c,b))

# 최단 거리 저장 리스트
distance=[INF]*(n+1)

# 다익스트라 알고리즘
def dijkstra(start):
    q=[]
    heapq.heappush(q,(0,start))
    distance[start]=0

    while q:
        dist,now=heapq.heappop(q)

        if dist>distance[now]:
            continue
        else:
            for j in graph[now]:
                cost=dist+j[0]
                if cost<distance[j[1]]:
                    distance[j[1]]=cost
                    heapq.heappush(q,(cost,j[1]))

# 각 노드에 대해서 다익스트라 수행 -> 결과값 출력
for k in range(1,n+1):
    dijkstra(k)
    
    for i in range(1,n+1):
        if distance[i]==INF:
            print(0,end=" ")
        else:
            print(distance[i],end=" ")
    distance=[INF]*(n+1)
    print()

'''

# 1916 최소비용 구하기

'''
import sys
import heapq
input=sys.stdin.readline

INF=1e9

# n->노드개수, m->간선개수
n=int(input())
m=int(input())

# 간선 정보 저장을 위한 graph 선언
graph=[[]*(n+1) for _ in range(n+1)]


# 간선 정보 입력
for i in range(m):
    a,b,c=map(int,input().split()) # a노드에서 b노드로 이동하는 비용 -> c
    graph[a].append((c,b))

start,end=map(int,input().split())

# 최단 거리 저장 리스트
distance=[INF]*(n+1)

# 다익스트라 알고리즘
def dijkstra(start):
    q=[]
    heapq.heappush(q,(0,start))
    distance[start]=0

    while q:
        dist,now=heapq.heappop(q)

        if dist>distance[now]:
            continue
        else:
            for j in graph[now]:
                cost=dist+j[0]
                if cost<distance[j[1]]:
                    distance[j[1]]=cost
                    heapq.heappush(q,(cost,j[1]))

dijkstra(start)
print(distance[end])
    
'''


# 1389 케빈 베이컨의 6단계 법칙

'''
풀이법

노드 100개 미만 -> 플루드 워셜 알고리즘

'''

'''
INF=1e9

n,m=map(int,input().split())

# 거리 정보 저장할 테이블 
graph=[[INF]*(n+1) for i in range(n+1)]

# graph 간선 정보 입력
for _ in range(m):
    a,b=map(int,input().split())
    graph[a][b]=1
    graph[b][a]=1


# 플루드 워셜 알고리즘
for k in range(1,n+1):
    for i in range(1,n+1):
        for j in range(1,n+1):
            graph[i][j]=min(graph[i][j],graph[i][k]+graph[k][j])


person=0 # 관계 총 합이 적은 사람 번호
min_num=INF
sum=0

for i in range(1,n+1):
    for j in range(1,n+1):
        # 친구 관계x -> 넘기기
        if graph[i][j]==INF:
            pass
        # 친구 관계o -> 합 저장
        else:
            sum+=graph[i][j]

    # 가장 적은 비용을 가진 사람이라면 번호 갱신(person), 최소 비용 갱신(min_num)
    if sum<min_num:
        min_num=sum
        person=i
    sum=0
            
print(person)

'''


# 1238 파티 -> 재밌었다 ,,, 1시간 사용

'''
풀이법

최소힙을 이용한 다익스트라 알고리즘

-> 왕복 거리를 저장할 리스트(total_dist) 사용

1. 각각의 집에서 최단 경로 검색(dijkstra(출발지)) -> 파티장까지 가는 최단 경로 추가 : total_dist[출발지]+=distance[도착지]
2. 파티장에서 최단 경로 검색(dijkstra(도착지)) : total_dist(출발지)+=distance(출발지)

'''

'''
import sys
import heapq
input=sys.stdin.readline

INF=1e9


# n:노드개수, m:간선개수, x:도착노드
n,m,x=map(int,input().split())


# 간선 정보 저장 graph 
graph=[[]*(n+1) for _ in range(n+1)]


# 간선 정보 입력
for i in range(m):
    a,b,c=map(int,input().split()) # a노드에서 b노드로 이동하는 비용 -> c
    graph[a].append((c,b))


# 최단 거리 저장 리스트
distance=[INF]*(n+1)

# 다익스트라 알고리즘
def dijkstra(start):
    q=[]
    heapq.heappush(q,(0,start))
    distance[start]=0

    while q:
        dist,now=heapq.heappop(q)

        if dist>distance[now]:
            continue
        else:
            for j in graph[now]:
                cost=dist+j[0]
                if cost<distance[j[1]]:
                    distance[j[1]]=cost
                    heapq.heappush(q,(cost,j[1]))



# 왕복 총 거리를 저장할 리스트
total_dist=[0]*(n+1)

# 편도 비용 (출발점->도착지(x))
for i in range(1,n+1):
    dijkstra(i)
    total_dist[i]+=distance[x] # 파티장까지 가는 거리 추가
    distance=[INF]*(n+1) # 거리 테이블 초기화



# 도착 지점에서 모든지점에 대해 최단경로 검색 -> dijkstra(도착지점)
dijkstra(x)


# 집까지의 최단 거리 total_dist 각각에 더해주기
for i in range(1,n+1):
    total_dist[i]+=distance[i]


# 가장 많은 비용 출력
print(max(total_dist))

'''

# 1261 알고스팟 -> BFS ,, PASS




# 18352 특정 거리의 도시 찾기

# 1389 케빈 베이컨의 6단계 법칙

'''
import sys
import heapq
input=sys.stdin.readline

INF=1e9

# n->노드개수, m->간선개수
n,m,x,k=map(int,input().split())

# 간선 정보 저장을 위한 graph 선언
graph=[[]*(n+1) for _ in range(n+1)]


# 간선 정보 입력
for i in range(m):
    a,b=map(int,input().split()) # a노드에서 b노드로 이동하는 비용 -> c
    graph[a].append((1,b))


# 최단 거리 저장 리스트
distance=[INF]*(n+1)

# 다익스트라 알고리즘
def dijkstra(start):
    q=[]
    heapq.heappush(q,(0,start))
    distance[start]=0

    while q:
        dist,now=heapq.heappop(q)

        if dist>distance[now]:
            continue
        else:
            for j in graph[now]:
                cost=dist+j[0]
                if cost<distance[j[1]]:
                    distance[j[1]]=cost
                    heapq.heappush(q,(cost,j[1]))

dijkstra(k)
sum=0

#print(distance)


for i in range(1,n+1):
    if distance[i]==x:
        sum+=distance[i]
        print(i)
    else:
        pass

if sum==0:
    print(-1)
else:
    pass
    
'''



# 1504 특정한 최단경로

'''
풀이법

3번에 걸쳐 다익스트라 알고리즘 각각 수행 -> 최단 경로 누적

1238 파티 유사

'''
'''
import sys
import heapq
input=sys.stdin.readline

INF=1e9

n,m=map(int,input().split())

# 간선 저장 그래프
graph=[[]*(n+1) for i in range(n+1)]

# 간선 정보 입력
for i in range(m):
    a,b,c,=map(int,input().split()) 
    graph[a].append((c,b)) # a에서 b로 가는 비용 -> c
    graph[b].append((c,a)) # 양방향


# 다익스트라 함수
def dijkstra(start):
    distance=[INF]*(n+1)
    q=[]
    heapq.heappush(q,(0,start))
    distance[start]=0

    while q:
        dist,now=heapq.heappop(q)
        if dist>distance[now]:
            continue
        else:
            for j in graph[now]:
                cost=dist+j[0]
                if cost<distance[j[1]]:
                    distance[j[1]]=cost
                    heapq.heappush(q,(cost,j[1]))
    return distance


v1,v2=map(int,input().split())


st_path=dijkstra(1)
v1_path=dijkstra(v1)
v2_path=dijkstra(v2)


# 1->v1->v2->n
v12_path=st_path[v1]+v1_path[v2]+v2_path[n]
v21_path=st_path[v2]+v2_path[v1]+v1_path[n]

result=min(v12_path,v21_path)
print(result if result<INF else -1)

'''

'''
내 코드 (정답->비효율), 함수에서 distance 반환

# 1->v1->v2->n
total_dist1=0
dijkstra(1)

total_dist1+=distance[v1]
distance=[INF]*(n+1)

dijkstra(v1)
total_dist1+=distance[v2]
distance=[INF]*(n+1)

dijkstra(v2)
total_dist1+=distance[n]
distance=[INF]*(n+1)


# 1->v2->v1->n
total_dist2=0
dijkstra(1)

total_dist2+=distance[v2]
distance=[INF]*(n+1)

dijkstra(v2)
total_dist2+=distance[v1]
distance=[INF]*(n+1)

dijkstra(v1)
total_dist2+=distance[n]


total_dist=min(total_dist1,total_dist2)

if total_dist>=INF:
    print(-1)
else:
    print(total_dist)

'''
    


# 이코테 Youtube
# 벨만 폴드 알고리즘

# 11657 타임머신

'''
import sys
input=sys.stdin.readline
INF=int(1e9)


# n:노드개수, m:간선개수
n,m=map(int,input().split())

# 간선 정보 리스트
edges=[]

# 최단 거리 정보 리스트
dist=[INF]*(n+1)


# 간선 정보 입력받기 -> 엣지 리스트
for _ in range(m):
    a,b,c=map(int,input().split())
    edges.append((a,b,c)) # a노드에서 b노드로 이동하는 비용 -> c


# 벨만 폴드
def bf(start):
    
    # 시작 노드에 대해서 초기화
    dist[start]=0

    # 전체 n번의 업데이트 반복(n-1번까지 -> 최단경로, n번 -> 음수 사이클 판단)
    for i in range(n):
        
        # 모든 간선에 대해 업데이트
        for j in range(m):
            cur=edges[j][0]
            next_node=edges[j][1]
            cost=edges[j][2]
            
            # 현재 간선을 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if dist[cur]!=INF and dist[next_node]>dist[cur]+cost:
                dist[next_node]=dist[cur]+cost

                # n번째 라운드(i=n-1)에서도 값이 갱신된다면 음수 사이클 존재
                if i==n-1:
                    return True
    return False


negative_cycle=bf(1)

if negative_cycle:
    print("-1")

else:
    # 1번 노드제외 다른 노드까지 최단거리 출력
    for i in range(2,n+1):
        if dist[i]==INF:
            print("-1")
        else:
            print(dist[i])

'''


## BFS 복습 후 마저 풀이 
# 1219 오민식의 고민

'''
풀이법

버스 비용 -> 사용하는 비용(-값)


방문 비용 -> 얻는 비용(+값)


N-1번 모든 간선 업데이트 -> 최적 경로 확정
** 문제 **
1. 버스를 사용하는 비용은 음수임으로 무조건 음의 사이클 나옴 -> 변형

2. 일반 벨만포드에서 무제한 사이클의 전제가 음수가 아니라 양수로 변경, 즉 무한으로 돈을 벌 수 있음

3. 




추가 1회 수행 -> 업데이트가 된다면 무한 음수, 안된다면 도착지까지 비용 출력

'''

'''
import sys
input=sys.stdin.readline

# 음의무한
INF=-int(1e9)

# n:도시개수, s:시작도시, e:도착도시, m:간선개수
n,s,e,m=map(int,input().split())

# 간선정보 입력
graph=[]
for _ in range(m):
    a,b,c=map(int,input().split())
    graph.append((a,b,-c))

# 방문한 도시에서 받는 비용
get_money=list(map(int,input().split()))


# 최소비용
dist=[INF]*n


# 벨만포드 알고리즘
def bf(start):
    dist[start]=get_money[start]

    # 총 n번 수행
    for i in range(n):
        # 모든 간선 업데이트
        for j in range(m):
            now_node=graph[j][0]
            next_node=graph[j][1]
            cost=graph[j][2]
            plus=get_money[next_node]

            # 다음 노드 기존 금액 < 현재 노드 비용 + 현재에서 다음노드로 가는 비용 + 도착 비용
            if dist[now_node]!=INF and dist[next_node]<dist[now_node]+cost+plus:
                dist[next_node]=dist[now_node]+cost+plus 

                # n번째에서 업데이트가 발생하였다면 -> 양수 사이클
                if i==n-1:
                    return True

    return False



positive_cycle=bf(s)

# 양수 사이클이지만 도착지에는 관련없는 경우 !!!!!!!!! 비용 출력해야함 !!!!!!!!!!
# n-1번 수행 후, n번째 반복때 만약 거리리스트가 업데이트 된다면 BFS로 확인

if positive_cycle:
    if dist[e]==INF: # 양수 사이클인데 목적지까지 가는 경로 없음
        print('gg')
    else:
        print('Gee') # 목적지 경로 있지만 양수 사이클


        
else:
    if dist[e]==INF: # 양수 사이클 아닌데 경로가 없음
        print('gg')
    else:
        print(dist[e]) # 비용 출력

'''


# 2458 키 순서
# pypy로 제출 안해서 애먹음 ,, 

'''
풀이법

전체 노드 N에서,
자신이 갈 수 있는 노드 -> 자신보다 큰 노드
자신에게 오는 노드 -> 자신보다 작은 노드

판별) 두 개의 합이 N-1인 경우 자신의 위치를 알 수 있음, 아닌 경우 자신의 위치를 알 수 없음


플로이드 워셜 알고리즘 -> 하나의 정점에서 모든 정점까지의 최단 거리를 구할 수 있음

'''

'''
import sys
input=sys.stdin.readline


INF=1e9 # 무한

n,m=map(int,input().split())

graph = [[INF] * (n + 1) for _ in range(n + 1)]


# 대각선 성분 비용 0으로 초기화
for i in range(1,n+1):
    for j in range(1,n+1):
        if i==j:
            graph[i][j]=0

# 간선 정보 입력
for _ in range(m):
    a,b=map(int,input().split())
    graph[a][b]=1



# 플루이드 워셜 알고리즘

for k in range(1,n+1):
    for i in range(1,n+1):
        for j in range(1,n+1):
            graph[i][j]=min(graph[i][j],graph[i][k]+graph[k][j])


# 자기가 몇번째인지 아는 학생
result=0

# 자신에게 오는 비용이 무한이 아닌 노드 개수 + 자신으로부터 가는 비용이 무한이 아닌 노드 개수 -> n-1 -> 출력
for k in range(1,n+1):
    sub_sum=0

    for row in range(1,n+1):
        if graph[row][k]!=0 and graph[row][k]!=INF:
            sub_sum+=1

    for col in range(1,n+1):
        if graph[k][col]!=0 and graph[k][col]!=INF:
            sub_sum+=1
    
    if sub_sum==n-1:
        result+=1

    # print('k:',k,' ', 'sub_sum',sub_sum)

print(result)

'''

# 14938 서강그라운드

'''
import sys
input=sys.stdin.readline


INF=1e9 # 무한

# 지역개수, 수색범위, 간선 개수
n,m,r=map(int,input().split())

# 아이템 값
item_lst=list(map(int,input().split()))

graph = [[lNF] * (n + 1) for _ in range(n + 1)]


# 대각선 성분 비용 0으로 초기화
for i in range(1,n+1):
    for j in range(1,n+1):
        if i==j:
            graph[i][j]=0

# 간선 정보 입력
for _ in range(r):
    a,b,c=map(int,input().split())
    graph[a][b]=c
    graph[b][a]=c # 양방향 간선




# 플루이드 워셜 알고리즘

for k in range(1,n+1):
    for i in range(1,n+1):
        for j in range(1,n+1):
            graph[i][j]=min(graph[i][j],graph[i][k]+graph[k][j])

    

max_sum=0

for k in range(1,n+1):
    sub_sum=0
    for j in range(1,n+1):
        if graph[k][j]<=m:
            sub_sum+=item_lst[j-1]

    if sub_sum>max_sum:
        max_sum=sub_sum

print(max_sum)
        
'''






    


    
