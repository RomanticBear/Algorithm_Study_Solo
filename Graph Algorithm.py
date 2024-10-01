# 동빈나 이코테 Youtube
# 그래프 이론

# 서로소 집합 알고리즘

'''
# 비효율적 find 함수
def find_parent(parent, x):
    # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출 -> 루트 노드는 자신의 번호를 초기화 과정에서 자신의 번호를 가지고 있음
    if parent[x]!=x:
        return find_parent(parent, parent[x])
    return x
'''

'''
# find함수
def find_parent(parent,x):
    if parent[x]!=x:
        parent[x]=find_parent(parent,parent[x])
    return parent[x]




# union 연산
def union_parent(parent, a, b):
    a_root=find_parent(parent, a) # a 루트 노드 탐색
    b_root=find_parent(parent, b) # b 루트 노드 탐색

    
    # 이 부분이 가장 헷갈렸음 #####
    # 연결된 A,B의 합이 아니라 A와 B의 루트노드에 관해 합해줌
    if a_root<b_root:
        parent[b_root]=a_root # B'가 A'를 향하게 함 (B'->A')
    else:
        parent[a_root]=b_root




# v:루트노드, e:간선개수
v,e=map(int,input().split())
parent=[0]*(v+1) # 부모 테이블 초기화

# 부모 테이블에서 부모를 자기 자신으로 초기화 -> V개의 트리 형태
for i in range(1,v+1):
    parent[i]=i

# union 연산 모두 수행
for i in range(e):
    a,b=map(int,input().split())
    union_parent(parent,a,b)

# 각 원소가 속한 집합 출력 -> 루트노드 출력 -> 루트 노드가 같다면 같은 집합, 다르다면 다른 집합
print('각 원소가 속한 집합: ', end='')
for i in range(1,v+1):
    print(find_parent(parent,i), end=' ')


# 각각의 부모 노드 출력 -> parent 테이블 = 부모 노드 정보 테이블
print('부모 테이블: ', end='')
for i in range(1, v+1):
    print(parent[i],end=' ')

'''


# 서로소 집합을 이용한 사이클 판별

'''
def find_parent(parent,x):
    if parent[x]!=x:
        parent[x]=find_parent(parent,parent[x])
    return parent[x]

def union_parent(parent,a,b):
    a=find_parent(parent,a)
    b=find_parent(parent,b)

    if a<b:
        parent[b]=a
    else:
        parent[a]=b

# v:노드개수, e:간선개수
v,e=map(int,input().split())

# 루트 테이블
parent=[0]*(v+1)

# 자기 자신을 루트노드로 초기화
for i in range(1,v+1):
    parent[i]=i

# 사이클 발생 여부
cycle=False

# 사이클 확인 과정
for _ in range(e):
    a,b=map(int,input().split())

    # 사이클이 발생한 경우 종료
    if find_parent(parent,a)==find_parent(parent,b):
        cycle=True
        break
    # 사이클이 발생하지 않았다면 합집합(union) 수행
    else:
        union_parent(parent,a,b)


if cycle:
    print('사이클 발생')
else:
    print('사이클 없음')
    
'''


# 크루스칼 알고리즘

'''
# 루트노드 찾기 함수
def find_parent(parent,x):
    if parent[x]!=x:
        parent[x]=find_parent(parent,parent[x])
    return parent[x]

# 합집합 수행 함수
def union_parent(parent,a,b):
    a=find_parent(parent,a)
    b=find_parent(parent,b)

    if a<b:
        parent[b]=a
    else:
        parent[a]=b

# v: 노드개수, e:간선개수
v,e=map(int,input().split())
parent=[0]*(v+1)

# edges: 간선 정보 입력 리스트, result: 신장트리 형성 최소 비용
edges=[]
result=0

# 부모 노드 자신으로 초기화
for i in range(1,v+1):
    parent[i]=i

# 간선에 대한 정보 입력
for i in range(e):
    a,b,cost=map(int,input().split())
    edges.append((cost,a,b))


# 비용에 대하여 오름차순 정렬
edges.sort()


# 크루스칼 알고리즘 수행
for edge in edges:
    cost,a,b=edge
    # 사이클이 발생하지 않는 경우만 집합에 포함
    if find_parent(parent,a)!=find_parent(parent,b):
        union_parent(parent,a,b)
        result+=cost

print(result)

'''


# 위상 정렬 알고리즘
# 사이클 없다고 가정하고 풀이

'''
from collections import deque

v,e=map(int,input().split())

# 모든 노드에 대해 진입차수는 0으로 초기화
indegree=[0]*(v+1)

# 간선 정보 입력 리스트 초기화
graph=[[]for _ in range(v+1)]

# 간선 정보 입력
for _ in range(e):
    a,b=map(int,input().split())
    graph[a].append(b) # 노드 a -> 노드 b 이동
    indegree[b]+=1 # 진입 차수 1증가

result=[] # 결과 저장 리스트

# 위상 정렬 함수
def topology_sort(graph):
    q=deque() # 큐 변수 선언
    

    # 진입차수 0인 노드 삽입
    for i in range(1,v+1):
        if indegree[i]==0:
            q.append(i)


    # 큐가 빌 때까지 반복
    while q:
        # 큐에서 원소 추출
        now=q.popleft()
        result.append(now) # 결과 리스트에 삽입

        # 해당 원소와 연결된 노드 진입차수 1빼기, 새롭게 진입차수가 0이 되는 노드 큐 삽입
        for i in graph[now]:
            indegree[i]-=1 
            if indegree[i]==0:
                q.append(i)

topology_sort(graph)

for i in result:
    print(i, end=' ')

'''


# 1717 집합의 표현

# 서로소 집합 알고리즘

'''
import sys
input=sys.stdin.readline
sys.setrecursionlimit(10000)

n,m=map(int,input().split())

# 부모 노드 집합
parent=[0]*(n+1)

# 자기 자신 루트 노드로 초기화
for i in range(1,n+1):
    parent[i]=i

# 루트 노드 탐색 함수
def find_parent(parent,x):
    if parent[x]!=x:
        parent[x]=find_parent(parent,parent[x])

    return parent[x]


# 합집합 합치기 함수
def union_parent(parent,a,b):
    a=find_parent(parent,a)
    b=find_parent(parent,b)

    if a<b:
        parent[b]=a
    else:
        parent[a]=b


result=[]

for _ in range(m):
    r,a,b=map(int,input().split())

    if r==0:
        union_parent(parent,a,b)
    elif r==1:
        if find_parent(parent,a)==find_parent(parent,b):
            result.append('YES')
        else:
            result.append('NO')
    


for i in result:
    print(i)

'''


# 1647 도시 분할 계획 (재밌음)

'''
풀이법

1. 크루스칼 알고리즘을 통한 신장 트리 형성

2. 두 개의 마을로 나뉘므로 가장 큰 cost간선 제거 -> last
   -> "일단 분리된 두 마을 사이에 있는 길들은 필요가 없으므로 없앨 수 있다"

3. 신장 트리에서 사이클이 존재하지 않으므로 2번 과정으로 나뉘어진 마을에서도 사이클이 존재 x
   -> "임의의 두 집 사이에 경로가 항상 존재하면 더 길을 없앨 수 있다" 조건 성립 x

'''

'''
# 크루스칼 알고리즘

# 루트 노드 탐색 함수
def find_parent(parent,x):
    if parent[x]!=x:
        parent[x]=find_parent(parent,parent[x])
    return parent[x]

# 합집합 함수
def union_parent(parent,a,b):
    a=find_parent(parent,a)
    b=find_parent(parent,b)

    if a<b:
        parent[b]=a
    else:
        parent[a]=b


n,m=map(int,input().split())

# 루트 노드 리스트
parent=[0]*(n+1)

# 자기 자신을 부모 노드로 초기화
for i in range(1,n+1):
    parent[i]=i
    

edges=[] # 간선 정보 저장 리스트
result=0 # 최소 경로 비용

for _ in range(m):
    a,b,cost=map(int,input().split())
    edges.append((cost,a,b))

# 간선 오름차순 정렬
edges.sort()
last=0 # 가장 큰 cost값을 저장하기 위한 변수

# 크루스칼 알고리즘
for edge in edges:
    cost,a,b=edge

    # 사이클을 포함하진 않는 경우 집합에 포함
    if find_parent(parent,a)!=find_parent(parent,b):
        union_parent(parent,a,b)
        result+=cost
        last=cost # 크루스칼 알고리즘 마지막 연결 노드 cost값 -> edges가 오름차순 정렬되어 있으므로
        

print(result-last)

'''

# 2252 줄 세우기

'''
풀이법

- 위상정렬

- 모든 학생들을 다 비교해 본 것이 아니고, 일부 학생들의 키만을 비교 -> 사이클 형성 X (Point)

- https://www.acmicpc.net/problem/2252

'''

'''
from collections import deque
import sys
input=sys.stdin.readline

n,m=map(int,input().split())

# 진입 차수 저장 리스트
indgr=[0]*(n+1)

# 간선 정보 저장 리스트
graph=[[] for _ in range(n+1)]

# 간선 정보 입력 및 진입 차수 설정
for _ in range(m):
    a,b=map(int,input().split())
    graph[a].append(b) # a->b 이동
    indgr[b]+=1

result=[]

def topology_sort(graph):

    q=deque()
,
    for i in range(1,n+1):
        if indgr[i]==0:
            q.append(i) #  인덱스 == 노드   >>  진입차수가 0인 노드의 인덱스 == 시작노드 번호

    while q:
        now=q.popleft() 
        result.append(now)

        for i in graph[now]:
            indgr[i]-=1
            if indgr[i]==0:
                q.append(i)
        


topology_sort(graph)
            
for i in result:
    print(i,end=' ')
    
'''



    

    










