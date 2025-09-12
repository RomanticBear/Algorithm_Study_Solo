# Greedy

# 1. 거스름돈

'''
N=int(input())

coin_lst=[500,100,50,10]
ans=0

for coin in coin_lst:
    if N//coin>=1:
        ans+=(N//coin)
        N%=coin
        print(coin, ans, N)
    

print(ans)

'''


# 2. 큰수의 법칙

'''
n,m,k=map(int,input().split())
arr=list(map(int,input().split()))

mok=m//k
na=m%k

arr.sort(reverse=True)
ans=(arr[0]*mok*k)+(arr[1]*na)

print(ans)

'''


# 곱하기 혹은 더하기

'''
arr=list(map(int,input()))

ans=0
for i in range(len(arr)):
    if arr[i]<=1 or ans<=1:
        ans+=arr[i]
    else:
        ans*=arr[i]
    
    print(arr[i],ans)

print(ans)

'''

# 모험가 길드

'''
길드 정렬

차례로 접근 -> 공포도 확인 -> 해당 공포도의 인원만큼 패스

'''

'''
N=int(input())
arr=list(map(int,input().split()))

arr.sort()
flag=True # 그룹 조건(True: 새 그룹 / False: 그룹 만드는 과정)
ans=0 # 총 그룹 수 
num=0 # 각 그룹의 인원수 


for i in range(len(arr)):
    if flag==True: # 그룹 만들 수 있으면
        flag=False  # 그룹 만든 과정으로 상태 변환
        goal=arr[i] # 그룹 형성을 위한 공포도 조건 받고,
        num+=1 # 인원 한명 추가
        
        
    else:
        num+=1 # 그룹 만드는 과정이라면 한명 추가
        goal=arr[i]
    
    if  num>=goal: # 그룹이 형성되었다면
        flag=True # 새 그룹 가능으로 조건 변환
        ans+=1 # 현재 그룹 개수 추가
        num=0
    

print(ans)

'''



# AVATA

# 상하좌우

'''
N=int(input()) # 지도 크기

lst=list(input().split())  # 이동 방향

dir={'R':(0,1),'L':(0,-1),'U':(-1,0),'D':(1,0)} 

x,y=0,0  # 시작점 

for d in lst:
    dx,dy=dir[d]

    nx=x+dx
    ny=y+dy

    if (0<=nx<N) and (0<=ny<N):
        x,y=nx,ny

ans_x,ans_y=x+1,y+1
# print(ans_x,ans_y)

'''

# 시각

'''
3이 총 몇번 들어갔는지 확인하기

- 3이 하나라도 포함되어 있으면 카운트, 하나라도 없으면 노카운트

00시 00분 00초 ~ N시 59분 59초

'''

'''
H=int(input())
cnt=0

for i in range(H+1):
    for j in range(60):
        for k in range(60):
            if ('3' in str(i)) or ('3' in str(j)) or ('3' in str(k)):
                cnt+=1


print(cnt)

'''


# 왕실의 나이트

'''
idx=input()
lst=list(idx)

row_idx={'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8}

x=row_idx[lst[0]]
y=int(lst[1])


dxy=[(-2,1),(-2,-1),(2,1),(2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]

ans=0
for dir in dxy:
    dx,dy=dir[0],dir[1]
    nx,ny=x+dx,y+dy

    if (1<=nx<9) and (1<=ny<9):
        ans+=1


print(ans)

    
'''


# 문자열 재정렬

'''
lst=list(input())
lst.sort()

print(lst)

num_lst=[]
ch_lst=[]

for i in range(len(lst)):
    if ord('1')<=ord(lst[i])<=ord('9'):
        num_lst.append(lst[i])
    else:
        ch_lst.append(lst[i])


num_ans=sum(list(map(int,num_lst)))
ch_ans=''.join(ch_lst)

ans=ch_ans+str(num_ans)

print(ans)

'''