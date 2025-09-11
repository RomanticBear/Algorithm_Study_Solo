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