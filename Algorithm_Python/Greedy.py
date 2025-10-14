'''
# 문제1) 1이 될 때까지 (내 코드)

num, k = map(int,input().split())

count=0

while True:
    if num%k==0: ## num이 k의 배수
        if num//k==1: ## num = k
            count+=1
            break
        else:
            num//=k
            count+=1
    else:
        num-=1
        count+=1

print(count)



'''
'''
문제 1) 풀이법 (로그 복잡도)


1. target = (num // k) * k   ## num보다 작은 값 중 k의 배수와 가장 가까운 값 -> target

2. count += num - target ## target 값 될때까지 -1 수행해야 하는 연산 -> count 추가

3. num = target 변경 (num: k의 배수)

4. num = num // k, count += 1

-> while문 반복, 종료 조건: num < k
-> while문 탈출, count += num-1 (1이 될때까지 -1 해야하는 연산 횟수 count 추가)

'''


'''
# 문제2) 곱하기 혹은 더하기  

num=input()
lst=list(map(int,str(num)))

s
result = 0


for i in lst:

    if i==0:
        continue
    elif i!=0 and result=<1:
        result+=i
            
    else:
        result*=i

print (result)

'''


# 문제3) 모험가 길드 (풀이법)

'''
1.입력 및 리스트 정렬

2. 그룹내 인원수 -> count, 그룹의 개수 -> group 설정 및 0으로 초기화

3. for문, count값 1증가 후 해당 원소 값과 비교,
   case 1) count >= 원소 값: group값 1증가, coount 초기화
   case 20 count < 원소 값: 다음 for문 -> 1증가 된 count 값과 다음 원소 비교 -> 반복 ,,,
   
4. group 출력

'''
'''
number = int(input())
num_list=list(map(int,input().split()))

num_list.sort()

count=0
group=0

for i in range(number):
    count+=1
    if count>=i:
        group+=1
        count=0
    else:
        continue

print(group)

'''

# 2023-09-21
# 15903 카드 합체 놀이

'''
N,m=map(int,input().split())
Card=list(map(int,input().split()))
Card_sum=0
sum=0


for i in range(m):
    Card.sort()
    Card_sum=Card[0]+Card[1]
    Card[0]=Card_sum
    Card[1]=Card_sum


for i in range(len(Card)):
    sum+=Card[i]

print(sum)

'''

# 13904 과제

''' 내 코드 -> 실패

N=int(input()) # 과제 개수
Use_day=0


lst=[]
sum=0

for i in range(N):
    lst.append(list(map(int,input().split())))


lst.sort(key=lambda x:-x[0]) # 남은 날짜 내림차순 정렬

for i in range(N-1):
    if lst[i][0]!=lst[i+1][0]:  # 가장 많이 남은 날의 개수가 한개 -> 무조건 가능
        sum+=lst[i][1]
        del lst[i] # 원소 제거

        Use_day+=1
        N-=1
        for j in range(N):
            lst[j][0]-=1 # 남은 과제 날짜 1감소
        
        
    else: # 두개 이상이라면
        # Left_day=lst[i+1][0] # 남은 과제 중 가장 긴 값 -> 남은 날짜 초기화
        break
    
lst.sort(key=lambda x:-x[1])  # 과제 점수 내림차순 정렬


for i in range(N):  # 남은 날짜만큼 차례로 sum
    if Use_day<=lst[i][0]:
        sum+=lst[i][1]
        Use_day+=1


print(sum)

'''


'''
풀이법

1. 마감일이 많이 남은 순 -> 점수가 높은 순 정렬

2. 가장 기간이 많은 날 부터 첫날까지 하루마다 가장 많은 점수를 취할 수 있게끔 작성

'''


# 1026 보물

''''
N=int(input())

A=list(map(int,input().split()))
B=list(map(int,input().split()))

A.sort()
B.sort(reverse=True)

result=0


for i in range(N):
    result+=A[i]*B[i]

print(result)

'''


# 1213 팰린드롬

'''
text=input() # 입력 문자열
N=len(text)

A_txt=[]  # 아스키코드 변환 문자열


# 결과 저장 문자열
R_txt=[0 for i in range(N)]

for i in text:
    A_txt.append(ord(i))

A_txt.sort()


for i in range(N):
    if i%2==0:
        j=i//2
        R_txt[j]=chr(A_txt[i])
        
    else:
        R_txt[N-1]=chr(A_txt[i])
        N-=1
        
# print(R_txt)

M=len(A_txt)
# print(M//2)

for i in range(M//2):

    if R_txt[i]==R_txt[M-1-i]:
        pass
        if i+1==M//2:
            result="".join(R_txt)
            print(result)
        else:
            pass

    else:
        print("I'm Sorry Hansoo")
        break

'''

'''
잘못된 내 코드 설명

1. 문자열 받기

2. 아스키 코드로 변환 -> 오름차순 정렬 : A_txt

3. 빈 리스트 앞뒤, 앞뒤 차례대로 값 넣어주기  : R_txt


반례) AAABB 입력

정답) ABABA
내 코드) AABBA -> 나머지 문자가 짝수, 아스키 값이 작은 숫자가 홀수일때 문제 발생
               -> AAABBB의 경우, ABBBA 출력



풀이법

- collecions모듈의 Counter함수 사용 -> 딕셔너리처럼 접근 가능

- 반복문을 통해 홀수 문자 개수 count

  : 홀수라면 홀수 개수 변수(add)에 추가, 홀수 문자 변수(odd_alpha)에 저장

  : 짝수라면 절반만큼 변수에 저장(ans)


- odd > 1 : print('sorry')

  odd = 0 : print(ans + ans[::-1])

  odd = 1 : print(ans + odd_alpha + ans[::-1])

'''

# 딕셔너리 사용법
# colections 모듈 알아보기 

'''
from collections import Counter

txt=input()
txt=sorted(txt)

count=Counter(txt)

odd_cnt=0 # 홀수인 문자 개수

half_txt=""


odd_txt=""

for i in count:
    
    if count[i]%2!=0:
        odd_cnt+=1
        odd_txt=i
        
        if odd_cnt>1:
            break

        for _ in range(count[i]//2):
            half_txt+=i
          
        

    else:
        for _ in range(count[i]//2):
            half_txt+=i


if odd_cnt>1:
    print("I'm Sorry Hansoo")
elif odd_cnt==1:
    print(half_txt + odd_txt + half_txt[::-1])
else:
    print(half_txt + half_txt[::-1])
    
'''

# 14720 우유 축제

'''
#딸기(0) -> 초코(1) -> 바나나(2)


N=int(input())

lst=list(map(int,input().split()))
cnt=0


person=[0,1,2]
N=0

for i in range(len(lst)):
    if lst[i]==person[N]:
        cnt+=1
        if N==2:
            N=0
        else:
            N+=1
    else:
        pass

print(cnt)
        
'''

# 1758 알바생 강호
'''
import sys

N=int(input())
lst=[]
place=1  # 등수 
sum=0 # 팁 받는 총 금액
for i in range(N):
    tip=int(sys.stdin.readline())
    lst.append(tip)

lst.sort(reverse=True)

for i in range(N):
    if lst[i]-(place-1)>=0:
        sum+=lst[i]-(place-1)
    place+=1

print(sum)

'''
# 1339 단어 수학

'''
정답 풀이
-> 각 문자마다 위치해있는 자리수를 통해 10의 거듭제곱 꼴로 표현 (딕셔너리 사용)
-> 큰 값부터 차례로 숫자 부여 
'''
'''
N=int(input())
s=[]
dic={}
for _ in range(N):
    string=input()
    s.append(string)

for x in s:
    num=1
    for i in range(len(x)-1,-1,-1):
        if x[i] not in dic:
            dic[x[i]]=num
        else:
            dic[x[i]]+=num
        num*=10

so_dic=sorted(dic.items(),key=lambda x:x[1],reverse=True)

ans=0
gop=9
for x in so_dic:
    number=x[1]
    ans+=number*gop
    gop-=1

print(ans)

'''


# 12904 A와 B

# 재귀 풀이 -> 시간초과
'''
lst=list(map(str,input()))

check=list(map(str,input()))
ans=0

def dfs(string):
    global ans

    
    if string==check:
        ans=1
        return
    
    if len(string)>len(check):
        return

    dfs(list(reversed(string))+['B'])
    dfs(string+['A'])

dfs(lst)
print(ans)

'''



# 1931 회의실 배정
'''
t=[0]*24

n=int(input())
arr=[]
for _ in range(n):
    a,b=map(int,input().split())
    arr.append((a,b))

arr.sort(key=lambda x:(x[1],x[0]))

cnt=1
t=arr[0][1]

for i in range(1,len(arr)):
    s,e=arr[i][0],arr[i][1]

    if s>=t:
        t=e
        cnt+=1

print(cnt)

'''

# 1541 잃어버린 괄호
# (-)뒤에 모든 (+)연산 수행 후 (-)연산 -> -연산 이후 등장하는 모든 연산자는 -연산
# 실패
'''
from collections import deque

sik=input()
sub=[]
ans=-1e9
flg=True
for i in range(len(sik)):
    print(i)
    if sik[i].isdigit():
        sub.append(sik[i])
    else:
        sub.reverse()
        num=int(''.join(sub))
        if ans==-1e9:
            ans=num
            continue
        if flg and sik[i]=='+':
            ans+=num
        else:
            ans-=num
            flg=False
        sub=[]
        print(ans)

print(ans)
'''

# 정답: (-)기준 split 함수 사용
'''
lst=input().split('-')
num=[]
for x in lst:
    sm=0
    tmp=x.split('+')
    for j in tmp:
        sm+=int(j)
    num.append(sm)

ans=num[0]
for i in range(1,len(num)):
    ans-=num[i]

print(ans)

'''

# 보석 도둑 [답지 이해 안가서 보류]
'''
import sys
import heapq
input=sys.stdin.readline

N,K=map(int,input().split())
items=[]
bag=[]
INF=1e9
for _ in range(N):
    w,v=map(int,input().split())
    # 첫번째 요소를 기준으로 정렬하므로 위치 바꿔주기
    heapq.heappush(itmes,(-v,w))

for _ in range(K):
    m=int(input())
    bag.append(m)

bag.sort()

ans=0

for m in bag:
    while itmes:
        jewel=heapq.heappop(items)
        v=-jewel[0]
        w=jewel[1]

'''

# 2847 게임을 만든 동준이
'''
import sys

n=int(input())
scr=[]
ans=0

for _ in range(n):
    scr.append(int(input()))

for i in range(len(scr)-2,-1,-1):
    if scr[i]>=scr[i+1]:
        ans+=scr[i]-scr[i+1]+1
        scr[i]=scr[i+1]-1

print(ans)

'''

# 11501 주식
# swea 백만 장자 동일 문제
'''
T=int(input())
for _ in range(T):
    n=int(input())
    lst=list(map(int,input().split()))

    mon=lst[-1]
    ans=0
    for i in range(len(lst)-2,-1,-1):
        if lst[i]<mon:
            ans+=(mon-lst[i])
        else:
            mon=lst[i]

    print(ans)
'''

# 13975 파일 합치기 3

'''
1) 매순간 가장 작은 값 두개 더함

10 2 2 10

2+2=4

10+4=14

14+10=24

-> 4+14+24=42


2) 작은 합으로 전체를 묶어보기

10+2=12

10+2=12

12+12=24

-> 12+12+24=48

'''
'''
import sys
import heapq
input=sys.stdin.readline

T=int(input())
for _ in range(T):
    n=int(input())
    lst=list(map(int,input().split()))
    heapq.heapify(lst)
    lst.sort()

    ans=0

    while len(lst)>1:
        a=heapq.heappop(lst)
        b=heapq.heappop(lst)
        sm=a+b
        ans+=sm
        heapq.heappush(lst,sm)

    print(ans)
'''

# 13904 과제

'''
n=int(input())
arr=[]
v=[False]*1001
for _ in range(n):
    t,s=map(int,input().split())
    arr.append((t,s))

arr.sort(key=lambda x:-x[1])

for x in arr:
    t,s=x[0],x[1]
    for i in range(t,0,-1):
        if v[i]==False:
            v[i]=s
            break

print(sum(v))

'''

# 19941 햄버거
'''
N,K=map(int,input().split())
s=list(map(str,input()))

cnt=0
for i in range(len(s)):
    if s[i]=='P':
        for j in range(i-K,i+K+1):
            if 0<=j<N:
                if s[j]=='H':
                    s[j]='O' # P로 초기화 시 뒤에 영향을 미침
                    cnt+=1
                    break

print(cnt)

'''

# 1461 도서관

'''
- 가장 긴 거리를 먼저 방문 해야함 
-> 짧은 거리일 수록 왕복 거리 작으므로 긴 거리부터 방문
-> 가장 긴거리 편도값 빼주면 끝

- 슬라이싱을 이용해서 M단위로 거리 더해주기 

'''
'''
N,M=map(int,input().split())
lst=list(map(int,input().split()))
A=[]
B=[]
dis=0

for x in lst:
    if x>0:
        A.append(x)
    else:
        B.append(-x)

if len(B)==0: # 음수가 없는 경우
    A,B=B,A
elif len(A)>0 and len(B)>0:
    if max(A)>max(B): # A가 더 값이 작은 리스트
        A,B=B,A


A.sort(reverse=True)
B.sort(reverse=True)

# 짧은 경로에 대한 책 가져다 놓기
for i in range(0,len(A),M):
    dis+=(A[i]*2)

# 긴 경로에 대한 책 가져다 놓기
for i in range(0,len(B),M):
    dis +=(B[i]*2)

dis-=B[0] # 긴 경로의 가장 먼 거리 빼주기

print(dis)

'''

# 2812 크게 만들기
# 내 코드, 재귀(메모리 초과)
'''
K개를 지웠을 때 -> 전체-K 를 뽑았을 때

import sys
sys.setrecursionlimit(10**6)

N,K=map(int,input().split())
C=N-K
arr=list(map(int,input()))
ans=0

def dfs(n,lst,cnt):
    global ans

    if cnt==C: # 다 뽑았을 때
        num=int(''.join(map(str,lst)))
        if num>ans:
            ans=num
        return

    if n==N: # 마지막 까지 살폈을 때
        return


    dfs(n+1,lst+[arr[n]],cnt+1) # 포함
    dfs(n+1,lst,cnt) # 미포함

dfs(0,[],0)
print(ans)

'''
# 2812 크게 만들기

'''
스택 이용

- 다음 원소가 스택안의 있는 마지막 원소보다 크면 스택 안의 원소 pop, K감소 (* 스택안의 원소는 내림차순으로 정렬될 수 밖에 없음)

- 스택안에 있는 모든 원소보다 큰 경우: 마지막 스택 원소부터 차례로 하나씩 빼고 cnt감소 시키기 -> cnt가 0보다 작다면 삭제할 수 없음

for에서 삭제할 경우 인덱스 에러 발생 -> while문 사용

'''
'''
N,K=map(int,input().split())
arr=list(map(int,input()))

stack=[]
for x in arr:
    if len(stack)==0:
        stack.append(x)
    else:
        while K>0:
            if len(stack)==0 or stack[-1]>=x:  # 스택이 비거나, 스택 마지막 원소가 x보다 클때까지 검사
                stack.append(x)
                break
            else:
                stack.pop(-1)
                K-=1

        if K<=0:  # 교체를 다 소진했는데 스택으로 옮기지 못한 숫자들이 있을 경우
            stack.append(x)

if K>0: # K가 남아있는 경우(더 빼야하는 경우)
    stack=stack[:len(stack)-K]

print(int(''.join(map(str,stack))))

'''

# 1343 폴리오미노
'''
lst=list(map(str,input()))
stack=[]
flg=True
cnt=0 # #개수
for i in range(len(lst)):
    if lst[i]=='X':
        cnt+=1
    else:
        if cnt>0:
            stack.append(cnt)
            cnt=0
        stack.append('.')

if cnt>0:
    stack.append(cnt)

ans=''
for x in stack:
    if x=='.':
        ans+=x
    else:
        if x%2!=0:
            flg=False
            break
        else:
            fir=x//4
            x%=4
            sec=x//2
            ans+=('AAAA'*fir+'BB'*sec)
if flg:
    print(ans)
else:
    print(-1)

'''


# 1911 흙길 보수하기
# while문 시간초과
'''
import sys
input=sys.stdin.readline

N,L=map(int,input().split())
lst=[]
for _ in range(N):
    lst.append(list(map(int,input().split())))

lst.sort(key=lambda x:(x[0],-x[1]))

cnt=0 # 널빤지 개수
l=lst[0][0] # 널빤지 처리 길이, 초기:웅덩이 시작점

for s,e in lst:
    while l<e: # 같거나 크면 종료
        if l<s:
            l=s+L
            cnt+=1
        else:
            l+=L
            cnt+=1

print(cnt)

'''

# for문 해결
'''
널빤지로 처리한 길이가 웅덩이 시작점 보다 작을 때
- 웅덩이 시작점 부터 처리
- 이외에는 전부 이어서 붙일 수 있음

for문으로 경계값 체크하면서 웅덩이당 cnt,l 증가 시킨다면 정답 도출 가능

'''
'''
import sys
input=sys.stdin.readline

N,L=map(int,input().split())
lst=[]
for _ in range(N):
    lst.append(list(map(int,input().split())))

lst.sort(key=lambda x:x[0])
l=lst[0][0]
cnt=0

for s,e in lst:
    if l<s:
        l=s
    gap=e-l

    if gap%L==0:
        cnt+=(gap//L)
        l=e
    else:
        cnt+=(gap//L+1)
        l+=(gap//L+1)*L

print(cnt)

'''

# 1374 강의실
# while문 시간초과
'''
N=int(input())
lst=[]
cnt=0
for _ in range(N):
    n,s,e=map(int,input().split())
    lst.append([s,e])

lst.sort()

for i in range(len(lst)):
    s,e=lst[i][0],lst[i][1]
    if s!=-1:
        idx=i
        end_t=e
        cnt+=1
        while True:
            idx+=1
            if idx==len(lst):
                break
            s,e=lst[idx][0],lst[idx][1]
            if end_t<=s:
                end_t=e
                lst[idx][0]=-1

print(cnt)
'''

# for문 시간초과
'''
import sys
input=sys.stdin.readline

N=int(input())
lst=[]
cnt=0
for _ in range(N):
    n,s,e=map(int,input().split())
    lst.append([s,e])

lst.sort()

for i in range(len(lst)-1):
    s1,e1=lst[i][0],lst[i][1]
    if s1!=-1:
        end_t = e1
        cnt += 1
        for j in range(i+1,len(lst)):
            s2,e2=lst[j][0],lst[j][1]
            if end_t<=s2:
                lst[j][0]=-1
                end_t=e2


if lst[-1][0]!=-1:
    cnt+=1

print(cnt)
'''

# 1246 온라인 판매

import sys
input=sys.stdin.readline

N,M=map(int,input().split())
lst=[]
for _ in range(M):
    lst.append(int(input()))

lst.sort()

ans_pri=0
ans_pro=0

for i in range(len(lst)):
    pri=lst[i] # 가격
    if len(lst)-i<=N:
        pro=pri*(len(lst)-i) # 이득
    else:
        pro=pri*N

    if ans_pro<pro:
        ans_pri=pri
        ans_pro=pro

print(ans_pri,ans_pro)



