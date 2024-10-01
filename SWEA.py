# 2072 홀수만 더하기
'''
T=int(input())

for i in range(T):
    sum=0
    num_list=list(map(int,input().split()))

    for j in range(len(num_list)):
        if num_list[j]%2!=0:
            sum+=num_list[j]
        else:
            pass

    print("%s%d %d" %("#",i+1,sum))
'''


# 1954 달팽이 숫자
'''
T=int(input())
for i in range(T):


n=int(input())
arr=[[0 for i in range(n)]for j in range(n)]

row=0
col=0
number=1

for i in range(n):
    for j in ragne(n):

'''

'''
def GCD(A,B):
    if A%B==0:
        num=B
        return num
    print(A,B)
    temp=B
    B=A%B
    A=temp
    return GCD(A,B)

result=GCD(192,162)
print(result)

'''

# 1206 [S/W 문제해결 기본] 1일차 - View

'''
풀이법

i번째 건물에서 조경권을 확보할 권리
-> i-2~i+2 건물의 가장 높은 층보다 높은 가구에 대해서 조경권 획득
-> result = stair[i] - max(i-2,i-1,i+1,i+2): 음수라면 패스 / 양수라면 sum+=result

'''

'''
T = 10
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T + 1):

    N=int(input()) # 건물 수
    H=list(map(int,input().split())) # 건물 높이 리스트

    sum=0
    for i in range(2,N-2):
        max_H=max(H[i-2],H[i-1],H[i+1],H[i+2])
        print(i,max_H,sum)

        if max_H<H[i]:
            sum+=H[i]-max_H
        else:
            pass
    print("#{} {}".format(test_case, sum))


'''

# 1244 [S/W 문제해결 응용] 2일차 - 최대 상금

'''
풀이법

num -> 횟수

i : 0~N-1까지
j : i+1,N-1까지

각 인덱스(i) 값 -> 가장 큰 숫자(j) 값 교체 ,  num-=1

-> if num -> 0 : for문 탈출

* 문제 : 정렬했는데 num이 존재했을 경우

-> N-1, N-2 인덱스 남은 num만큼 교

'''

# 내코드

'''
문제점

32888 2 -> 정답: 88823 / 내 코드: 88832

'''

'''
T = int(input())
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for TC in range(1, T + 1):

    num,cnt=map(int,input().split())
    lst=[]
    
    for i in map(int,str(num)):   
        lst.append(i)


    for i in range(len(lst)-1):
        if cnt==0:
            break
        
        max_val=0
        j_idx=0
        for j in range(i+1,len(lst)):
            if lst[j]>=max_val: # 숫자가 같은 경우 오른쪽에 있는 숫자랑 바꿔치기하는게 유리 -> 같은 숫자(=)일때 if문 실행
                max_val=lst[j]
                j_idx=j # max_val 저장된 원소 인덱스
                
        if lst[i]<max_val: # i인덱스 값 < max_val
            lst[i],lst[j_idx]=lst[j_idx],lst[i]
            cnt-=1
            # print(lst,cnt)
        else:
            pass

    # cnt가 내림차순 정렬 후에도 남아있는 경우 -> 끝에 두개 변경 -> 짝수면 동일, 홀수면 변경
    if cnt!=0:
        if cnt%2==0:
            pass
        else:
            lst[len(lst)-1],lst[len(lst)-2]=lst[len(lst)-2],lst[len(lst)-1]


    # 결과 출력
    ten=10**(len(lst)-1)
    result=0
    
    for i in range(len(lst)):
        result+=lst[i]*ten
        ten//=10
        # print(result,ten)

    print("#{} {}".format(TC,result))

'''


# 1204 최빈수 구하기

'''
파이썬 리스트 요소 개수 탐색

1) 특정 요소 개수 구하기 -> count()사용

ex)
    i = [1,1,3,4,5,3,3,7,6,8,9,3,2,5,9]

    print(i.count(3))

    >>> 4

2) https://www.daleseo.com/python-collections-counter/



'''

'''

T = int(input())

for test_case in range(1, T + 1):
    time=int(input())
    
    lst=list(map(int,input().split()))


    # array: 빈도 저장 리스트
    array=[0]*(max(lst)+1)  # 100점 -> array[101]저장

    for i in lst:
        array[i]+=1

    max_score=0

    print(array)

    for j in range(len(array)-1,-1,-1):
     
        if array[j]>max_score:
            max_score=array[j]
            max_idx=j # 최빈값 점수
            
            
    print("#{} {}".format(time,max_idx))


'''


# 1954 달팽이 숫자

'''

풀이법

array생성(N*N) : [0]초기화

i,j -> (0,0) -> 리스트 범위 안 + val이 0일 때

d[x]=[1,0,-1,0] 
d[y]=[0,-1,0,1]  -> 좌, 하, 우, 상 움직이면서 val값 갱신


종료 조건: 상하좌우 val != 0


'''


# 16800 구구단 걷기

'''
import math

T=int(input())

for TC in range(1,T+1):
        
    N=int(input())


    min_sum=10**12
    for i in range(1,int(math.sqrt(N))+1):
        sum=0
        
        if N%i==0:
            num1=i
            num2=N//i  
            sum=num1+num2
            
            if min_sum>sum:
                min_sum=sum
                
        if min_sum==0:
            num1=1
            num2=N

    if min_sum!=0:
        distance=min_sum-2
    else:
        distance=N-1
        
    print("#{} {}".format(TC,distance))

'''

# 16910 원 안의 점
        
'''
풀이법

-> 문제 공식 이용 x^2+y^2<=N : (x,y) 원안에 포함
-> 4사 분면에 대해서만 count(원점 제외 + x축 제외) : count *4 +1(원점) -> 전체 원 격자수

'''

'''
T=int(input())

for tc in range(1,T+1):
        
    N=int(input())

    cnt=0


    for i in range(1,N+1): # 행 : x축 제거(i=0)
        for j in range(N+1): # 열
            if (i**2)+(j**2)<=(N**2):
               # print(i,j)
                cnt+=1

    result=(cnt*4+1)
    print("#{} {}".format(tc,result))
    
            
'''


# 14555 공과 잡초

'''
풀이법

input -> 리스트 저장 -> for문 1번

if array[i] -> '(' : cnt +=1
            -> ')' : cnt +=1
            -> '()' : '(' 바로 다음 array[i+1] -> ')' : per_cnt+=1

result=cnt-per_cnt

'''

'''
T=int(input())

for tc in range(1,T+1):
    S=str(input().rstrip())
    garden=list(S)

    cnt=0
    per_cnt=0

    for i in range(len(S)):
        if garden[i]=='(' or garden[i]==')':
            cnt+=1

    for i in range(len(S)-1):
        if garden[i]=='(' and garden[i+1]==')':
            per_cnt+=1

    result=cnt-per_cnt

    print("#{} {}".format(tc,result))

'''

# 13728 숫자 조작

'''
문제 잘못이해함

-> 자리 변경 횟수 제한 없는줄 

풀이법

가장 큰 수 -> 내림차순 정렬 후 합쳐서 출력
가장 작은 수

- 오름차순 정렬 후

1) 첫번째 원소 !=0 -> 그래도 출력
2) 첫번째 원소 ==0 -> 0아 아닌 가장 작은 원소 POP -> 가장 앞에 추가 -> 그대로 출력


T=int(input())

for tc in range(1,T+1):
        
    N=int(input())

    lst=[]
    for i in map(int,str(N)):
        lst.append(i)

    # 최대값
    max_lst=sorted(lst,reverse=True)
    max_M=''.join(map(str,max_lst))


    # 최소값
    row_lst=sorted(lst)


    if row_lst[0]!=0:
        row_M=''.join(map(str,row_lst))

    else:

        i=0
        while(True):
            if row_lst[i]!=0:
                pop_num=row_lst.pop(i)
                break
            
            else:
                i+=1

        row_lst.insert(0,pop_num)
        row_M=''.join(map(str,row_lst))

    print("#{} {} {}".format(tc,row_M,max_M))

    
'''


# 13547 팔씨름

'''
풀이법

1) 졋을 때

-> x가 8개이상

2) 이겼을 때
-> o가 8개 이상

3) 이길 가능성이 있을 때
-> x<8

즉,
확실히 졌을 때 -> x>8
이기거나 이길 가능성 있을 때 -> x<8 or o>=8

'''

'''

T=int(input())

for tc in range(1,T+1):
    S=input().strip()
    lst=list(S)

    cnt_x=lst.count('x')
    cnt_o=lst.count('o')

    if cnt_x>=8:
        result='NO'

    elif cnt_o>=8 or cnt_x<8:
        result='YES'

    print("#{} {}".format(tc,result))

'''

# 11856 반반

'''
T=int(input())

for tc in range(1,T+1):

    S=input().strip()
    lst=list(S)

    dic={}
    for i in lst:
        if i in dic:
            dic[i]+=1
        else:
            dic[i]=1


    for value in dic.values():
        if value!=2:
            result='NO'
            break
        else:
            result='YES'

    print("#{} {}".format(tc,result))

'''


# 12741 두 전구


'''
# 시간초과

T=int(input())

for tc in range(1,T+1):

    a,b,c,d=map(int,input().split())


    table=[0]*(max(a,b,c,d)+1)

    for i in range(a,b+1):
        table[i]+=1


    for j in range(c,d+1):
        table[j]+=1

    result=table.count(2) 

    if result==0:
        print(0)
    else:
        print("#{} {}".format(tc,result-1)) # 시작하는 시각 카운트 제거

'''

'''
시간초과 2 (1보단 나음)

T=int(input())

for tc in range(1,T+1):

    a,b,c,d=map(int,input().split())

    lst1=[x for x in range(a,b+1)]
    lst2=[x for x in range(c,d+1)]

    # 교집합 개수 구하기

    result=max(0,len(set(lst1)&set(lst2))-1)
    print("#{} {}".format(tc,result))

'''
    



# 12004 구구단 1

'''
T=int(input())

for tc in range(1,T+1):
        
    N=int(input())

    result=False

    for i in range(1,10):
        a=i
        if N%i==0:
            b=N//i
            if a<=9 and b<=9:
                result=True


    if result==False:
        print("#{} {}".format(tc,'No'))

    else:
        print("#{} {}".format(tc,'Yes'))
'''


# 12211 구구단2

'''
T=int(input())

for tc in range(1,T+1):

    a,b=map(int,input().split())

    if a>9 or b>9:
        result=-1
    else:
        result=a*b

    print("#{} {}".format(tc,result))
        
'''

# 13038 교환학생

'''
풀이법

일주일 동안 열리는 수업(1) 횟수 카운트 -> cnt

지내야 하는 날: mok(N%cnt)*7 + 나머지 마지막날(마지막 인덱스+1 -> last_day)


나머지가 0이라면 (mok-1)*7 + 마지막주 남은 날짜 카운트

나머지가 0이 아니라면 mok*7 + 마지막주 남은 날짜 카운트



+ 첫날이 일요일이라면 괜찮지만 다른 요일의 경우 빼줘야함

'''

'''
# 75% 통과

T=int(input())

for tc in range(1,T+1):

    N=int(input())

    lst=list(map(int,input().split()))
    print(lst)

    start_day=0

    for i in range(len(lst)):
        if lst[i]==1:
            break
        else:
            start_day+=1

            


    # 일주일 수업 횟수
    cnt=lst.count(1)

    mok=N//cnt
    remain=N%cnt

    # print(mok,remain)

    last_day=0

    if remain==0:
        mok-=1

        for i in range(len(lst)):
            if lst[i]==1:
                cnt-=1
                last_day+=1
            else:
                last_day+=1

            if cnt==0:
                break


    else:

        for i in range(len(lst)):
            if lst[i]==1:
                remain-=1
                last_day+=1
            else:
                last_day=i+1
            if remain==0:
                break


    result=mok*7+last_day-start_day

    print("#{} {}".format(tc,result))

'''


# 13218 조별과제

'''

T=int(input())

for tc in range(1,T+1):

    N=int(input())

    result=(N//3)

    print("#{} {}".format(tc,result))

'''


# 12368 평범한 숫자

'''
T=int(input())

for tc in range(1,T+1):

    a,b=map(int,input().split())

    a_t=a+b

    if a_t<24:
        result=a_t
    else:
        result=a_t%24


    print("#{} {}".format(tc,result))

'''


# 12051 프리셀 통계 -> 문제 이해 X


'''
풀이법

D: 오늘 이긴 경기 수
G: 지금까지 이긴 경기 수

N: 적어도 오늘 경기 수

Pd: 오늘 이긴 확률
Pg: 지금 까지 이긴 확률

'''

# 11387 몬스터 사냥

'''
T=int(input())

for tc in range(1,T+1):

    D,L,N=map(int,input().split())



    total=0


    for num in range(N):
        total+=D*(1+num*(L/100))


    print("#{} {}".format(tc,int(total)))

'''

# 11445 무한 사전

'''
T=int(input())

for tc in range(1,T+1):

    p=input().strip()
    q=input().strip()

    if p+'a'==q:
        result='N'
    else:
        result='Y'

    print("#{} {}".format(tc,result))

'''

# 19012 외로운 문자

'''
T=int(input())

for tc in range(1,T+1):
        
    lst=list(map(str,input().strip()))

    dic={}
    for i in lst:
        if i in dic:
            dic[i]+=1
        else:
            dic[i]=1

    odd_lst=[]
    for key,val in dic.items():
        if val%2==0:
            pass
        else:
            odd_lst.append(key)
            

    odd_lst.sort()




    if len(odd_lst)==0:
        print("#{} {}".format(tc,'Good'))
    else:
        result=''.join(map(str,odd_lst))
        print("#{} {}".format(tc,result))

'''


# 10804 문자열의 거울상

'''
T=int(input())

for tc in range(1,T+1):
    lst=list(input().strip())

    N=len(lst)
    rs=[0]*N

    for i in range(N):
        if lst[i]=='b':
            rs[N-i-1]='d'
        elif lst[i]=='d':
            rs[N-i-1]='b'
        elif lst[i]=='p':
            rs[N-i-1]='q'
        else:
            rs[N-i-1]='p'

    result=''.join(map(str,rs))
    print("#{} {}".format(tc,result))
    

'''

# 14178 1차원 정원

'''
T=int(input())

for tc in range(1,T+1):

    N,D=map(int,input().split())

    mok=N//(2*D+1)

 
    if N%(2*D+1)!=0:
        result=mok+1
    else:
        result=mok


    print("#{} {}".format(tc,result))   

'''

# 11746 평범한 숫자

'''
T=int(input())

for tc in range(1,T+1):

    N=int(input())

    lst=list(map(int,input().split()))
    cnt=0

    for i in range(1,N-1):
        max_num=max(lst[i-1],lst[i],lst[i+1])
        min_num=min(lst[i-1],lst[i],lst[i+1])

        if lst[i]!=max_num and lst[i]!=min_num:
            cnt+=1
        else:
            pass

    print("#{} {}".format(tc,cnt))  


'''

# 1961 숫자 배열 회전 -> 포기 ㅅㅂ ㅈㄴ조잡함

'''

T=int(input())


for tc in range(1,T+1):

    
    N=int(input())  
    
    lst=[]
    for i in range(N):
        lst.append(list(map(int,input().split())))


    ro_lst=[[0,0,0],[0,0,0],[0,0,0]]

    # 90도
    lst1=[[0]*N for j in range(N)]

    # 180도
    lst2=[[0]*N for j in range(N)]

    # 270도
    lst3=[[0]*N for j in range(N)]


    # 90도
    for i in range(N):
        for j in range(N):
            lst1[i][j]=lst[N-1-j][i]        
            lst2[i][j]=lst[N-1-i][N-1-j]
            lst3[i][j] = lst[j][N-1-i]

    print("#{}".format(tc))
    for i in range(N):
        print(''.join(map(str,lst1[i])),end=' ')
        print(''.join(map(str,lst2[i])),end=' ')
        print(''.join(map(str,lst3[i])))
        


'''

# 2001 파리 퇴치

'''
T=int(input())

for tc in range(1,T+1):

    N,M=map(int,input().split())

    arr=[]
    for i in range(N):
        arr.append(list(map(int,input().split())))

    max_sum=0

    for i in range(N-M+1):
        for j in range(N-M+1):

            sum=0
            for row in range(M):
                for col in range(M):
                    sum+=arr[i+row][j+col]

            if max_sum<sum:
                max_sum=sum

            
    print("#{} {}".format(tc,max_sum))


'''


# 1959 두 개의 숫자열

'''
T=int(input())

for tc in range(1,T+1):
        

    N,M=map(int,input().split())

    A_arr=list(map(int,input().split()))  # N
    B_arr=list(map(int,input().split()))  # M


    max_sum=0

    if N<M:
        
        for i in range(M-N+1):
            sum=0
            for j in range(N):
                sum+=A_arr[j]*B_arr[i+j]

            if max_sum<sum:
                max_sum=sum

    elif N>M:
        
        for i in range(N-M+1):
            sum=0
            for j in range(M):
                sum+=A_arr[i+j]*B_arr[j]

            if max_sum<sum:
                max_sum=sum

    else:
        for i in range(N):
            max_sum+=A_arr[i]*B_arr[i]



    print("#{} {}".format(tc,max_sum))
        
                
'''


# 1979 어디에 단어가 들어갈 수 있을


'''
풀이법

가로로 [1] * 3 cnt
세로로 [1] * 3 cnt

-> 1이면 cnt+=1, 마지막 열이거나 0이면 cnt가 3인지 확인

https://unie2.tistory.com/1001


'''

'''
T=int(input())

for tc in range(1,T+1):

    N,K=map(int,input().split())

    arr=[]
    for i in range(N):
        arr.append(list(map(int,input().split())))

    # 퍼즐 공간 개수
    cnt=0

    # 가로 탐색

    for i in range(N):
        sum=0
        for j in range(N):
            if arr[i][j]==1:
                sum+=1
                #print(j,sum)

            if arr[i][j]==0 or j==N-1:
                if sum==K:
                    cnt+=1
                if arr[i][j]==0:
                    sum=0


    # 세로 탐색
    for j in range(N):
        sum=0
        for i in range(N):
            if arr[i][j]==1:
                sum+=1
                #print(j,sum)

            if arr[i][j]==0 or i==N-1:
                if sum==K:
                    cnt+=1
                if arr[i][j]==0:
                    sum=0

    print("#{} {}".format(tc,cnt))


'''



# 1227 미로2

# DFS

# 3을 찾을 때 reuslt->1 변환하고 함수 바깥에서 출력했는데 왜 0이라고 하는지 모르겠다 ... 왜 ,, !!

'''
graph=[]

for i in range(16):
    graph.append(list(map(int,input())))



dx=[1,-1,0,0]
dy=[0,0,-1,1]


def dfs(x,y):
    result=0

    if graph[x][y]==0 or graph[x][y]==2:
        graph[x][y]=5

        for i in range(4):
            nx=x+dx[i]
            ny=y+dy[i]

            if 0<=nx<16 and 0<=ny<16:

                if graph[nx][ny]==3:
                    result=1
                    print('ok')
                    return result
                    
                elif graph[nx][ny]==0:
                    dfs(nx,ny)

    return result
                   

res=dfs(1,1)
print(res)

'''


############ 블로그 보고 문제풀기 ##################

# 1204. 최빈수구하기
# 최빈수가 여러개 일때 가장 큰 점수 뽑아라 -> 뒤에서 접근

'''
T=int(input())

for tc in range(1,T+1):
    N=int(input())
        
    lst=list(map(int,input().split()))

    cnt_arr=[0]*(max(lst)+1)

    max_cnt=0
    max_idx=0

    for i in lst:
        cnt_arr[i]+=1

    for j in range(len(cnt_arr)-1,-1,-1):  # 인덱스 -> 점수 -> j
        if cnt_arr[j]>max_cnt:
            max_cnt=cnt_arr[j]
            max_idx=j

    print("#{} {}".format(N,max_idx))
        

'''

# 1984. 중간 평균값 구하

'''
T=int(input())

for tc in range(1,T+1):
    
    lst=list(map(int,input().split()))

    max_num=max(lst)
    min_num=min(lst)


    sum=0
    cnt=0
    for i in range(len(lst)):
        if lst[i]!=max_num and lst[i]!=min_num:
            sum+=lst[i]
            cnt+=1

    if cnt==0 or sum==0:
        print("#{} {}".format(tc,0))

    else:
        print("#{} {}".format(tc,round(sum/cnt)))

'''

# 1946 간단한 압축풀기

'''
T=int(input())

for tc in range(1,T+1):

    lst=[]
    
    N=int(input())
    
    string=''
    for i in range(N):
        C, k=input().split()
        k=int(k)
        string+=C*k


    print("#{}".format(tc))


    for i in range(len(string)) :
        if (i+1)%10 == 0 :
            print(string[i])
        else :
            print(string[i], end="")

'''

# 1983 조교의 성적 매기기

'''
T=int(input())

for tc in range(1,T+1):
        

    N,K=map(int,input().split())
    K=K-1

    lst=[]
    for i in range(N):
        M1,M2,M3=map(int,input().split())
        score=M1*0.35+M2*0.45+M3*0.2
        lst.append(score)

    # 총점 저장 리스트 

    cnt=0 # 자신보다 낮은 사람 수

    for i in range(N):
        if i==K:
            pass
        else:
            if lst[i]<lst[K]:
                cnt+=1


    #print(cnt)

    # 본인 성적 %
    per=cnt/N

    if cnt==0 or per>=0.9:
        rank='A+'
    elif per>=0.8:
        rank='A0'
    elif per>=0.7:
        rank='A-'
    elif per>=0.6:
        rank='B+'
    elif per>=0.5:
        rank='B0'
    elif per>=0.4:
        rank='B-'
    elif per>=0.3:
        rank='C+'
    elif per>=0.2:
        rank='C0'
    elif per>=0.1:
        rank='C-'
    else:
        rank='D0'

    print("#{} {}".format(tc,rank))
        

        
# 다른풀이

-> python index함수 이용: 원하는 값의 위치를 찾아줌, 중복된 값 중 최소 위치 반환

-> https://pydole.tistory.com/entry/Python-index-%ED%95%A8%EC%88%98-%EB%B0%B0%EC%97%B4%EC%97%90%EC%84%9C-%EC%9B%90%ED%95%98%EB%8A%94-%EA%B0%92%EC%9D%98-%EC%9C%84%EC%B9%98-%EC%B0%BE%EA%B8%B0
        
-> https://velog.io/@shon4bw/SWEA-1983-%EC%A1%B0%EA%B5%90%EC%9D%98-%EC%84%B1%EC%A0%81-%EB%A7%A4%EA%B8%B0%EA%B8%B0-%ED%8C%8C%EC%9D%B4%EC%8D%AC

'''

# 1288. 새로운 불면증 치료법

'''
T=int(input())

for tc in range(1,T+1):
        

    N=int(input())

    lst=[0]*10


    cnt=1
    num=N
    result=0
    while(True):
        
        arr=list(map(int,str(num)))


        for i in arr:
            if lst[i]==1:
                pass
            else:
                lst[i]=1

        if sum(lst)==10:
            result=num
            break
        
        else:
            cnt+=1
            num=N*cnt # N은 고정 cnt만 바뀜

    print("#{} {}".format(tc,result))
            
'''

# 1926. 간단한 369게임

'''
N=int(input())
t_lst=[3,6,9]

for i in range(1,N+1):
    num_lst=list(map(int,str(i)))

    for j in range(len(num_lst)):
        if num_lst[j] in t_lst:
            print('-',end='')
        else:
            print(num_lst[j],end='')
            
    
'''



# 1225. 암호생성기

'''
T=10

for i in range(T):
        
    N=int(input())

    lst=list(map(int,input().split())) # 8개 숫자 입력


    cnt=1
    while(True):

        if cnt==6:
            cnt=1
            
        num=lst.pop(0)-cnt

        if num<=0:
            num=0
            lst.append(num)
            break
        else:
            lst.append(num)
            cnt+=1


    print("#{}".format(N),end=' ')

    for x in lst:
        print(x, end=' ')

    print()

'''          

# 1220 Magntic -> 문제박스(WOW)추가

# 코드1) flag 사용

'''
for tc in range(1,11):
        
    N=int(input())

    lst=[]
    col_lst=[]
    for i in range(N):
        lst.append(list(map(int,input().split())))

    cnt=0

    for j in range(N):
        
        flag=0 # 열이 바뀌면 falg값 갱신
        
        for i in range(N):
            if lst[i][j]==1: # N극이라면 flag:1
                flag=1
            elif lst[i][j]==2: # S극일때
                if flag==1: # N극을 만나고 왔다면 cnt증가, flag값 0 초기화(다시 N극을 만나고 와야함)
                    cnt+=1
                    flag=0

    print("#{} {}".format(tc,cnt))

'''

# 코드2) stack사용

'''
for tc in range(1,11):
    

    N=int(input())

    lst=[]
    for i in range(N):
        lst.append(list(map(int,input().strip())))


    cnt=0
    for j in range(N):
        r=0 # 행
        stack=[] # 열이 바뀔 때마다 스택 초기화
        
        while r<N: # 첫행부터 끝행까지
            if not stack and lst[r][j]==1:
                stack.append(1)
            elif stack and lst[r][j]==2:
                cnt+=stack.pop()
            r+=1
            

    print("#{} {}".format(tc,cnt))

'''

# 1234 비밀번호

'''
for tc in range(1,11):

    N,M=input().split()

    lst=list(M)


    stack=[]

    for i in lst:
        if len(stack)==0:
            stack.append(i)
        else:
            if stack[-1]==i:
                stack.pop()
            else:
                stack.append(i)
                
    # result=''.join(map(str,stack))
    print(f'#{tc}',' ',*stack,sep='')

'''


# 2805 농작물 수확하기

'''
풀이법

행 mid -> start,end 초기화

열 mid까지 -> start-1, end+1

열 mid 후부터 -> start+1, end-1

'''

'''
# 왜 리스트 인덱스 밖이냐 ,, 시발 진짜 ,,

N=int(input())

lst=[]
for i in range(N):
    lst.append(list(map(int,input().strip())))


mid=N//2
s,e=mid,mid

sum=0
for i in range(N):
    for j in range(s,e+1):
        sum+=lst[i][j]

        if i<mid:
            s-=1
            e+=1
        else:
            s+=1
            e-=1


print(sum)
            
'''


# 9611 명진이와 동휘의 숫자 맞추기

'''
제외 숫자 명단 리스트
YES 리스트 비교

'''

'''
N=int(input())

Y_lst=[]
N_lst=[]
for i in range(N):
    N1,N2,N3,N4,Q=input().split()

    if Q=='YES':
        Y_lst.append(int(N1))
        Y_lst.append(int(N2))
        Y_lst.append(int(N3))
        Y_lst.append(int(N4))
    
    else:
        N_lst.append(int(N1))
        N_lst.append(int(N2))
        N_lst.append(int(N3))
        N_lst.append(int(N4))


Y_lst=list(set(Y_lst))
N_lst=list(set(N_lst))

for x in N_lst:
    if x in Y_lst:S
        Y_lst.remove(x)

print(Y_lst[0])
    
'''

# 6692 다솔이의 월급 상자

'''
T=int(input())

for tc in range(1,T+1):

    N=int(input())

    avg=0

    for i in range(N):
        P,X=map(float,input().split())
        avg+=P*X

    print("#{} {}".format(tc,avg))

'''



################ SAFFY 12 ################
################### D2 ###################
 
# 1945 간단한 소인수분해

'''
T = int(input())

for test_case in range(1, T + 1):
    num=int(input())
    lst=[2,3,5,7,11]
    ans=[]

    for x in lst:
        cnt=0
        while True:
            if num%x!=0:
                ans.append(cnt)
                break
            else:
                num//=x
                cnt+=1
    print(f'#{test_case}', *result)
    
'''



# 1986 지그재그 숫자

'''
T = int(input())

for test_case in range(1, T + 1):
    num=int(input())
    ans=0
    for i in range(1,num+1):
        if i%2!=0:
            ans+=i
        else:
            ans-=i
    print(f'#{test_case}',ans)
    
'''

# 1288 새로운 불면증 치료법

'''
T=int(input())

for tc in range(1,T+1):
    
    dic={i:0 for i in range(1,10)}
    t=int(input())
    cnt=1

    while True:
        num=t*cnt
        lst=list(str(num))
        for x in lst:
            if x in dic:
                pass
            else:
                dic[x]=1
                
        if sum(dic.values())==10:
            break
        else:
            cnt+=1

    print(f'#{tc}',num)
        
'''

# 1284 수도 요금 경쟁
'''
T=int(input())

for tc in range(1,T+1):
    P,Q,R,S,W=map(int,input().split())

    if W<=R:
        ans=min(Q,P*W)
    else:
        ans=min(Q+(W-R)*S,P*W)

    print(f'#{tc}', ans)
'''

# 1989 초심자의 회문 검사

'''
T = int(input())

for tc in range(1, T + 1):
    lst=list(input())
    start=0
    end=len(lst)-1

    while True:
        if start==end or start>end:
            ans=1
            break
        else:
            print(start,end)
            if lst[start]==lst[end]:
               start+=1
               end-=1
            else:
                ans=0
                break

    print(f'#{tc}',ans)

'''

# 1959 두 개의 문자열
'''
T=int(input())

for tc in range(1,T+1):
    n,m=map(int,input().split())
    A=list(map(int,input().split()))
    B=list(map(int,input().split()))

    ans=0

    if n>m:
        A,B=B,A
        
    if n==m:
        for i in range(n):
            ans+=(A[i]*B[i])
    else:
        lst=[]
        for i in range(len(B)-len(A)+1):
            lst.append(B[i:i+len(A)])

        for x in lst:
            sum=0
            for i in range(len(A)):
                sum+=x[i]*A[i]
            ans=max(ans,sum)

    print(f'#{tc}',ans)

'''

# 1996 숫자를 정렬하자

'''
T = int(input())

for test_case in range(1, T + 1):
     n=int(input())
     lst=list(map(int,input().split()))
     lst.sort()
     print(f'#{test_case}',*lst)

'''


# 1947 날짜 계산기

'''
시작달~마지막달 까지 날짜 더함
시작달 시작날짜 빼주기
마지막달 마지막 날짜 더해주기



T=int(input())

for tc in range(1,T+1):
    day=0
    mon=[0,31,28,31,30,31,30,31,31,30,31,30,31]
    sm,sd,em,ed=map(int,input().split())

    for i in range(sm,em):
        day+=mon[i]

    day-=sd
    day+=ed

    print(f'#{tc}', day+1)
    

'''

# 1285 아름이의 돌 던지기

'''
n=int(input())
lst=list(map(int,input().split()))

dic={}
for x in lst:
    if abs(x) in dic:
        dic[abs(x)]+=1
    else:
        dic[abs(x)]=1

sorted_dic=sorted(dic.items())
return sorted_dic[0][0],sorted_dic[0][1]

'''


# 1970 쉬운 거스름돈

'''
T = int(input())
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for tc in range(1, T + 1):
    mon=[50000,10000,5000,1000,500,100,50,10]
    ans=[0]*len(mon)
    price=int(input())

    for i in range(len(mon)):
        if price>=mon[i]:
            ans[i]=(price//mon[i])
            price%=mon[i]

        if price==0:
            break

    print(f'#{tc}')
    print(*ans)
            
'''


# 1976 시각 덧셈

'''
T = int(input())
for tc in range(1, T + 1):
    h1,m1,h2,m2=map(int,input().split())
    sum_h=h1+h2
    sum_m=m1+m2

    if sum_h>12:
        sum_h%=12
        
    if sum_m>=60:
        sum_h+=(sum_m//60)
        sum_m%=60

    if sum_h>12:
        sum_h%=12

    print(f'#{tc} {sum_h} {sum_m}')

'''



# 파리 퇴치
'''
T=int(input())

for tc in range(1,T+1):
    n,m=map(int,input().split())
    arr=[]

    for _ in range(n):
        arr.append(list(map(int,input().split())))

    ans=0
    for i in range(n-m+1):
        for j in range(n-m+1):
            sum=0
            for k1 in range(m):
                for k2 in range(m):
                    sum+=arr[i+k1][j+k2]
            ans=max(ans,sum)

    print(f'#{tc}',ans)

'''


# 1974 스도쿠 검증
'''
check=[1,2,3,4,5,6,7,8,9]

def inspection(arr):
    
    # 가로 검사
    for x in arr:
        lst=[]
        for i in x:
            lst.append(i)
        lst.sort()
        if lst!=check:
            return 0

    # 세로 검사
    for j in range(len(x)):
        lst=[]
        for i in range(len(x)):
            lst.append(arr[i][j])
        lst.sort()
        if lst!=check:
            return 0

        
    # 박스 검사
    r_idx=0
    for _ in range(3):
        c_idx=0
        for _ in range(3):
            lst=[]
            for i in range(r_idx,r_idx+3):
                for j in range(c_idx,c_idx+3):
                    lst.append(arr[i][j])
            lst.sort()
            if lst!=check:
                return 0

            c_idx+=3
        r_idx+=3
            
    return 1


T = int(input())
for tc in range(1, T + 1):
    arr=[]
    for _ in range(9):
        arr.append(list(map(int,input().split())))   

    ans=inspection(arr)
    print(f'#{tc}',ans)

'''


# 2005 파스칼의 삼각형
'''
T = int(input())

for tc in range(1, T + 1):
    n=int(input())
    arr=[[0]*n for _ in range(n)]
    arr[0][0]=1

    for i in range(1,n):
        for j in range(i+1):
            if j==0 or j==i:
                arr[i][j]=1
            else:
                arr[i][j]=arr[i-1][j-1]+arr[i-1][j]

    print(f'#{tc}')
    for x in arr:
        for i in x:
            if i!=0:
                print(i,end=' ')
        print()

'''

# 1926 간단한 369게임
'''
n=int(input())
lst=['3','6','9']
for num in range(1,n+1):
    num=str(num)
    cnt=0
    for i in range(len(num)):
        if num[i] in lst:
            cnt+=1

    if cnt==0:
        print(num,end='')
    else:
        print('-'*cnt,end='')
    
    if num!=n:
        print(end=' ')
    
'''


# 간단한 압축 풀기
'''
T=int(input())

for tc in range(1,T+1):
    
    n=int(input())
    string=''
    cnt=0
    
    for _ in range(n):
        w,k=input().split()
        k=int(k)
        string+=w*k

    
    print(f'#{tc}') 
    for i in range(len(string)):
        cnt+=1
        if cnt==10:
            print(string[i])
            
    
        else:
            print(string[i],end='')

    print()
    
'''


# 1204 최빈수 구하기
'''
T=int(input())
for tc in range(1,T+1):
    t=int(input())
    dic={}
    lst=list(map(int,input().split()))

    for x in lst:
        if x in dic:
            dic[x]+=1
        else:
            dic[x]=1

    max_dic=[k for k,v in dic.items() if max(dic.values()) == v]  # 방법2
    max_dic.sort(reverse=True)
    print(f'{tc}',max_dic[0])

'''

# 1984 중간 평균값 구하기

''' 
T=int(input())

for tc in range(1,T+1):
    lst=list(map(int,input().split()))
    lst.sort()
    print(f'#{tc}',round(sum(lst[1:len(lst)-1])//(len(lst)-2)))
'''


# 1859 백만 장자 프로젝트


'''
내 생각1) max val -> 무조건 판매?

반례) 1 1 1 1 2  -> 안사는게 이득

내 생각2) 이전 값보다 커지면 무조건 판매?

반례) 1 1 3 1 1 100 -> 마지막에 다 파는게 이득



# 풀이방법

거
꾸
로 보기 ,,, 

끝 >> 처음 순서 인덱스 접근

- 중간에서 자신보다 큰 숫자가 나온다면 멈춤 -> 직전까지 이득(현재까지 가장 큰 숫자 - 현재 인덱스 값) 더해주기 -> 가장 큰 숫자값 변경

고민1) 음수라면 >> 음수가 나올 수 없음, 자신보다 작은 숫자에 대해서는 무조건 이득 봄
ex: 1 1 5 1 ... 2 4 -> 5와 4사이의 숫자(4보다 작은 숫자만 있다고 가정)에 대해서는 모두 4에 대한 이득을 취함


고민 2)
나눠서 이득을 취하는게 최대인 경우와 가장 큰 최댓값까지 모으고 한번에 파는게 최대인 경우를 어떻게 분류할까 ..
>> 뒤에 인덱스부터 접근하면서 최대 판매가를 갱신 시키면서 이득을 누적하면 그럴일 없음

'''

'''
T=int(input())  # 테스트 케이스 개수

for tc in range(1,T+1):
    
    n=int(input())  
    lst=list(map(int,input().split()))  # 리스트 입력


    max_val=0    # 판매가
    point=0  # 총 이득(결과값)

    for i in range(len(lst)-1,-1,-1): # 뒤에서 부터 앞으로 순차 접근
        if lst[i]>=max_val: # max_val보다 더 큰 값이 나올 경우 갱신 
            max_val=lst[i]

        
        else:  # 아니라면 이득 누적
            point+=max_val-lst[i]   # 최종 판매가(max_val) - 구입가

    print(f'#{tc}',point)
        
'''       
    
# 패턴 마디의 길이

'''
T=int(input()) 

for tc in range(1,T+1):
    s=input()

    for i in range(1,11):
        mok=(10//i)+1
        if s[:i]*mok==s[:len(s[:i]*mok)]:
            res=i
            break

    print(f'#{tc}',res)

'''


# 1983 조교의 성적 매기기 
'''
rank=['A+', 'A0', 'A-', 'B+', 'B0', 'B-', 'C+', 'C0', 'C-', 'D0']
T=int(input())

for tc in range(1,T+1):   
    n,k=map(int,input().split())
    scr=[0]*n
    cnt=n//10

    for i in range(n):
        s1,s2,s3=map(int,input().split())
        scr[i]=((s1*0.35)+(s2*0.45)+(s3*0.2))


    sorted_lst=sorted(scr,reverse=True)
    print(f'#{tc}',rank[sorted_lst.index(scr[k-1])//cnt])
    
'''

# 1979 어디에 단어가 들어갈 수 있을까




# 1954 달팽이 숫자

'''
N * N 배열 할당 (FALSE 초기화)

동 -> 남 -> 서 -> 북 차례로 이동하며 범위를 이동하거나 0이 아닌 좌표를 만나는 경우  방향 변경하며 n*n번 초기화 진행 

'''

'''
T=int(input())

for tc in range(1,T+1):

    # 우,하,좌,상
    dist=0  # 0:우, 1:하, 2:좌, 3:상
    dr=[0,1,0,-1]
    dc=[1,0,-1,0]

    n=int(input())
    arr=[[0]*n for _ in range(n)]

    # 가로, 세로 좌표
    r,c=0,0

    for num in range(1,n*n+1):
        arr[r][c]=num
        r+=dr[dist]
        c+=dc[dist]
        
        # 범위 밖으로 좌표가 이동하거나 다음 좌표가 앞선 단계에서 할당 받은 좌표인 경우 -> 방향 변경
        if r<0 or c<0 or r>=n or c>=n or arr[r][c]!=0:
            
            # 되돌리기
            r-=dr[dist]
            c-=dc[dist]

            # 방향 변경
            dist=(dist+1)%4   # %를 이용해서 인덱스 범위내에 값이 존재하게 설정

            # 재진행
            r+=dr[dist]
            c+=dc[dist]

    print(f'#{tc}')
    for x in arr:
        print(*x)
        
'''       


################### D3 ###################

# 3431 준환이의 운동관리

'''
T = int(input())

for tc in range(1, T + 1):
    L,U,X=map(int,input().split())

    if X>U:
        print(f'#{tc}',-1)
    elif L<=X<=U:
        print(f'#{tc}',0)
    else:
        print(f'#{tc}',L-X)

'''

# 13218 조별과제

'''
T = int(input())

for tc in range(1, T + 1):
    n=int(input())

    if n<3:
        print(f'#{tc}',0)
    else:
        print(f'#{tc}',n//3)
'''

# 10505 소득 불균형
'''
T = int(input())

for tc in range(1, T + 1):
    cnt=0
    n=int(input())
    lst=list(map(int,input().split()))
    for x in lst:
        if x<=(sum(lst)/len(lst)):
            cnt+=1

    print(f'#{tc}',cnt)

'''

# 4406 모음이 보이지 않는 사람
'''
T = int(input())

for tc in range(1, T + 1):
    s=input()
    check=['a','e','i','o','u']
    res=''
    for i in range(len(s)):
        if s[i] not in check:
            res+=s[i]

    print(f'#{tc}',res)

'''

# 1289 원재의 메모리 복구하기
'''
T = int(input())

for tc in range(1, T + 1):
    lst=list(map(int,input()))

    check_num=0
    cnt=0

    for x in lst:
        if x!=check_num:
            cnt+=1
            if check_num==1:
                check_num=0
            else:
                check_num=1
                
    print(f'#{tc}',cnt)

'''


# 11688 Calkin-Wilf tree 1

'''
T = int(input())

for tc in range(1, T + 1):
    s=list(input())

    a=1
    b=1
    root=a/b
    for x in s:
        if x=='L':
            b=(a+b)   
        else:
            a=(a+b)
            
    print(f'#{tc}',a,b)
        
'''


# 1217 [S/W 문제해결 기본] 4일차 - 거듭 제곱

'''
t=int(input())
n,m=map(int,input().split())
cnt=0

def recusive(a,b):
    if b==1:
        return a

    else:
        return a*recusive(a,b-1)

print(f'#{t}',recusive(n,m))

'''

# 10570 제곱 팰린드롬 수

'''
T = int(input())

for tc in range(1, T + 1):
    a,b=map(int,input().split())
    cnt=0

    def pal_fun(num):
        lst=list(str(num))
        if len(lst)==1:
            return True

        else:
            max_idx=len(lst)//2
            for i in range(max_idx):
                if lst[i]!=lst[len(lst)-1-i]:
                   return False
            else:
                return True

    for num in range(a,b+1):
        if pal_fun(num)==True:
            if int(num**(1/2))==num**(1/2):
                if pal_fun(int(num**(1/2)))==True:
                    cnt+=1

    print(f'#{tc}',cnt)

'''


# 5601 [Professional] 쥬스 나누기
'''
t=int(input())

for tc in range(1,t+1):
    n=int(input())
    print(f'#{tc} ', end='')
    print(('1/'+str(n)+' ')*n)

'''

# 1220 [S/W 문제해결 기본] 5일차 - Magnetic 

'''
# 파란 1: S, 빨간 2: N
for tc in range(10):
    n=int(input())
    arr=[]
    cnt=0

    for _ in range(n):
        arr.append(list(map(int,input().split())))


    for j in range(n):
        cur_num=0
        for i in range(n):
            if arr[i][j]==1: # 열의 위에서 부터 가장 가까운 N극 
                cur_num=arr[i][j]

            if cur_num==1 and arr[i][j]==2:
                cur_num=arr[i][j]
                cnt+=1

    print(f'#{tc}',cnt)

'''

# 1225 [S/W 문제해결 기본] 7일차 - 암호생성기

'''
for tc in range(1,11):
    n=int(input())
    lst=list(map(int,input().split()))

    cnt=1

    while True:
        num=lst[0]-cnt

        if num<=0:
            num=0
            lst.append(num)
            del lst[0]
            break

        else:
            lst.append(num)
            del lst[0]
            cnt+=1

            if cnt>5:
                cnt=1

    print(f'#{tc}',*lst)


'''


# 13229 일요일
'''
dic={'MON':1,'TUE':2,'WED':3,'THU':4,'FRI':5, 'SAT':6, 'SUN':7}

n=int(input())
day=input()
print(dic['SUN']-dic[day])

'''


# 5431 민석이의 과제 체크하기 
'''
t=int(input())
for tc in range(1,t+1):

    n,m=map(int,input().split())
    check=list(map(int,input().split()))
    res=[]

    for p in range(1,n+1):
        if p not in check:
            res.append(p)

    print(f'#{tc}',*res)

'''


# 19113 식료품 가게
# 정렬 수행 후 순차 접근해서 정상가(x/0.75)가 리스트에 존재한다면 할인가,정상가 관계이므로 정상가 값을 0으로 값 초기화

'''
t=int(input())

for tc in range(1,t+1):
    n=int(input())
    lst=list(map(int,input().split()))
    lst.sort()
    ans=[]

    for x in lst:
        if x!=0 and x/0.75 in lst:
            ans.append(x)
            lst[lst.index(x/0.75)]=0
    
    print(f'#{tc}',*ans)

'''

# 3314 보충학습과 평균

'''
t=int(input())
lst=list(map(int,input().split()))

for i in range(len(lst)):
    if lst[i]<40:
        lst[i]=40

print(f'#{t}',int(sum(lst)/5))

'''



# 1234 [S/W 문제해결 기본] 10일차 - 비밀번호

'''
for tc in range(1,11):
    n,num=map(int,input().split())
    lst=list(map(int,str(num)))

    idx=0

    while True:
        if idx>=len(lst):
            break


        if idx+1<len(lst):
            if lst[idx]==lst[idx+1]:
                del lst[idx:idx+2]
                idx=0
                continue

        idx+=1

    ans=0
    cnt=10**(len(lst)-1)

    for x in lst:
        ans+=(x*cnt)
        cnt//=10
        
    print(f'#{tc}',ans)

'''


# 1213. [S/W 문제해결 기본] 3일차 - String
'''
for tc in range(1,11):
    n=int(input())
    find_w=input()
    string=input()

    print(f'#{tc}',string.count(find_w))

'''



# 2805 농작물 수확하기
'''
t=int(input())
for tc in range(1,t+1):
    n=int(input())
    arr=[]

    for _ in range(n):
        arr.append(list(map(int,input())))

    mid=n//2
    cnt=0
    row=0
    answer=0

    for x in arr:
        answer+=sum(x[mid-cnt:mid+cnt+1])
        
        if row<mid:
            cnt+=1
        else:
            cnt-=1
        row+=1

    print(f'#{tc}',answer)
    
'''



# 2814 최장 경로 [풀이 실패]

'''
T=int(input())

for tc in range(1,T+1)
    n,m=map(int,input().split())
    graph=[[] for _ in range(n+1)]  # 인접리스트

    for _ in range(m):
        s,e=map(int,input().split())    # 무방향
        graph[s].append(e)
        graph[e].append(s)


    result=0  # 정답 

    def dfs(start,cnt):
        global result
        
        # 방문처리
        visited[start]=True

        # 방문하지 않은 인접 원소 -> dfs 재귀 홀출
        for end in graph[start]:
            if visited[end]!=True:
                dfs(end,cnt+1)

        result=max(result,cnt)




    for start in range(1,n+1):
        cnt=0
        visited=[False]*(n+1)
        dfs(start,1)

    print(f'#{tc}',result)


'''


# 2814 최장 경로 [풀이중]

'''
문제 ex)
     3
    /
   2
  /
1 ㅡ 4ㅡ5
   
일반적인 bfs
-> 인접 노드를 재귀적으로 호출하며 방문체크, 더이상 인접한 노드가 없을 시 처음 지점에서 다른 방향 탐색(1->2->3->4->5) 
-> 실행결과: 5 (노드탐색)

해당 문제
-> 막다른 길 도달 시, 경로 추가 불가(1->2->3  ,,4(x)), 노드의 시작점에 따라 값이 바뀜
-> 실행결과: 3 (최장 경로 탐색)
-> 문제에서 원하는 값: 5 (3->2->1->4->5 or 5->4->3->2->1)

일반적인 bfs에서 변경해야할 점
-> 막다른 길에서 돌아오지 못하게 해야함
-> 모든 노드에 대해 경로 검색

궁금한점: 모든 노드에 대해 검색한다고 하여도 각각 노드의 갈림길에서 최선의 선택은 어떻게 함 ,,?
궁금 ex)

     3
    /
   2 - 6
  /
1 ㅡ 4ㅡ5      >> 2에서 3으로 가야할지 6으로 가야할지에 대한 판단은 어떻게 ,,?


https://flex2020.com/posts/7

'''

'''
T=int(input())

for tc in range(1,T+1):
    n,m=map(int,input().split())
    graph=[[] for _ in range(n+1)]

    for _ in range(m):
        v1,v2=map(int,input().split())
        graph[v1].append(v2)
        graph[v2].append(v1)
        

    visited=[False]*(n+1)
    cnt=0

    def dfs(v):
        global cnt
        
        visited[v]=True
        cnt+=1

        for j in graph[v]:
            if visited[j]!=True:
                dfs(j)

    dfs(1)
    print(f'#{tc}',cnt)


'''



# 1244 [S/W 문제해결 응용] 2일차 - 최대 상금 [풀이중]

'''
T=int(input())
for tc in range(1,T+1):
    numbers,cnt=map(int,input().split())
    lst=list(map(int,str(numbers)))

    for i in range(len(lst)):
        # 교체 잔여 횟수가 0이라면 종료
        if cnt==0:
            break
        
        if lst[i]==max(lst[i:]):
            pass
        else:
            temp=lst[i]
            lst[i]=max(lst[i:])
            for j in range(len(lst)-1,i,-1):    # 반복할때마다 가장 큰 숫자가 여러개인 경우 일의 자리부터 교체
                if lst[j]==max(lst[i:]):
                    lst[j]=temp
                    break
            cnt-=1  # 횟수 1감소



    # 리스트 전체에서 가장 큰 숫자가 여러개 일때 -> 가장 큰 숫자의 개수만큼 lst 하위 원소 오름차순 정렬
    max_cnt=lst.count(max(lst))
    low_lst=[]
    if max_cnt>1:
        for _ in range(max_cnt):
            num=lst.pop()
            low_lst.append(num)
        low_lst.sort(reverse=True)
        lst+=low_lst


    가장 큰 숫자가 여러개 붙어 있는 경우 문제 발생  -> 알아서 정렬
    EX) 21399, 2 -> 91392 -> 99312 (99321 x)



    # 가장 큰값으로 만들었는데, 교체 잔여 횟수가 남아있는 경우
    if cnt>0:
        if cnt%2!=0: # 홀수인경우(일의 자리와 십의 자리 숫자 교체)
            temp=lst[len(lst)-1]
            lst[len(lst)-1]=lst[len(lst)-2]
            lst[len(lst)-2]=temp
        

    ans=0
    cnt=10**(len(lst)-1)
    for x in lst:
        ans+=(x*cnt)
        cnt/=10
        
    print(f'#{tc}',int(ans))    
    #print(f'#{tc} ',*lst,sep='')

'''


# 20551 증가하는 사탕 순열

'''
T=int(input())
for tc in range(1,T+1):
    a,b,c=map(int,input().split())
    cnt=0

    if b<2 or c<b:
        answer=-1
    
    elif a<b<c:
        answer=0

    else:
        if c<=b:
            cnt+=(b-c+1) 
            b-=(b-c+1)

        if a>=b:
            cnt+=(a-b+1)
        answer=cnt

    print(f'#{tc}',answer)

'''
# 9280 진용이네 주차타워 [풀이중]

'''
#(3*2)+(1*3)+(2*2)+(5*8)=6+3+4+40=53
 
from collections import deque
 
T=int(input())
 
for tc in range(1,T+1):
    n,m=map(int,input().split())
    R_i=[] # 주차공간 시간당 비용 리스트
    W_i=[] # 자동차 무게 리스트
    money=0
 
    for _ in range(n):
        R_i.append(int(input()))
         
    for _ in range(m):
        W_i.append(int(input()))
 
 
    # 대기차량 - FIFO
    q=deque()
 
 
    park=[0]*n
 
    for _ in range(2*m):
        w=int(input())
 
        # 자동차 들어올 때
        if w>0:
            if 0 in park:   # 빈자리가 있다면 순서대로 집어넣음
                for i in range(m):
                    if park[i]==0:
                        park[i]=W_i[w-1]
                        break
            else:
                q.append(w)
 
        # 자동차 나갈 때
        if w<0:
            w*=(-1)      
            money+=(R_i[park.index(W_i[w-1])]*W_i[w-1])
            left_idx=park.index(W_i[w-1])
            park[left_idx]=0
 
            if len(q)>0:    # 차가 나갈때마다 한자리씩 자리가 나게 됨
                left_w=W_i[q.popleft()-1]
                park[left_idx]=left_w
 
 
    print(f'#{tc}',money)

'''

# 4047 영준이의 카드 카운팅 [내 풀이]

'''
T=int(input())
for tc in range(1,T+1):
    
    S_dic={'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,'10':0,'11':0,'12':0,'13':0}
    D_dic={'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,'10':0,'11':0,'12':0,'13':0}
    H_dic={'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,'10':0,'11':0,'12':0,'13':0}
    C_dic={'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,'10':0,'11':0,'12':0,'13':0}

    lst=[]
    TXY=input()
    idx=0
    for _ in range(len(TXY)//3):
        lst.append(TXY[idx:idx+3])
        idx+=3

    for x in lst:
        type=x[0]
        num=x[1]+x[2]

        if type=='S':
            S_dic[num]+=1
        elif type=='D':
            D_dic[num]+=1
        elif type=='H':
            H_dic[num]+=1
        else:
            C_dic[num]+=1


    if max(S_dic.values())>1 or max(D_dic.values())>1 or max(H_dic.values())>1 or max(C_dic.values())>1:
        print(f'#{tc}','ERROR')
        continue
        
    else:
        S_num=list(S_dic.values()).count(0)
        D_num=list(D_dic.values()).count(0)
        H_num=list(H_dic.values()).count(0)
        C_num=list(C_dic.values()).count(0)
        print(f'#{tc}',S_num,D_num,H_num,C_num)

'''      

# 4047 영준이의 카드 카운팅 [인터넷 풀이 _ 효율적]
# 중복검사->set함수를 적용한 후 len으로 검사
# 하나하나씩 숫자를 확인하지 않고 해당하는 문자의 딕셔너리 값을 1씩 감소시켜줌

'''
from collections import OrderedDict

T = int(input())

for test_case in range(1, T + 1):
    s = input()
    cards = [ s[x:x+3] for x in range(0, len(s), 3) ]
    
    if len(set(cards)) != len(cards):
        print(f'#{test_case} ERROR')
        
    else:
        counts = OrderedDict({ 'S': 13, 'D': 13, 'H': 13, 'C':13 })

        for card in cards:
            counts[card[0]] -= 1
        
        print(f'#{test_case}', end=' ')
        for t in counts.values():
            print(t, end=' ')
        print()

'''

# 20728. 공평한 분배 2
'''
T=int(input())
for tc in range(1,T+1):
    n,m=map(int,input().split())
    lst=list(map(int,input().split()))
    lst.sort()
    max_gap=1e9

    for i in range(0,len(lst)-m+1):
        gap=lst[i+m-1]-lst[i]
        if gap<max_gap:
            max_gap=gap

    print(f'#{tc}',max_gap)

'''



# 3456. 직사각형 길이 찾기
'''
T=int(input())
for tc in range(1,T+1):
    lst=list(map(int,input().split()))

    for x in lst:
        if lst.count(x)==1 or lst.count(x)==3:
            result=x
            break

    print(f'#{tc}',result)
'''


# 5549. 홀수일까 짝수일까
'''
T=int(input())
for tc in range(1,T+1):
    num=int(input())

    if num%2==0:
        res='Alice'
    else:
        res='Bob'

    print(f'#{tc}',res)

'''

# 14178. 1차원 정원
'''
T=int(input())
for tc in range(1,T+1):
    n,d=map(int,input().split())

    if n%(2*d+1)==0:
        res=n//(2*d+1)
    else:
        res=n//(2*d+1)+1
        
    print(f'#{tc}',res)

'''


# 14555. 공과 잡초 [풀이중]
# 접근 방법이 지저분(문제도 지저분 - 반쪽 짜리 공 절대 안들어옴) -> ().(|,|) 세주면 끝
'''
T=int(input())
for tc in range(1,T+1):
    lst=input()
    stack=[]
    cnt=0
    flag=False
    if lst[0]=='|' or lst[0]=='(':
        flag=True
    stack.append(lst[0])
    
    
    for i in range(1,len(lst)):
        if flag==True:
            if (stack[-1]=='|' and lst[i]==')') or (stack[-1]=='(' and lst[i]==')') or stack[-1]=='(' and lst[i]=='|':
                cnt+=1
                flag=False

        else:
            if lst[i]=='|' or lst[0]=='(':
                flag=True

        stack.append(lst[i])

    print(f'#{tc}',cnt)

'''

# 1208. [S/W 문제해결 기본] 1일차 - Flatten

'''
T=10
for tc in range(1,T+1):
    lst=[1,3,4,14]
    n=int(input())

    for _ in range(n):
        lst[lst.index(min(lst))]+=1
        lst[lst.index(max(lst))]-=1

    print(f'#{tc}',max(lst)-min(lst))

'''  


# 3142 영준이와 신비한 뿔의 숲

'''
T=int(input())
for tc in range(1,T+1):
    n,m=map(int,input().split())
    uni=0
    twin=0

    while True:
        if n%2==0:
            twin=(n//2)
            if twin+uni==m:
                break
            else:
                uni+=2
                n-=2
        else:
            uni+=1
            n-=1
            if twin+uni==m:
                break
        

    print(f'#{tc}',uni,twin)
            
'''

# 5162. 두가지 빵의 딜레마
'''
T=int(input())
for tc in range(1,T+1):
    a,b,c=map(int,input().split())

    if a>b:
        a,b=b,a

    print(f'#{tc}',(c//a))

'''

# 10200. 구독자 전쟁
'''
T=int(input())
for tc in range(1,T+1):
    n,a,b=map(int,input().split())
    result=(a+b-n)
    
    if result<0:
        print(f'#{tc}',max(min(a,b),0),min(min(a,b),0))
        

    else:
        print(f'#{tc}',max(min(a,b),result),min(min(a,b),result))

'''




# 11856 반반
'''
T=int(input())
for tc in range(1,T+1):
    lst=list(map(str,input()))
    lst.sort()
    answer='No'
    

    if lst[0]==lst[1] and lst[2]==lst[3] and lst[1]!=lst[3]:
        answer='Yes'
        

    print(f'#{tc}',answer)
'''

# 3499. 퍼펙트 셔플
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=list(input().split())
    print(f'#{tc}',end=' ')

    if n%2==0:
        for i in range(n//2):
            print(lst[i],lst[i+n//2],end=' ')
        print()
    else:
        for i in range(n//2):
            print(lst[i],lst[i+1+n//2],end=' ')
        print(lst[n//2])

'''


# 5515. 2016년 요일 맞추기
'''
T=int(input())
for tc in range(1,T+1):
    mon=[31,29,31,30,31,30,31,31,30,31,30,31]
    day=[0,1,2,3,4,5,6]

    m,d=map(int,input().split())
    total=(sum(mon[:m-1])+d)

    print(f'#{tc}',(day.index(4)-1+total)%7)

'''


# 5356. 의석이의 세로로 말해요
'''
T=int(input())
for tc in range(1,T+1):
    lst=[]
    max_len=0
    ans=''

    for i in range(5):
        string=input()
        lst.append(string)
        max_len=max(max_len,len(string))

    for i in range(5):
        lst[i]=lst[i].ljust(max_len,' ')


    
    for j in range(max_len):
        for i in range(5):
            if lst[i][j]==' ':
                pass
            else:
                ans+=lst[i][j]
                
     
    print(f'#{tc}',ans)

'''


# 1221. [S/W 문제해결 기본] 5일차 - GNS
'''
T=int(input())
for _ in range(1,T+1):
    tc,n=input().split()

    dic={"ZRO":0, "ONE":1, "TWO":2, "THR":3, "FOR":4, "FIV":5, "SIX":6, "SVN":7, "EGT":8, "NIN":9}
    lst=list(input().split())
    lst.sort(key=lambda x:dic[x])
    print(f'{tc}',*lst)

'''

# 1206 [S/W 문제해결 기본] 1일차 - View
'''
T=10
for tc in range(1,T+1):
    n=int(input())
    lst=list(map(int,input().split()))
    ans=0

    for i in range(2,n-2):
        if lst[i]==max(lst[i-2:i+3]) and lst[i-2:i+3].count(lst[i])==1:
            H=lst[i]
            dist=lst[i-2:i+3]
            dist.remove(lst[i])
            next_H=max(dist)
            ans+=H-next_H


    print(f'#{tc}',ans)
'''


# 5789. 현주의 상자 바꾸기
'''
T=int(input())
for _ in range(1,T+1):
    N,Q=map(int,input().split())
    lst=[0]*N

    for i in range(1,Q+1):
        L,R=map(int,input().split())
        for j in range(L-1,R):
            lst[j]=i

    print(f'#{tc}',*lst)

'''

# 4676. 늘어지는 소리 만들기
'''
T=int(input())
for tc in range(1,T+1):
    lst=list(map(str,input()))
    n=int(input())
    place=list(map(int,input().split()))
    dic={}
    cnt=0 # 밀리는 문자 위치

    for x in place:
        if x not in dic:
            dic[x]=1
        else:
            dic[x]+=1

    dic=dict(sorted(dic.items(),key=lambda x:x[0]))


    for k,v in dic.items():
        lst.insert(k+cnt,'-'*v)
        cnt+=1
        
    print(f'#{tc}',''.join(lst))

>> 제일 큰 인덱스 부터 삽입하면 밀리는 위치 생각안해도 됨

'''

# 10912. 외로운 문자
'''
T=int(input())
for tc in range(1,T+1):
    word=list(input())
    word.sort()
    stack=[]

    for i in range(len(word)):
        if len(stack)==0:
            stack.append(word[i])
        else:
            if stack[-1]==word[i]:
                stack.pop()
            else:
                stack.append(word[i])

    if len(stack)==0:
        print(f'#{tc}','Good')
    else:
        print(f'#{tc} ',*stack,sep='')
'''

# 5603. [Professional] 건초더미
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=[]
    
    for _ in range(n):
        lst.append(int(input()))
        
    avg=sum(lst)//len(lst)
    gap=0
    for x in lst:
        gap+=abs(x-avg)
    print(f'#{tc}',gap//2)

''' 


# 1493. 수의 새로운 연산 [풀이중]
'''
graph=[[0]*10 for _ in range(10)]
graph[0][0]=1
row_cnt=1
col_cnt=2

for i in range(len(graph)-1):
    for j in range(len(graph)-1):
        if i==0 and j==0:
            pass
        else:
            if i==0:
                graph[i][j]=graph[i][j-1]+col_cnt
                col_cnt+=1
            elif j==0:
                graph[i][j]=graph[i-1][j]+row_cnt
                row_cnt+=1
            else:
                graph[i][j]=graph[i-1][j+1]-1
                

for i in range(len(graph)):
    print(graph[i])

'''

# 5948. 새샘이의 7-3-5 게임
'''
from itertools import combinations

T=int(input())
for tc in range(1,T+1):
    lst=list(map(int,input().split()))

    com_lst=list(combinations(lst,3))

    for i in range(len(com_lst)):
        com_lst[i]=sum(com_lst[i])

    com_lst=list(set(com_lst))
    com_lst.sort(reverse=True)

    print(f'#{tc}',com_lst[4])

'''

# 12004. 구구단 1
'''
T=int(input())
for tc in range(1,T+1):
    flag=False
    ans='No'
    num=int(input())
    for i in range(9,0,-1):
        if num%i==0:
            j=num//i
            if j<10:
                ans='Yes'
                break

    print(f'#{tc}',ans)

'''

# 9480. 민정이와 광직이의 알파벳 공부 [풀이중]

'''
from itertools import combinations

n=int(input())
lst=[]
com=[]
dic={}

for _ in range(n):
    lst.append(input())


for i in range(1,n+1):
    com.append(list(combinations(lst,i)))

print(com)

'''


# 15230 알파벳 공부

'''
T=int(input())
for tc in range(1,T+1):
    check='abcdefghijklmnopqrstuvwxyz'

    word=input()
    cnt=0

    for i in range(len(word)):
        if check[i]==word[i]:
            cnt+=1
        else:
            break

    print(f'#{tc}',cnt)

''' 

# 10726 이진수 표현

'''
T=int(input())
for tc in range(1,T+1):
    n,m=map(int,input().split())
    lst=list(map(int,bin(m)[2:]))

    if sum(lst[len(lst)-n:])==n:
        ans='ON'
    else:
        ans='OFF'

    print(f'#{tc}',ans)

'''

# 3307. 최장 증가 부분 수열
# 모든 0<=j<i에 대하여, D[i]=max(D[i],D[j]+1) if lst[j]<=lst[i]

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=list(map(int,input().split()))
    dp=[1]*(len(lst))

    for i in range(1,len(lst)):
        for j in range(i):
            if lst[i]>=lst[j]:
                dp[i]=max(dp[j]+1,dp[i])

    print(f'#{tc}',max(dp))

'''


# 16800 구구단 걷기 

'''
T=int(input())
for tc in range(1,T+1):
    ans=1e15
    n=int(input())

    for i in range(1,int(n**(1/2))+1):
        if n%i==0:
            j=n//i

            ans=min((i-1)+(j-1),ans)
            print(i,j)

    print(f'#{tc}',ans)

'''


# 6019 기차 사이의 파리

'''
T=int(input())
for tc in range(1,T+1):
    D,A,B,F=map(int,input().split())

    print(f'#{tc}',D/(A+B)*F)

'''


# 10580 전봇대

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=[]
    cnt=0

    for _ in range(n):
        lst.append(list(map(int,input().split())))


    for i in range(len(lst)-1):
        for j in range(i+1,len(lst)):
            x1,y1=lst[i][0],lst[i][1]
            x2,y2=lst[j][0],lst[j][1]

            if x1<x2 and y1>y2 or x1>x2 and y1<y2:
                cnt+=1


    print(f'#{tc}',cnt)

'''

# 20019 회문의 회문
'''
T=int(input())
for tc in range(1,T+1):
    s=input()
    ans='NO'

    if s==s[::-1]:
        # 홀수
        if len(s)%2!=0:
            mid_idx=len(s)//2
            if s[:mid_idx]==s[mid_idx-1::-1] and s[mid_idx+1:]==s[-1:mid_idx:-1]:
                ans='Yes'
        # 짝수
        else:
            mid_idx=len(s)//2
            if s[:mid_idx]==s[mid_idx-1::-1] and s[mid_idx:]==s[:mid_idx:-1]:
                ans='Yes'

    print(f'#{tc}',ans)

'''

# 3233 정삼각형 분할 놀이
'''
T=int(input())
for tc in range(1,T+1):
    a,b=map(int,input().split())
    a,b=a//b,b//b
    ans=0

    for i in range(a):
        ans+=(2*i+1)

    print(f'#{tc}',ans)
    
'''

# 17319. 문자열문자열
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    s=input()
    ans='No'

    if len(s)%2==0:
        if s[:len(s)//2]==s[len(s)//2:]:
            ans='Yes'

    print(f'#{tc}',ans)

'''

# 3376 파도반 수열
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    dp=[0]*101

    dp[1]=1
    dp[2]=1
    dp[3]=1

    for i in range(4,101):
        dp[i]=dp[i-2]+dp[i-3]

    print(f'#{tc}',dp[n])

'''


# 싸피 11 기출
'''
n,k=map(int,input().split())    # n: 원소 개수, k: 거리
lst=list(map(int,input().split()))
ans=0   # 정답

for i in range(len(lst)-1):
    sub_lst=[lst[i]]    # 기준 원소 삽입한 서브 리스트

    for j in range(i+1,len(lst)):  # 기준 원소 한칸 옆 ~ 끝까지
        sub_lst.append(lst[j])  # 일단 원소 하나 넣고
        
        if max(sub_lst)-min(sub_lst)>k:  # 서브 리스트의 가장 큰 값과 작은 값 차이가 K 이상이라면,  서브 리스트에서 하나 뺀(방금 넣은 원소) 원소 개수와 ans에 저장된 값 중 큰 값 갱신
            ans=max(ans,len(sub_lst)-1)
            break
    else:    # for 문이 멈추지 않고 순회 다했을 때 (if문에 걸리지 않고 끝까지 돌았다면), ans값 확인 후 갱신
        ans=max(ans,len(sub_lst))
        

print(ans)
            
'''       
        

# 3809 화섭이의 정수 나열
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=list(map(str,input().split()))
    s=''
    while True:
        s+=map(''.join(maP
    num=1

    while True:
        if str(num) in s:
            num+=1
            pass

        else:
            print(f'#{tc}',num)
            break

'''
'''
n=10
value=''

while True :
    value += ''.join(map(str, input().split()))
    if len(value) == n :
        break

print(value)
'''



# 4789 성공적인 공연 기획

'''
T=int(input())
for tc in range(1,T+1):
    lst=list(map(int,input()))
    p=lst[0]
    hire_p=0
    for i in range(1,len(lst)):
        if lst[i]==0:
            pass
        else:
            if p>=i:
                p+=lst[i]
            else:
                hire_p+=(i-p)
                p=i+lst[i]

    print(f'#{tc}',hire_p)

'''


# 14361 숫자가 같은 배수
'''
T=int(input())
for tc in range(1,T+1):
    num1=int(input())
    lst1=list(map(int,str(num1)))
    lst1.sort(reverse=True)
    max_num=int(''.join(map(str,lst1)))
    ans='impossible'

    num2=num1
    cnt=1
    while True:
        cnt+=1
        num2=num1*cnt
        if num2>max_num:
            break 
        else:
            lst2=sorted(list(map(int,str(num2))),reverse=True)
            if lst1==lst2:
                ans='possible'
                break
            
    print(f'#{tc}',ans)
            
'''



# 1860 진기의 최고급 붕어빵
'''
T=int(input())
for tc in range(1,T+1):
    p,s,b=map(int,input().split())  # p: 사람수, s: 제작시간, b: 제작개수
    p_lst=list(map(int,input().split()))
    p_lst.sort()
    b_lst=[0]*(max(p_lst)+1)
    ans='possible'

    for i in range(1,len(b_lst)):
        b_lst[i]=(i//s)*b

    for j in p_lst:
        if b_lst[j]<=0:
            ans='impossible'
            break
        else:
            for k in range(j,len(b_lst)):
                b_lst[k]-=1

    print(f'#{tc}',ans)

'''

# 5215 햄버거 다이어트 [2차원 배열]
'''
T=int(input())
for tc in range(1,T+1):
    N,L=map(int,input().split())    # N: 아이템 개수, W: 배낭 무게 제한

    item=[[0,0]]   # 아이템 정보 2차원 리스트
    for _ in range(N):
         item.append(list(map(int,input().split())))

    # DP 테이블 (행:N+1, 열:W+1) 
    dp=[[0 for _ in range(L+1)] for _ in range(N+1)]


    for i in range(1,N+1):
        for j in range(1,L+1):  # j: weight 범위(0~W)
            scr=item[i][0]
            kal=item[i][1]

            if j<kal:    # (j-wt)가 음수인 경우 넣을 수 없음 -> 넣지 않기
                dp[i][j]=dp[i-1][j]
                
            else:   # 넣을 수 있는 경우 -> 넣고 이득 더한 값과 넣지 않았을 때 값 비교 -> Max값 
                dp[i][j]=max(dp[i-1][j],dp[i-1][j-kal]+scr)
            
            

    print(f'#{tc}',dp[N][L])

'''

# 5215 햄버거 다이어트 [1차원 배열]

'''
N,K=map(int,input().split())
item=[]

for _ in range(N):
    v,w=map(int,input().split())
    item.append((v,w))

dp=[0]*(K+1)

for i in range(N):
    v=item[i][0]
    w=item[i][1]
    for j in range(K,w-1,-1):
        dp[j]=max(dp[j],dp[j-w]+v)

print(dp[-1])

'''


# 3282 0/1 Knapsack
'''
T=int(input())
for tc in range(1,T+1):
        
    n,k=map(int,input().split())

    items=[]
    for _ in range(n):
        w,v=map(int,input().split())
        items.append((w,v))

    dp=[0]*(k+1)

    for item in items:
        w,v=item
        for j in range(k,w-1,-1):
            dp[j]=max(dp[j],dp[j-w]+v)
        print(dp)
        
    print(f'#{tc}',dp[-1])

'''


# 5986 새샘이와 세 소수
'''
T=int(input())
for tc in range(1,T+1):
    num=int(input())
    cnt=0
    so_lst=[]   # 소수 리스트

    for i in range(2,1000):
        for j in range(2,i):
            if i%j==0:
                break
        else:
            so_lst.append(i)


    for i in range(len(so_lst)):
        for j in range(i,len(so_lst)):
            for k in range(j,len(so_lst)):
                if so_lst[i]+so_lst[j]+so_lst[k]==num:
                    cnt+=1
                if so_lst[i]+so_lst[j]+so_lst[k]>num:
                    break

    print(f'#{tc}',cnt)

'''



# 3975 승률 비교하기

'''
T=int(input())
for tc in range(1,T+1):
    a,b,c,d=map(int,input().split())

    if a/b>c/d:
        ans='ALICE'
    elif a/b<c/d:
        ans='BOB'
    else:
        ans='DRAW'

    
    print(f'#{tc}',ans)

'''


# 3304. 최장 공통 부분 수열 [시간초과]
# 단순 반복

'''
T=int(input())
for tc in range(1,T+1):
    
    w1,w2=input().split()
    str1=list(map(str,w1))
    str2=list(map(str,w2))
    max_len=0
    idx=-1
    s_len=0

    for k in range(len(str1)):
        for i in range(k,len(str1)):
            flag=False
            for j in range(len(str2)):
                if str1[i]==str2[j] and j>idx:
                    flag=True
                    s_len+=1
                    idx=j
                    max_len=max(max_len,s_len)
                    break
                    
      
    print(f'#{tc}',max_len)
                    
'''

# 3304. 최장 공통 부분 수열 [통과]
# LCS 알고리즘

'''
T=int(input())
for tc in range(1,T+1):
    a,b=input().split()
    dp=[[0 for _ in range(len(b)+1)] for _ in range(len(a)+1)]

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):

            if a[i-1]==b[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
                

    print(f'#{tc}',max(dp[-1]))

'''

# 5688 세제곱근을 찾아라

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    max_num=int(n**(1/3)+1)
    
    for num in range(1,max_num):
        if num**3==n:
            ans=num
            break
    else:
        ans=-1

    print(f'#{tc}',ans)


'''

# 6057 그래프의 삼각형 [시간초과]

'''
T=int(input())
for tc in range(1,T+1):
    n,m=map(int,input().split())
    lst=[]
    cnt=0
    
    for _ in range(m):
        lst.append(list(map(int,input().split())))


    for i in range(len(lst)-2):
        for j in range(i+1,len(lst)-1):
            for k in range(j+1,len(lst)):
                sum_p=lst[i]+lst[j]+lst[k]
                if len(set(sum_p))==3:
                    cnt+=1
                    break

                
    print(f'#{tc}',cnt)


'''

# 6057 그래프의 삼각형 [그려보기]

'''
T=int(input())
for tc in range(1,T+1):
    n,m=map(int,input().split())
    graph=[[] for _ in range(n+1)]
    cnt=0

    for _ in range(m):
        a,b=map(int,input().split())
        graph[a].append(b)
        graph[b].append(a)
    print(graph)


    for i in range(1,n+1):
        for j in range(i+1,n+1):
            for k in range(j+1,n+1):
                if i in graph[j] and j in graph[k] and k in graph[i]:
                    cnt+=1
    print(f'#{tc}',cnt)

'''

# 19003 팰린드롬 문제

'''
T=int(input())
for tc in range(1,T+1):
    n,m=map(int,input().split())
    lst=[]

    for _ in range(n):
        lst.append(input())

    pal_o=0    # 단어 자체가 팰린드롬
    pal_w=0    # 두 단어가 합쳐서 팰린드롬

    # 두 단어가 합쳐서 팰린드롬인 원소 카운트, 자기 자신이 팰린드롬인 경우도 같이 카운트되므로 먼저 단어자체가 팰린드롬일 때 먼저 if문 걸기
    for s in lst:
        if s==s[::-1]:    # 자기 자신이 팰린드롬이면서 다른 단어와 합쳐서 팰린드롬인 경우 존재x (문재애서 문자열 각각은 서로다르다함)
            pal_o+=1
     
            
        elif s[::-1] in lst:
            pal_w+=1


    if pal_w==0:
        if pal_o==0:
            ans=0
        else:
            ans=m
    elif pal_w!=0:
        if pal_o==0:
            ans=pal_w*m

        else:
            ans=(pal_w*m+m)
        

    print(f'#{tc}',ans)
        
'''



# 18662 등차수열 만들기

'''
- a가 변경하는 값이라 생각하고 만들었을 때, a 변경값

- b가 변경하는 값이라 생각하고 만들었을 때, b 변경값

- c가 변경하는 값이라 생각하고 만들었을 때, c 변경값


중 최솟값

'''

'''
T=int(input())
for tc in range(1,T+1):
    a,b,c=map(int,input().split())

    if b-a==c-b:
        ans=0
    else:
        fir=abs(a-(b-(c-b)))
        sec=abs(b-(a+c)/2)
        thr=abs(c-(b+(b-a)))
        ans=min(fir,sec,thr)

    print(f'#{tc} {ans:0.1f}')

'''

# 13732 정사각형 판정 [tc 20개 중 17개 통과 _ 뭐가 문제인지 모르겠다]
# 정사각형이 될 수 없는 변의 길이 추가해보기

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())   
    graph=[]
    for _ in range(n):
        graph.append(list(map(str,input())))

    square=[]   # '#' 좌표 리스트
    row=[]
    col=[]
    for i in range(n):
        for j in range(n):
            if graph[i][j]=='#':
                square.append((i,j))
                row.append(i)
                col.append(j)

    check=[]    # square 좌표 시작점 부터 끝점 기준으로 정사각형을 만드는 # 좌표 넣기 
    
    for i in range(min(row),max(row)+1):
        for j in range(min(col),max(col)+1):
            check.append((i,j))

    ans='no'
    
    # 일치하는지 확인
    if max(col)-min(row)==max(row)-min(col):
        if max(row)-min(row)==max(col)-min(col):
            if square==check:
                ans='yes'

    if len(square)==1:
        ans='yes'
        
    print(f'#{tc}', ans)
                
'''

# 13732 정사각형 판정 [답지_BFS 풀이법]

'''
from collections import deque

t=int(input())

def bfs(square,graph):
    len_sq=len(square)**0.5

    if len_sq%1!=0: # 정사각형 변의 길이를 이룰 수 있는 숫자가 아닐 때
        return 'no'

    x,y=square.popleft()    # 가장 작은 좌표 저장

    for i in range(x,x+int(len_sq)):
        for j in range(y,y+int(len_sq)):
            if graph[i][j]!='#':
                return 'no'
    return 'yes'


for tc in range(1,t+1):
    n=int(input())
    graph=list((input()) for _ in range(n))

    square=deque((i,j) for i in range(n) for j in range(n) if graph[i][j]=='#')

    ans=bfs(square,graph)
    print(ans)

'''
# 13428 숫자 조작 [실패]

'''
T=int(input())
for tc in range(1,T+1):
    lst=list(map(int,input()))

    # 최솟값
    def min_func(lst):
        num=lst[:]  # Point: 함수 인자의 리스트가 외부 리스트를 참조하고 있기 때문에 값만 복사해와서 별도의 리스트를 선언
        min_val=int(''.join(map(str,num)))

        if num==sorted(num):
            min_val=int(''.join(map(str,num)))
            pass
        else:
            for i in range(len(num)):
                if num[i]==min(num[i:]):
                    pass          
                else:
                    for j in range(len(num)-1,i,-1):
                        if num[j]==min(num[i:]):
                            if i==0 and min(num[i:])==0:
                                break
                            else:
                                num[i],num[j]=num[j],num[i]
                                min_val=int(''.join(map(str,num)))
                                break
                    break
        return min_val


    # 최댓값
    def max_func(lst):
        num=lst[:]
        max_val=int(''.join(map(str,num)))

        if num==sorted(num,reverse=True):
            max_val=int(''.join(map(str,num)))
            pass
        else:
            for i in range(len(num)):
                if num[i]==max(num[i:]):
                    pass          
                else:
                    for j in range(i+1,len(lst)):
                        if num[j]==max(num[i+1:]):
                            num[i],num[j]=num[j],num[i]
                            max_val=int(''.join(map(str,num)))
                            break
                    break
        return max_val

    print(f'#{tc}',min_func(lst),max_func(lst))

'''


# 13038 교환학생

# 내 코드 [실패]
# 내가 고려하지 못한 것: day:2, cls: 0 1 0 0 0 1 1 -> ans=2 (토, 일) -> 즉, 수업이 있는 모든날에 대해서 체크하고 최솟값 출력


'''
T=int(input())
for tc in range(1,T+1):
    day=int(input()) # 수강해야하는 날짜 수
    cls=list(map(int,input().split()))  # 수업 시간표
    ans=0

    week=day//cls.count(1)
    left_day=day%cls.count(1)


    ans+=week*7
    ans-=cls.index(1)   # 수업이 시작된 날 부터 다니면 되므로

    if left_day==0:
        ans=week*7
        ans-=cls.index(1)  # 수업이 시작한 요일
        for i in range(len(cls)-1,-1,-1):   # 수업이 끝난 요일
            if cls[i]==1:
                break
            else:
                ans-=1

              
    elif left_day>0:
        if week==0:
            for i in range(len(cls)):
                if cls[i]==1:
                    left_day-=1
                    if left_day==0:
                        ans=(i+1)
                        ans-=cls.index(1)

        else:                
            ans=week*7
            ans-=cls.index(1)  # 수업이 시작한 요일
            for i in range(len(cls)):
                if cls[i]==1:
                    left_day-=1
                    if left_day==0:
                        ans+=(i+1)
                        break

    print(f'#{tc}',ans)

'''


# 재풀이

'''
T=int(input())
for tc in range(1,T+1):
    day=int(input()) # 수강해야하는 날짜 수
    cls=list(map(int,input().split()))  # 수업 시간표
    min_ans=1e9

    st_lst=[]

    for i in range(len(cls)):
        if cls[i]==1:
            st_lst.append(i)

    for i in st_lst:
        check=0  # 수업 들은 숫자
        ans=0  # 누적 일수

        while True:
            print(i)
            if cls[i]==1:
                check+=1
            if i==6:
                i=0
            else:
                i+=1
            ans+=1

            if day==check:
                min_ans=min(min_ans,ans)
                break
                
    print(f'#{tc}',min_ans)

'''

# 12741 두 전구
# 내 코드1[시간초과]

'''
T=int(input())
for tc in range(1,T+1):
    ans=0

    arr=[0]*(101)

    a,b,c,d=map(int,input().split())

    for i in range(a,b+1):
        arr[i]+=1

    for i in range(c,d+1):
        arr[i]+=1

    if arr.count(2)==0:
        ans=0
    else:
        ans=arr.count(2)-1
        
    print(f'#{tc}',ans)

'''

# 내 코드 (출력, 입력 따로 문제 _ trash problem)
'''
T=int(input())
res=[]

for tc in range(1,T+1):
    a,b,c,d=map(int,input().split())
    t1=(a,b)
    t2=(c,d)

    if t1[0]>t2[0]:
        t1,t2=t2,t1    # 시작 시간: t1이 더 빠르게 바꿔줌
        

    if t1[1]<t2[0]:
        ans=0

    else:
        if t1[1]>t2[1]:
            ans=t2[1]-t2[0]
        else:
            ans=t1[1]-t2[0]

    res.append(ans)

for tc in range(1,T+1):
    print(f'#{tc}',res[tc-1])

'''

# 15758 무한 문자열

'''
T=int(input())
for tc in range(1,T+1):
    s,t=input().split()

    st_len=max(len(s),len(t))
    end_len=len(s)*len(t)

    for i in range(st_len,end_len+1):
        if i%len(s)==0 and i%len(t)==0:
            s_l=i//len(s)
            e_l=i//len(t)
            break

    if s*s_l==t*e_l:
        ans='yes'
    else:
        ans='no'

    print(f'#{tc}',ans)

'''


# 15612 체스판 위의 룩 배치

'''
T=int(input())
for tc in range(1,T+1):
    ans='yes'

    arr=[]
    for _ in range(8):
        arr.append(list(map(str,input())))


    for i in range(8):
        if arr[i].count('O')!=1:
            ans='no'
            break
        col_cnt=0
        for j in range(8):
            if arr[j][i]=='O':
                col_cnt+=1
        if col_cnt!=1:
            ans='no'
            break

    print(f'#{tc}',ans)
        
'''


# 12051 프리셀 통계 [풀이중]

'''
- 게임 횟수: N 이하
- 오늘 한 게임 수: D, 오늘 승률: Pd

-지금까지 한 게임 수: G, 지금까지 승률: Pg
-알고 있는것: N, Pd, Pg -> 유추해야 하는 것: D, G

오늘 승률 = 오늘 승수/오늘경기횟수 *100
전체 승률 = 승수/경기횟수 *100

-N번 이하의 게임 중 Pd,PG를 만족 시키는 D,G값이 존재할까? 


'''



# 11315 오목 판정 [85/100개 정답]
# https://chaemi720.tistory.com/49 -> 방향 인덱스 이용한 풀이

'''
T=int(input())
for tc in range(1,T+1):     
    n=int(input())
    ans='NO'
    arr=[]
    crs1=0
    crs2=0

    for _ in range(n):
        arr.append(list(map(str,input())))

    for i in range(n):
        row_cnt=0
        col_cnt=0
        for j in range(n):
            # 가로 빙고
            if arr[i][j]=='o':
                row_cnt+=0
                if row_cnt>=5:
                    ans='YES'
                    break
            else:
                row_cnt=0
                
            # 세로 빙고    
            if arr[j][i]=='o':
                col_cnt+=1
                if col_cnt>=5:
                    ans='YES'
                    break
            else:
                col_cnt=0

            # 대각선 빙고 1    
            if i==j:
                if arr[i][j]=='o':
                    crs1+=1
                    if crs1>=5:
                        ans='YES'
                else:
                    crs1=0
                    
            # 대각선 빙고 2
            if n-(j+1)==i:
                if arr[i][j]=='o':
                    crs2+=1
                    if crs2>=5:
                        ans='YES'
                        break
                else:
                    crs2=0
                    
        if col_cnt>=5 or row_cnt>=5 or crs1>=5 or crs2>=5:
            break


    print(f'#{tc}',ans)

'''


# 10965 제곱수 만들기 [시간초과 - 에라토스테네스체 공부하고 풀기]

'''
T=int(input())
res=[]
for tc in range(1,T+1):     
    a=int(input())
    b=1
    while True:
        num=a*b
        if num**(1/2)==int(num**(1/2)):
            res.append(b)
            break
        else:
            b+=1
            
for tc in range(1,T+1):
    print(f'#{tc}',res[tc-1])
    
'''


# 9700 USB 꽂기의 미스터리

'''
한번 뒤집고 성공시키는 법

- 잘못 꽂아서 뒤집고 작동 성공

두번 뒤집고  성공시키는 법

- 잘 꽂았는데 작동안해서 뒤집어 끼우면 안되니깐 다시 돌려서 작동 성공 


'''
'''
T=int(input())
for tc in range(1,T+1):
    p,q=map(float,input().split())

    s1=(1-p)*q
    s2=p*(1-q)*q

    if s2>s1:
        ans='YES'
    else:
        ans='NO'

    print(f'#{tc}',ans)

'''



# 1244 [S/W 문제해결 응용] 2일차 - 최대 상금 [풀이]

'''
- 당장의 최대값만 고려해서 가면 정답 나올 수 없음. 즉, 그리디 접근 x
ex) 13888 -> 83188 -> 881328 != 88318

- 재귀를 이용해서 완전탐색을 진행하되, 가지치기를 통해 불필요한 메모리 생략 -> n번 재귀를 호출했을 때 해당하는 최대값이 정답

'''
# 가치지기 안했을 때

'''
def dfs(n):
    global ans
    if n==N:
        ans=max(ans,int(''.join(map(str,lst))))
        return

    else:
        for i in range(len(lst)-1):
            for j in range(i+1,len(lst)):
                lst[i],lst[j]=lst[j],lst[i]
                dfs(n+1)
                lst[i],lst[j]=lst[j],lst[i]
        
        
num,N=map(int,input().split())
lst=list(map(int,str(num)))
ans=0
dfs(0)

print(ans)

'''

# 가지치기 방법1

'''
- 그림 그려서 이해하기
- 방문 리스트에 들어가있지 않는 경우 dfs 수행, dfs를 수행하고 나와서 방문처리 해주기

'''

'''
def dfs(n):
    global ans
    if n==N:
        ans=max(ans,int(''.join(map(str,lst))))
        return

    else:
        for i in range(len(lst)-1):
            for j in range(i+1,len(lst)):
                lst[i],lst[j]=lst[j],lst[i]
                num=int(''.join(map(str,lst)))
                if (n,num) not in v:
                    dfs(n+1)  # n+1이하의 모든 경우의 수를 탐색한 후 방문 리스트에 추가, dfs보다 앞에 추가할 경우 하위에 호출되는 모든 경우의 수를 방문해야함
                    v.append((n,num))
                lst[i],lst[j]=lst[j],lst[i]  # 반드시 원상복구 해줘야함, 모든 경우의 수를 확인해야하므로 (위치를 바꾼 리스트는 재귀에 포함되어 이어나감)
        
        
num,N=map(int,input().split())
lst=list(map(int,str(num)))
ans=0
v=[] # 방문 리스트
dfs(0)

print(ans)

'''

# 가지치기 방법2
# n값이 최대 10이므로 num*100+n 정수값 형태로 방문표시 체크

'''
def dfs(n):
    global ans
    if n==N:
        ans=max(ans,int(''.join(map(str,lst))))
        return

    else:
        for i in range(len(lst)-1):
            for j in range(i+1,len(lst)):
                lst[i],lst[j]=lst[j],lst[i]
                
                num=int(''.join(map(str,lst)))
                if num*100+n not in v:
                    dfs(n+1)  # n+1이하의 모든 경우의 수를 탐색한 후 방문 리스트에 추가, dfs보다 앞에 추가할 경우 하위에 호출되는 모든 경우의 수를 방문해야함
                    v.append(num*100+n)
                    
                lst[i],lst[j]=lst[j],lst[i]  # 반드시 원상복구 해줘야함, 모든 경우의 수를 확인해야하므로 (위치를 바꾼 리스트는 재귀에 포함되어 이어나감)
        
        
T=int(input())
for tc in range(1,T+1):
    num,N=map(int,input().split())
    lst=list(map(int,str(num)))
    ans=0
    v=[] # 방문 리스트
    dfs(0)

    print(f'#{tc}',ans)

'''


# 6485 삼성시의 버스 노선

'''
T=int(input())

for tc in range(1,T+1):   
    N=int(input())
    arr=[0]*5001
    res=[]

    for _ in range(N):
        a,b=map(int,input().split())
        for i in range(a,b+1):
            arr[i]+=1

    P=int(input())

    for _ in range(P):
        n=int(input())
        res.append(arr[n])

    print(f'#{tc}',*res)

''' 



# 6190 정곤이의 단조 증가하는 수

'''
def check(num):     # 증가 체크 함수
    num=list(map(int,str(num)))
    cnt=0
    cur_num=-1
    for i in range(len(num)-1):
        if num[i+1]<num[i]:
            return False
    return True


T=int(input())
for tc in range(1,T+1):          
    n=int(input())
    lst=list(map(int,input().split()))
    ans=-1
    v=[]
    
    for i in range(len(lst)-1):
        for j in range(i+1,len(lst)):   
            num=lst[i]*lst[j]
            if num not in v:
                if check(num):
                    ans=max(ans,num)
                    v.append(num)


    print(f'#{tc}',ans)

'''

# 7584 자가 복제 문자열 [시간초과]

'''

T=int(input())

for tc in range(1,T+1):   
    num='0'
    k=int(input())

    while True:
        if len(num)>k-1:
            break
        
        a_num=''
        for i in range(len(num)):
            if num[i]=='1':
                a_num+='0'
            else:
                a_num+='1'
        a_num=a_num[::-1]
        num=(num+'0'+a_num)
        

    print(f'#{tc}',num[k-1])

'''
    

# 4751 다솔이의 다이아몬드 장식
'''
T=int(input())
for tc in range(1,T+1):   
    lst=list(map(str,input()))

    mid=''
    for s in lst:
        mid+=('#.'+ s+ '.')
    mid+='#'

    row1=''
    for i in range(len(mid)):
        if mid[i]=='.':
            row1+='#'
        else:
            row1+='.'

    row2=''
    for i in range(len(mid)):
        if mid[i]=='#' or row1[i]=='#':
            row2+='.'
        else:
            row2+='#'
            

    ans=[]
    ans.append(row2)
    ans.append(row1)
    ans.append(mid)
    ans.append(row1)
    ans.append(row2)

    for row in ans:
        print(row)

'''


# 4698. 테네스의 특별한 소수 [풀이중]

'''
N=10**6
arr=[True for i in range(N+1)]


# 에라토스테네스의 채
for i in range(2,N+1):
    if arr[i]!=False:
        continue
    for j in range(i*2,N+1,i):
        arr[j]=False      
print(arr)


t = int(input())
for tc in range(1, t + 1) :
    k,a,b=map(int,input().split())
    cnt=0

    for i in range(a,b+1):
        if arr[i]:
            if str(k) in str(i):
                cnt+=1


    print(f'#{tc}',cnt)


'''

# 4371 항구에 들어오는 배

'''
n=int(input())
arr=[]

for _ in range(n):
    arr.append(int(input()))

day=1

# 최소 공배수
while True:
    for x in arr:
        if day%x!=0:
            day+=1
            break
    else:
        break


ship=[0]*(day+1)
for i in range(day+1):
    for j in arr:
        if i%j==0:
            ship[i]+=1
print(ship)
print(min(ship[1:]))
            
'''

# 3750 Digit sum

'''
res=[]

def fn(n):
    lst=map(int,str(n))
    s_num=0
    for x in lst:
        s_num+=x
        
    if len(str(s_num))==1:
        return s_num
    else:
        return fn(s_num)


T=int(input())
for tc in range(1,T+1):
    n=int(input())
    res.append(fn(n))



for tc in range(1,T+1):
    print(f'#{tc}',res[tc-1])


'''


# 3408 세가지 합 구하기 [풀이중]

'''
# 등차수열 공식
N:((N+1)*N))//2
홀수:N**B
짝수:(N*1)*N

'''
'''
T=int(input())
for tc in range(1,T+1):
    N=int(input())
    arr=[x for x in range(1,N*2+1)]

    s1=0
    s2=0
    s3=0

    s1=sum(arr[:N])
    s2=sum(x for x in arr if x%2!=0)
    s3=sum(x for x in arr if x%2==0)

    print(f'#{tc}',s1,s2,s3)

'''

# 3131 100만 이하의 모든 소수
# 에라토스체

'''
N=10**6
arr=[i for i in range(N+1)]

for i in range(2,int(N**(1/2))+1):
    if arr[i]!=0:
        for j in range(i*2,N+1,i):
            arr[j]=0

ans=list(x for x in arr[2:] if x!=0)

print(*ans)

'''
        
    
    
# 2948 문자열 교집합

'''
T=int(input())
for tc in range(1,T+1):
    
    n,m=map(int,input().split())
    lst1=set(input().split())
    lst2=set(input().split())
    cnt=0

    lst=lst1.intersection(lst2)
    

    print(f'#{tc}',len(lst))

'''


# 2817 부분 수열의 합

# 내가 구현한 backtracking -> 실패
'''
T=int(input())
for tc in range(1,T+1):
    cnt=0
    n,k=map(int,input().split())
    arr=list(map(int,input().split()))
    v=[True]*len(arr)

    def dfs(num):
        global cnt

        if num==k:
            cnt+=1
            return
        if num>k:
            return
        else:      
            for i in range(len(arr)-1):
                for j in range(i+1,len(arr)):
                    if v[i]==True and v[j]==True:
                        num+=(arr[i]+arr[j])
                        v[i]=False
                        v[j]=False
                        dfs(num)
                        v[i]=True
                        v[j]=True


    dfs(0)
    print(f'#{tc}',cnt)

'''


# Youtube 풀이

'''
def dfs(n,sm):
    global cnt

    if sm>K:    # 원소가 자연수들이기 때문에 줄어들 수 없음, 가지치기
        return
    
    if n==N:
        if sm==K:
            cnt+=1
            return
    else:
        dfs(n+1,sm+lst[n])  # 포함 시킬 때
        dfs(n+1,sm) # 포함시키지 않을 때


T=int(input())
for tc in range(1,T+1):
    cnt=0
    N,K=map(int,input().split())
    lst=list(map(int,input().split()))


    dfs(0,0)
    print(f'#{tc}',cnt)

'''


# 2814 최장 경로

'''
dfs 풀이 - 끝까지 도달했을 때 최장길이 갱신, 모든 노드에 대해서 길이 확인

'''
'''

def dfs(i,cnt):
    global ans
    v[i]=True
    for n in graph[i]:
        if v[n]!=True:
            dfs(n,cnt+1)
            
    # 더 이상 진행할 노드가 없으면 방문 해제, ans값 갱신
    v[i]=False
    
    if cnt>ans:
        ans=cnt
    

T=int(input())
for tc in range(1,T+1):
   
    n,m=map(int,input().split())
    graph=[[] for _ in range(n+1)]
    ans=0
    
    for _ in range(m):
        a,b=map(int,input().split())
        graph[a].append(b)
        graph[b].append(a)
        
    v=[False]*(n+1)

    for i in range(1,n+1):
        dfs(i,1)

    print(f'#{tc}',ans)

'''



# 1491 원재의 벽 꾸미기
'''
T=int(input())
for tc in range(1,T+1):
    n,a,b=map(int,input().split())
    ans=1e9

    for r in range(1,n+1):
        c=1
        while r*c<=n:
            val=a*abs(r-c)+b*(n-r*c)
            ans=min(ans,val)
            c+=1

    print(f'#{tc}',ans)

'''

# 5642 [Professional] 합

# 백트랙킹 풀이 도전 -> 실패
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=list(map(int,input().split()))
    ans=0

    def dfs(n,sm):
        global ans
        
        if n==len(lst):
            return
        else:
            dfs(n+1,sm+lst[n])
            dfs(n+1,0)

        ans=max(ans,sm)

        
    dfs(0,0)
    print(f'#{tc}',ans)
'''

# 이중 포문 -> 시간초과

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=list(map(int,input().split()))
    ans=0

    for i in range(n):
        sm=0
        for j in range(i,n):
            sm+=lst[j]
            if ans<sm:
                ans=sm

    print(f'#{tc}',ans)


'''

# 정답 코드

# 음수가 되면 버리고 누적합 0초기화 -> for문 한개로 해결
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=list(map(int,input().split()))
    ans=-1e9
    sm=0
    for i in range(n):
        sm+=lst[i]
        if sm>ans:
            ans=sm
        if sm<0:
            sm=0

    print(f'#{tc}',ans)

'''

# 1209 [S/W 문제해결 기본] 2일차 - Sum [풀이중] - Pass 

'''
for _ in range(1,11):
    tc=int(input())
    arr=[list(map(int,input().split())) for _ in range(100)]
    ans=0
        
    for i in range(100):
        col_sum=0
        cr1_sum=0
        cr2_sum=0
        for j in range(100):
            if i==j:
                cr1_sum+=arr[i][j]
            if 99-j==i:
                cr2_sum+=arr[i][j]
                
            col_sum+=[j][i]
        ans=max(ans,sum(arr[i]),col_sum)

    ans=max(cr1_sum,cr2_sum,ans)
    
    print(f'#{tc}',ans)

'''


#9229 한빈이와 Spot Mart

'''
T=int(input())
for tc in range(1,T+1):
    n,m=map(int,input().split())
    arr=list(map(int,input().split()))
    ans=-1
    v=[]

    for i in range(len(arr)-1):
        for j in range(i+1,len(arr)):
            sm=arr[i]+arr[j]
            if sm not in v:  
                if sm<=m:
                    ans=max(ans,sm)
                    v.append(sm)

    print(f'#{tc}',ans)
                    
'''     


# 1215. [S/W 문제해결 기본] 3일차 - 회문1 [풀이중]

'''
T=int(input())
for tc in range(1,T+1):
    arr=[]
    for _ in range(8):
        arr.append(list(map(str,input())))

    cnt=0

    n=int(input())
    for i in range(len(arr)):
        for j in range(len(arr)-n+1):
            s1=arr[i][j:j+n]
            if s1==s1[::-1]:
                cnt+=1


            
            s2=''
            for k in range(n):
                s2+=arr[j+k][i]

            if s2==s2[::-1]:
                cnt+=1


    print(f'#{tc}',cnt)

'''


# 3752 가능한 시험 점수

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=list(map(int,input().split()))
    v=[]

    def dfs(n,sm):
        if n==len(lst):
            return
        if sm+lst[n] not in v:
            v.append(sm+lst[n])
            dfs(n+1,sm+lst[n])  # 포함 o
            dfs(n+1,sm) # 포함 x

    dfs(0,0)
    print(len(v)+1))

'''

# 14413 격자판 칠하기 (1차 실패 코드 - 이유: #??# 고려안함)
'''
T=int(input())
for tc in range(1,T+1):
    N,M=map(int,input().split())

    arr=[]

    for _ in range(N):
        arr.append(list(map(str,input())))


    dx=[-1,1,0,0]
    dy=[0,0,-1,1]
      
    ans='possible'
    flag=False
    for i in range(N):
        for j in range(M):
            if arr[i][j]=='#' or arr[i][j]=='.':
                w=arr[i][j]
                
                for k in range(4):
                    nx=i+dx[k]
                    ny=j+dy[k]
                    
                    if 0<=nx<N and 0<=ny<M:
                        if arr[nx][ny]==w:
                            ans='impossible'
                            flag=True
                            break
            if flag:
                break
        if flag:
            break
                        

    print(f'#{tc}',ans)  
                
'''              


# (1차 실패 코드 - 이유: 코드 오류)

'''
ans='P'
def dfs(x,y,num,w):
    global ans
    
    dx=[-1,1,0,0]
    dy=[0,0,-1,1]

    for k in range(4):
        nx=x+dx[k]
        ny=y+dy[k]

        if 0<=nx<N and 0<=ny<M:
            if num%2!=0:
                if arr[nx][ny]==w:
                    ans='I'
                    return
                else:
                    if w=='#':
                        w='.'
                    else:
                        w='#'
                    return dfs(nx,ny,num+1,w)
            else:
                if arr[nx][ny]!=w:
                    ans='I'
                    return
                else:
                    if w=='#':
                        w='.'
                    else:
                        w='#'
                    return dfs(nx,ny,num+1,w)
                

N,M=map(int,input().split())

arr=[]

for _ in range(N):
    arr.append(list(map(str,input())))


for i in range(N):
    for j in range(M):
        if arr[i][j]=='#' or arr[i][j]=='.':
            w=arr[i][j]
            dfs(i,j,1,w)


print(ans)

'''


# 3차 풀이 참고 (성공)
'''
def func1(arr):    # i+j가 홀수:#, 짝수:. 확인
    for i in range(N):
        for j in range(M):
            if (i+j)%2==0:
                if arr[i][j]=='#':
                    return False
            else:
                if arr[i][j]=='.':
                    return False
    return True




def func2(arr):    # i+j가 짝수:#, 홀수:. 확인
    for i in range(N):
        for j in range(M):
            if (i+j)%2==0:
                if arr[i][j]=='.':
                    return False
            else:
                if arr[i][j]=='#':
                    return False
    return True          



T=int(input())
for tc in range(1,T+1):
    N,M=map(int,input().split())

    arr=[]
    for _ in range(N):
        arr.append(list(map(str,input())))
        
    if func1(arr) or func2(arr):
        ans='possible'
    else:
        ans='impossible'


    print(f'#{tc}',ans)  

'''

# 13428. 숫자 조작

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    max_num=min_num=n

    lst=list(map(int,str(n)))

    for i in range(len(lst)-1):
        for j in range(i+1,len(lst)):
            lst[i],lst[j]=lst[j],lst[i]
            num=int(''.join(map(str,lst)))
            max_num=max(max_num,num)
            if len(str(n))==len(str(num)):
                min_num=min(min_num,num) 
            lst[i],lst[j]=lst[j],lst[i]

    print(f'#{tc}',min_num,max_num)

'''

# 9480 민정이와 광직이의 알파벳 공부 
'''
def dfs(n,lst):
    global cnt
    global alpha
    
    if n==N:
        return

    dfs(n+1,lst+arr[n])
    dfs(n+1,lst)

    lst+=arr[n]
    
    small_lst=[]
    for i in range(len(lst)):
        if lst[i] in alpha:
            small_lst.append(lst[i])
            
    if len(set(small_lst))==26:
        cnt+=1

        
T=int(input())
for tc in range(1,T+1):

    alpha='abcdefghijklmnopqrstuvwxyz'

    N=int(input())
    arr=[]

    for _ in range(N):
        arr.append(input())

    cnt=0        
    dfs(0,'')
    print(f'#{tc}',cnt)

'''



# 14555 공과 잡초

'''
T=int(input())
for tc in range(1,T+1):
    garden=input()
    ans=garden.count('(')+garden.count(')')-garden.count('()')


    print(f'#{tc}',ans)

'''


# 7675 통역사 성경이 [풀이중 - 런타임 에러]

'''
T=int(input())
for tc in range(1,T+1):
    up_lst=[]
    for i in range(ord('A'),ord('Z')+1):
        up_lst.append(chr(i))

    low_lst=[]
    for i in range(ord('a'),ord('z')+1):
        low_lst.append(chr(i))

    n=int(input())
    string=input()

    string=string.replace('?','.')
    string=string.replace('!','.')

    lst=string.split('.')
    lst.remove('')
    #print(lst)

    ans=[]
    for s in lst:
        sub_lst=s.split()
        #print(sub_lst)
        
        cnt=0  # 이름 개수
        
        for x in sub_lst:
            so_cnt=0  # 소문자 개수
            check=False
            #print(x)
            for i in range(len(x)):
                if i==0:
                    if x[i] in up_lst:
                        #print(x[i])
                        check=True
                else:
                    if x[i] in low_lst:
                        so_cnt+=1
                        #print(so_cnt)
            if check:
                if so_cnt==len(x)-1:
                    cnt+=1
                    
        ans.append(cnt)

    print(f'#{tc}',*ans)
                
   
'''      
        
        


# 9280 진용이네 주차타워 [실패 - 테스트케이스 예제만 통과]

'''
T=int(input())
for tc in range(1,T+1):
    
    n,m=map(int,input().split())

    R=[]    # 주차장 요금
    W=[]    # 차량 무게
    L=[]    # 출차 목록

    for _ in range(n):
        R.append(int(input()))
    for _ in range(m):
        W.append(int(input()))
    for _ in range(2*m):
        L.append(int(input()))


    P=[0]*(len(R))  # 주차장
    D=[]    # 대기 리스트
    L.reverse()

    ans=0

    while L:
        car=L.pop()

        while D:
            flag=False
            for i in range(len(P)):
                if P[i]==0:
                    P[i]=D.pop(0)
                    flag=True
            if flag==False:
                break
            
                    
        
        if car<0:   # 나가는 자동차
            c=W[abs(car)-1]
            idx=P.index(c) # 나가는 자동차 자리
            P[idx]=0   # 자리 비우기
            ans+=R[idx]*c    # 요금 정산


        else:   # 들어오는 자동차
            c=W[car-1]
            for i in range(len(R)):
                if P[i]==0:
                    P[i]=c
                    break
            else:
                D.append(c)


            
    print(f'#{tc}',ans)   

'''



# 8016 홀수 피라미드

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    num=0
    for i in range(1,n+1):
        num+=(2*i-1)
        if i==n:
            st=num-(2*i-1)
            end=num-1

    arr=[1]
    for i in range(num-1):
        arr.append(arr[-1]+2)

    print(f'#{tc}',arr[st],arr[end])
    
'''


# 2819 격자판의 숫자 이어 붙이기 (내코드)

'''
def dfs(i,j,n,s):
    global cnt
    global v
    
    if n==7:
        if s not in v:
            cnt+=1
            v.append(s)
        return
    
    else:
        s+=graph[i][j]
        for k in range(4):
            ni=i+dx[k]
            nj=j+dy[k]
            if 0<=ni<4 and 0<=nj<4:
                dfs(ni,nj,n+1,s)


T=int(input())
for tc in range(1,T+1):
    graph=[list(input().split()) for _ in range(4)]


    cnt=0
    v=[]  # 방문숫자

    dx=[-1,1,0,0]
    dy=[0,0,-1,1]

    for i in range(4):
        for j in range(4):
            dfs(i,j,0,'')       # dfs 수행


    print(f'#{tc}',cnt)

'''

# 시간복잡도 줄인 코드 -> 모든 값 다 담고 set으로 중복 제거
'''
def dfs(i,j,n,s):
    global cnt
    
    if n==7:
        v.append(s)
        return
    
    else:
        s+=graph[i][j]
        for k in range(4):
            ni=i+dx[k]
            nj=j+dy[k]
            if 0<=ni<4 and 0<=nj<4:
                dfs(ni,nj,n+1,s)


T=int(input())
for tc in range(1,T+1):
    graph=[list(input().split()) for _ in range(4)]


    cnt=0
    v=[]  # 방문숫자

    dx=[-1,1,0,0]
    dy=[0,0,-1,1]

    for i in range(4):
        for j in range(4):
            dfs(i,j,0,'')       # dfs 수행


    print(f'#{tc}',len(set(v)))

'''


# 4579 세상의 모든 팰린드롬 2

'''
T=int(input())
for tc in range(1,T+1):
    s=input()
    arr=list(map(str,s))
    ans='Exist'

 
    for i in range(len(s)//2):
        if arr[i]=='*' or arr[len(s)-i-1]=='*':
            break
        else:
            if arr[i]!=arr[len(s)-i-1]:
                ans='Not exist'
                break

                
    print(f'#{tc}',ans)
            
'''


# 3752 가능한 시험 점수 [풀이중]
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    scr=list(map(int,input().split()))
    temp=[0]
    res=[0]
    
    
    for i in range(len(scr)):
        for j in range(len(temp)):
            temp.append(temp[j]+scr[i])
      
        res=list(set(temp))
                

    print(f'#{tc}',len(res))
    
'''




# 1238 [S/W 문제해결 기본] 10일차 - Contact [DFS풀이: 7/10개 통과]

'''
def dfs(n,num,v): # n:노드번호, num:몇 번째 연락인지 저장
    global ans   
    v[n]=True   # 방문처리
    for x in graph[n]:
        if v[x]!=True:
            ans.append((x,num)) 
            dfs(x,num+1,v)


T=10
for tc in range(1,T+1):

    # 문제 입력
    graph=[[]*101 for _ in range(101)]
    v=[False]*101

    L,n=map(int,input().split())
    lst=list(map(int,input().split()))

    idx=0
    for _ in range(len(lst)//2):
        a=lst[idx]
        b=lst[idx+1]
        idx+=2
        graph[a].append(b)


    # DFS 수행
    ans=[]  # 방문 모드 저장, 전역
    dfs(n,0,v)     

    ans.sort(key=lambda x:(-x[1],-x[0]))    # 최대 num값 중, 가장 큰 n값 출력
    print(f'#{tc}',ans[0][0])  


'''

# BFS 풀이 (통과)

# 특정 노드에 연락할 수 있는 방법이 여러개 있다면 최소한의 방법으로 도착하는 횟수를 저장해야하기 때문에 BFS로 풀어야함

'''
from collections import deque


def bfs(start):
    v[start]=1
    q=deque()
    q.append(start)

    while q:
        n=q.popleft()
        for x in graph[n]:
            if v[x]==0:
                v[x]=v[n]+1 # 연락 순서 +1
                q.append(x)

T=10
for tc in range(1,T+1):
    L,s=map(int,input().split())
    lst=list(map(int,input().split()))

    graph=[[]*101 for _ in range(101)]
    for i in range(0,len(lst),2):
        graph[lst[i]].append(lst[i+1])
        
    v=[0]*101   # 방문표시 + 연락횟수

    bfs(s)
    print(v)
    max_call=0  # 최대 연락 횟수
    for i in range(1,len(v)):
        if v[i]>=max_call:
            max_call=v[i]
            ans=i
        
            
    print(f'#{tc}',ans)

'''



# 1859 백만 장자 프로젝트 [재귀 -> 시간초과]

'''
그 날에 할 수 있는 일
1. 사기
2. 안사기
3. 팔기

'''

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    arr=list(map(int,input().split()))

    ans=0
        

    def dfs(n,cnt,mon):
        global ans
        
        if ans<mon:
            ans=mon
            
        if n==len(arr):
            return
        
        else:
            dfs(n+1,cnt+1,mon-arr[n])  # 사기
            dfs(n+1,cnt,mon) # 안사기
            dfs(n+1,0,mon+arr[n]*cnt)  # 팔기
            


    dfs(0,0,0)

    print(f'#{tc}',ans)

'''



# 1859 백만 장자 프로젝트
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    lst=list(map(int,input().split()))

    mon=0

    for i in range(len(lst)-1,-1,-1):
        if i==len(lst)-1:
            pri=lst[-1]
            cnt=0
        else:
            if lst[i]>pri:
                pri=lst[i]

            else:
                mon+=(pri-lst[i])

            
    print(f'#{tc}',mon)

'''



# 1226 [S/W 문제해결 기본] 7일차 - 미로1 [8/10 통과]

'''

def dfs(i,j):
    global ans

    arr[i][j]=1    # 통로라면 방문표시
    
    dx,dy=[-1,1,0,0],[0,0,-1,1]
    
    for k in range(4):
        ni=i+dx[k]
        nj=j+dy[k]

        if 0<=ni<16 and 0<=nj<16:
            if arr[ni][nj]==3:
                ans=1
                return
            
            if arr[ni][nj]==0:  # 벽이 아니면 이동
                dfs(ni,nj)

                    

T=10
for tc in range(1,T+1):
    ans=2
    t=int(input())
    arr=[list(map(int,input())) for _ in range(16)]

    for i in range(16):
        for j in range(16):
            if arr[i][j]==2:
                dfs(i,j)

    print(f'#{tc}',ans)


'''




# 20551 증가하는 사탕 수열

'''
T=int(input())
for tc in range(1,T+1):
    a,b,c=map(int,input().split())

    ans=0

    if a<1 or b<2 or c<3:
        ans=-1
    else:
        if b>=c:
            ans+=b-(c-1)
            b=c-1
        if a>=b:
            ans+=a-(b-1)

    print(f'#{tc}',ans)
        
'''



# 18662 등차수열 만들기

'''
T=int(input())
for tc in range(1,T+1):
    a,b,c=map(int,input().split())
    ans=min(abs(a-2*b+c),abs(c-2*b+a),abs(b-((a+c)/2)))
    print(f'#{tc} {ans:0.1f}')

'''



# 13732 정사각형 판정

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())

    arr=[list(map(str,input())) for _ in range(n)]


    row_cnt=0
    flag=False

    for i in range(n):
        for j in range(n):
            if arr[i][j]=='#':
                flag=True
                st_r=i
                st_c=j

                for c in range(j,n):
                    if arr[i][c]=='#':
                        row_cnt+=1
                    else:
                        break
                if flag:
                    break
                    
        if flag:
            break


    sq=0
    total_sq=0
    for i in range(st_r,st_r+row_cnt):
        for j in range(st_c,st_c+row_cnt):
            if 0<=i<n and 0<=j<n:
                if arr[i][j]=='#':
                    sq+=1

    for i in range(n):
        total_sq+=arr[i].count('#')


    ans='no'
    if sq==row_cnt*row_cnt==total_sq:
        ans='yes'

    print(f'#{tc}',ans)

'''



# 1219 [S/W 문제해결 기본] 4일차 - 길찾기
'''
T=10
for tc in range(1,T+1):
    t,n=map(int,input().split())
    arr=list(map(int,input().split()))

    graph=[[]*100 for _ in range(101)]
    v=[False]*100
    for i in range(0,len(arr),2):
        a=arr[i]
        b=arr[i+1]
        graph[a].append(b)


    ans=0
    # A:0, B:99

    def dfs(n):
        global ans
        v[n]=True

        for x in graph[n]:
            if x==99:
                ans=1
                return  
            if v[x]!=True:
                dfs(x)


    dfs(0)
    print(f'#{tc}',ans)
        


'''


# 1861 정사각형 방
# 시간초과
'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    arr=[list(map(int,input().split())) for _ in range(n)]

    ans=0
    max_cnt=0

    def dfs(i,j,num):
        global cnt
        global ans
        global max_cnt
        
        dx=[-1,1,0,0]
        dy=[0,0,-1,1]
        for k in range(4):
            ni=i+dx[k]
            nj=j+dy[k]
            if 0<=ni<n and 0<=nj<n:
                if arr[ni][nj]==arr[i][j]+1:
                    if arr[ni][nj]==ans:    # 이전까지 최댓값인 ans가 나온다면 지금까지 cnt에 ans를 더하고 반환, 시간 초과 해결 x
                        cnt+=max_cnt
                        return
                    else:
                        dfs(ni,nj,arr[i][j]+1)
                        cnt+=1



    for num in range(1,n**2+1):
        cnt=1
        for i in range(n):
            for j in range(n):
                if arr[i][j]==num:
                    dfs(i,j,num)
                    if cnt>max_cnt:
                        ans=num
                        max_cnt=cnt


    print(f'#{tc}',ans,max_cnt)

'''
# num값에 대해 순차적으로 for문을 걸어줄 필요가 없음

'''
T=int(input())
for tc in range(1,T+1):
    n=int(input())
    arr=[list(map(int,input().split())) for _ in range(n)]

    ans=0
    max_cnt=0

    def dfs(i,j,num):
        global cnt
        
        dx,dy=[-1,1,0,0],[0,0,-1,1]
        for k in range(4):
            ni=i+dx[k]
            nj=j+dy[k]
            if 0<=ni<n and 0<=nj<n:
                if arr[ni][nj]==arr[i][j]+1:
                    cnt+=1
                    dfs(ni,nj,arr[i][j]+1)
                    



    for i in range(n):
        for j in range(n):
            cnt=1
            dfs(i,j,arr[i][j])
            if cnt>max_cnt:
                ans=arr[i][j]ㄴ
                max_cnt=cnt
            elif cnt==max_cnt:  # 개수가 같은 경우 낮은 번호 저장
                ans=min(ans,arr[i][j])

    
    print(f'#{tc}',ans,max_cnt)

'''



# 2806 N-Queen
'''
def dfs(row):
    global cnt
    
    if row==N:
        cnt+=1
        return

    for col in range(N):
        if check1[col]==1 or check2[col-row+N-1]==1 or check3[row+col]==1:
            continue

        check1[col]=1
        check2[col-row+N-1]=1
        check3[row+col]=1
        dfs(row+1)
        check1[col]=0
        check2[col-row+N-1]=0
        check3[row+col]=0


T=int(input())
for tc in range(1,T+1):      
    N=int(input())
    cnt=0

    check1=[0]*(2*N-1) # 세로 검사
    check2=[0]*(2*N-1) # 왼쪽 대각선 검사
    check3=[0]*(2*N-1) # 오른쪽 대각선 검사


    # row  == n의 개수, 각 행에 하나밖에 퀸을 못 설치하므로 row가 1추가 될 때마다 퀸을 하나 더 설치할 수 있다는 것을 의미
    dfs(0)

    print(f'#{tc}',cnt)
            
'''

# 5643 [Professional] 키 순서
# [내 생각 with 코드 -> 실패]
'''
1. 자신이 가르키는 노드들에 대해서 DFS 수행, 방문 표시
2. 방문 표시 안된 모든 노드들에 대해서 DFS 수행, 모든 노드에서 자신이 나온다면 ANS(전역변수)+1
'''

'''
def dfs1(n):
    v[n] = 1

    for x in graph[n]:
        if v[x] != 1:
            dfs1(x)

#1 -> 2 -> i 인경우 2부터 방문한 경우, 1에 대해서 처리가 안됨 v복구(sub_v 사용)
def dfs2(n, goal):
    global lw_cnt
    sub_v[n] = 1

    for x in graph[n]:
        if x == goal:
            lw_cnt += 1
            return

        if sub_v[x] != 1:
            dfs2(x, goal)

# main문
T=int(input())
for tc in range(1,T+1):
    n = int(input())
    m=int(input())
    graph = [[] for _ in range(n + 1)]
    ans = 0

    for _ in range(m):
        a, b = map(int, input().split())
        graph[a].append(b)


    for i in range(1,n+1):
        v=[0]*(n+1)
        left_n=[]   # i 노드에 위쪽에 걸리지 않는 노드
        lw_cnt=0 # 걸리지 않고 i에 도달하는 노드
        dfs1(i)
        up_cnt=v.count(1)
        for k in range(1,n+1):
            if v[k]!=1:
                left_n.append(k)

            for j in left_n:
                sub_v=v
                dfs2(j,i)

        if up_cnt+lw_cnt==(n-1):
            ans+=1

    print(f'#{tc}',ans)

'''

# 풀이방법 [답지]

'''
in, out 2개의 graph 형성 -> 길든 짧든 모든 간선이 연결 되어 있다면 순서를 파악할 수 있음
(양방향 간선으로 graph를 초기화하는 경우에는 모든 노드가 전부 방문되기 때문에 x)
in, out 2개의 방문 리스트를 만들고 나갈때 dfs, 들어올 때 수행 후 방문 노드 카운트, in+out == n-1 이라면 ans+1
'''

'''
def dfs(graph, start):
    v = [0] * (n + 1)
    v[start] = 1
    stack = [start]

    while stack:
        now = stack.pop()
        for x in graph[now]:
            if not v[x] != 1:
                v[x] = 1
                stack.append(x)

    return v.count(1)


n = int(input())
m = int(input())
ans=0
out_graph = [[] for _ in range(n + 1)]
in_graph = [[] for _ in range(n + 1)]


for _ in range(m):
    a, b = map(int, input().split())
    out_graph[a].append(b)
    in_graph[b].append(a)

for i in range(1,n+1):
    if dfs(in_graph,i)+dfs(out_graph,i)==n-1:
        ans+=1

print(ans)

'''

'''
def dfs(graph, start):
    v = [0] * (N + 1)
    # v[start] = 1  # 시작점을 체크하면 중복 카운트됨, 사이클은 형성안된 형태기에 가능
    stack = [start]

    while stack:
        n = stack.pop()
        for x in graph[n]:
            if v[x] != 1:
                v[x] = 1
                stack.append(x)

    return v.count(1)

T=int(input())
for tc in range(1,T+1):
    N = int(input())
    m = int(input())
    ans=0
    out_graph = [[] for _ in range(N + 1)]
    in_graph = [[] for _ in range(N + 1)]



    for _ in range(m):
        a, b = map(int, input().split())
        out_graph[a].append(b)
        in_graph[b].append(a)
    
    for i in range(1,N+1):
        if dfs(in_graph,i)+dfs(out_graph,i)==N-1:
            ans+=1

    print(f'#{tc}',ans)

'''

# 13428 숫자 조작 (복습)

'''
T=int(input())
for tc in range(1,T+1):
    arr=list(map(int,input()))
    L=len(arr)
    max_val=int(''.join(map(str,arr)))
    min_val=int(''.join(map(str,arr)))

    for i in range(len(arr)-1):
        for j in range(i+1,len(arr)):
            arr[i],arr[j]=arr[j],arr[i]
            num=int(''.join(map(str,arr)))
            if num>max_val:
                max_val=num
            if num<min_val and len(str(num))==L:
                min_val=num

            arr[i],arr[j]=arr[j],arr[i]

    print(f'#{tc}',min_val,max_val)

'''

# 5642 [Professional] 합 (복습)

# 재귀 구현 실패 코드
'''
def func(n,val):
    global max_val
    if n==N:
        return

    if val+arr[n]<0:
        return

    if val+arr[n]>max_val:
        max_val=val+arr[n]

    func(n+1,val+arr[n])

T=int(input())
for tc in range(1,T+1):
    N=int(input())
    arr=list(map(int,input().split()))
    max_val=0

    for i in range(len(arr)):
        func(i,0)

    print(f'#{tc}',max_val)

'''

# for문 한 줄로 해결할 수 있음 ..
'''
T=int(input())
for tc in range(1,T+1):
    N = int(input())
    arr = list(map(int, input().split()))
    max_val=-1e9
    sm = 0
    for i in range(len(arr)):
        sm+=arr[i]
        if sm>max_val:
            max_val=sm
        if sm<0:
            sm=0   # 음수일 경우 sum값 초기화 다음 수부터 다시 시작


    print(f'#{tc}',max_val)

'''
# 2817 부분 수열의 합
'''
T=int(input())
for tc in range(1,T+1):
    def dfs(n, sm):
        global ans

        if n == N:
            return
        if sm + arr[n] == K:
            ans += 1

        dfs(n+1, sm + arr[n])
        dfs(n+1, sm)

    N,K=map(int,input().split())
    arr=list(map(int,input().split()))
    ans=0
    dfs(0,0)

    print(f'#{tc}',ans)
'''

# 3307 최장 증가 부분 수열
'''
T=int(input())
for tc in range(1,T+1):
    N=int(input())
    arr=list(map(int,input().split()))
    dp=[1]*N
    
    for i in range(1,N):
        for j in range(i):
            if arr[i]>=arr[j]:
                dp[i]=max(dp[i],dp[j]+1)
    
    print(f'#{tc}',max(dp))
'''

# 3304 최장 공통 부분 수열 + 문자 출력하기 (돌아가지는 않는데 원리는 맞음 ,,) (복습)
'''
def lcs(x,y):
    x,y=' '+x,' '+y     # 마진 설정
    n, m = len(x), len(y)  # n:row, m:col
    lcs=[[0 for _ in range(m)] for _ in range(n)]
    check=[[0 for _ in range(m)] for _ in range(n)]

    for i in range(1,n):
        for j in range(1,m):
            if x[i]==y[j]:
                lcs[i][j]=lcs[i-1][j-1]+1
                check[i][j]=1
            else:
                lcs[i][j]=max(lcs[i][j-1],lcs[i-1][j])
                check[i][j]=3 if(lcs[i][j-1]>lcs[i-1][j]) else 2

    return lcs,check


def get_lcs(i,j,b,x):
    if i==0 or j==0:
        return ''
    else:
        if b[i][j]==1:
            return get_lcs(i-1,j-1,b,x)+x[i]
        elif b[i][j]==2:
            return get_lcs(i,j-1,b,x)
        else:
            return get_lcs(i-1,j,b,x)


str1,str2=input().split()
lcs,check=lcs(str1,str2)
print(lcs[-1][-1])

ans=get_lcs(len(str1),len(str2),check,str1)
print(ans)

'''

# 2814 최장 경로

'''
def dfs(n, depth):
    global ans
    v[n] = True

    for x in graph[n]:
        if v[x] != True:
            dfs(x, depth+1)

    v[n]=False
    if depth > ans:
        ans = depth


T=int(input())
for tc in range(1,T+1):
    N,M=map(int,input().split())
    graph=[[] for _ in range(N+1)]
    for _ in range(M):
        v1,v2=map(int,input().split())
        graph[v1].append(v2)
        graph[v2].append(v1)

    ans=0
    for i in range(1,N+1):
        v=[False]*(N+1)
        dfs(i,1)

    print(f'#{tc}',ans)

'''

# 1486 장훈이의 높은 선반
'''
def dfs(n,sm):
    if sm>=B:
        res.append(sm)
        return
    if n==N:
        return

    dfs(n+1,sm+arr[n])
    dfs(n+1,sm)

T=int(input())
for tc in range(1,T+1):
    N,B=map(int,input().split())
    arr=list(map(int,input().split()))
    res=[]

    dfs(0,0)
    print(f'#{tc}',min(res)-B)
'''

# 7465 창용 마을 무리의 개수
'''
def dfs(n):
    v[n]=True
    for x in graph[n]:
        if v[x]!=True:
            dfs(x)

T=int(input())
for tc in range(1,T+1):
    N,M=map(int,input().split())
    graph=[[] for _ in range(N+1)]
    v=[False]*(N+1)
    cnt=0
    for _ in range(M):
        v1,v2=map(int,input().split())
        graph[v1].append(v2)
        graph[v2].append(v1)

    for i in range(1,N+1):
        if v[i]!=True:
            cnt+=1
            dfs(i)

    print(f'#{tc}',cnt)

'''

# 5215 햄버거 다이어트

'''
T=int(input())
for tc in range(1,T+1):
    N,W=map(int,input().split())
    items=[]
    dp=[0]*(W+1)
    for _ in range(N):
        v,w=map(int,input().split())
        items.append((v,w))

    for item in items:
        v,w=item
        for i in range(W,w-1,-1):
            dp[i]=max(dp[i],dp[i-w]+v)

    print(f'#{tc}',dp[-1])

'''


# 20934 방울 마술
'''
time=int(input())
for tc in range(1,time+1):
     S,T=input().split()
     T=int(T)


     if S=='..o':
          if T==0:
               ans=2
          else:
               if T%2!=0:
                    ans=1
               else:
                    ans=0

     elif S=='.o.':
          if T%2!=0:
               ans=0
          else:
               ans=1

     else:
          if T%2!=0:
               ans=1
          else:
               ans=0

     print(f'#{tc}',ans)

'''

# 1959. 두 개의 숫자열 (복습)
'''
T=int(input())
for tc in range(1,T+1):
     N,M=map(int,input().split())
     A=list(map(int,input().split()))
     B=list(map(int,input().split()))
     ans=0

     if len(A)>len(B):
          A,B=B,A

     for i in range(len(B)-len(A)+1):
          sm=0
          k=0
          for j in range(i,i+len(A)):
               sm+=A[k]*B[j]
               k+=1
          if sm>ans:
               ans=sm

     print(f'#{tc}',ans)

'''

# 1221. [S/W 문제해결 기본] 5일차 - GNS
'''
lst=["ZRO", "ONE", "TWO", "THR", "FOR", "FIV", "SIX", "SVN", "EGT", "NIN"]
dic={val:idx for idx,val in enumerate(lst)}

T=int(input())
for tc in range(1,T+1):
     st,n=input().split()
     arr=list(input().split())
     arr.sort(key=lambda x:dic[x])
     print(st)
     print(arr)

'''


# 1244 [S/W 문제해결 응용] 2일차 - 최대 상금 (복습)
'''
def dfs(lst,t): # t: 교체횟수, num: 숫자, i:인덱스
    global ans

    if t==0:
        if ans<int(''.join(map(str,lst))):
            ans=int(''.join(map(str,lst)))
        return

    for i in range(len(lst)-1):
          for j in range(i+1,len(lst)):
              lst[i],lst[j]=lst[j],lst[i]
              num=int(''.join(map(str,lst)))
              if (num,t) not in check:
                  dfs(lst,t-1)
                  check.append((num,t))
              lst[i], lst[j] = lst[j], lst[i]

T=int(input())
for tc in range(1,T+1):
    num,t=map(int,input().split())
    lst=list(map(int,str(num)))
    ans=0
    check=[]  # 방문 배열
    dfs(lst,t)
    print(f'#{tc}',ans)

'''

# 1234. [S/W 문제해결 기본] 10일차 - 비밀번호 (복습)
'''
T=10
for tc in range(1,T+1):
    n,num=map(int,input().split())
    lst=list(map(int,str(num)))
    stack=[]

    for i in range(n):
        if len(stack)==0:
            stack.append(lst[i])
        else:
            if stack[-1]==lst[i]:
                stack.pop()
            else:
                stack.append(lst[i])

    ans=int(''.join(map(str,stack)))

    print(f'#{tc}',ans)

'''

# 9480. 민정이와 광직이의 알파벳 공부 (복습)
'''
def dfs(n,s):
    global ans

    if n==N:
        arr=list(set(s))
        cnt=0
        for x in arr:
            if ord('a')<=ord(x)<=ord('z'):
                cnt+=1
        if cnt==26:
            ans+=1
        return
    else:
        dfs(n+1,s+lst[n])
        dfs(n+1,s)

T=int(input())
for tc in range(1,T+1):
    N=int(input())
    lst=[]
    ans=0

    for _ in range(N):
        lst.append(input())

    dfs(0,'')
    print(f'#{tc}',ans)
    
'''

# 1979 어디에 단어가 들어갈 수 있을까 (복습) 실패-접근 방법x

'''
N,K=map(int,input().split())
arr=[]
ans=0
for _ in range(N):
    arr.append(list(map(int,input().split())))

for i in range(N):
    c=0
    while True:
        if arr[i][c]==1:
            cnt=0
            flg=False
            for k in range(K+1):
                if c+k<N:
                    if arr[i][c+k]==1:
                        cntt+=1
                        if cnt==K:
                            flg=True
                    if arr[i][c+k]==1 and flg:
                        break
                if cnt==K:
                    ans+=1
        else:
            c+=1
            if c==N:
                break
'''

# 답지 풀이
'''
풀이방법
-> 1이 연속될 때 누적 합을 구해주고
  ** 0이나 배열의 끝에 도달할 때 누적합이 K개 만큼 있는지 확인, 누적합 초기화
'''
'''
T=int(input())
for tc in range(1,T+1):
    N,K=map(int,input().split())
    arr=[]
    ans=0
    for _ in range(N):
        arr.append(list(map(int,input().split())))

    for i in range(N):
        r_cnt=0 # 가로 검사
        c_cnt=0 # 세로 검사
        for j in range(N):
            # 가로 검사
            if arr[i][j]==1:
                r_cnt+=1
            if arr[i][j]==0 or j==N-1:
                if r_cnt==K:
                    ans+=1
                r_cnt=0

            # 세로 검사
            if arr[j][i]==1:
                c_cnt+=1
            if arr[j][i]==0 or j==N-1:
                if c_cnt==K:
                    ans+=1
                c_cnt=0

    print(f'#{tc}',ans)
'''

# 13038 교환학생
'''
T=int(input())
for tc in range(1,T+1):
    k=int(input())
    arr=list(map(int,input().split()))
    ans=1e9

    for i in range(len(arr)):
        if arr[i]==1:
            ix=i
            day=0
            cnt=k
            while True:
                if cnt==0:
                    ans=min(ans,day)
                    break
                if arr[ix]==1:
                    cnt-=1
                day+=1
                ix+=1
                if ix==len(arr):
                    ix=0
    print(f'#{tc}',ans)

'''

# 9480 민정이와 광직이의 알파벳 공부
'''
def check(s):
    global ans
    lst = list(set(s))
    cnt = 0
    for x in lst:
        if ord('a') <= ord(x) <= ord('z'):
            cnt += 1
    if cnt == 26:
        ans += 1

    return


def dfs(n, s):
    if n == N:
        check(s)
        return
    dfs(n + 1, s + arr[n])
    dfs(n + 1, s)


T=int(input())
for tc in range(1,T+1):
    N=int(input())
    arr=[]
    ans = 0
    for _ in range(N):
        arr.append(input())

    dfs(0,'')
    print(f'#{tc}',ans)

'''

# 6057 그래프의 삼각형
'''
N,M=map(int,input().split())
graph=[[] for _ in range(N+1)]
cnt=0

for _ in range(M):
    v1,v2=map(int,input().split())
    graph[v1].append(v2)
    graph[v2].append(v1)

for i in range(1,N+1):
    for j in range(i+1,N+1):
        for k in range(j+1,N+1):
            if i in graph[j] and j in graph[k] and k in graph[i]:
                cnt+=1


print(cnt)

'''

# 1493 수의 새로운 연산
'''
T=int(input())
for tc in range(1,T+1):
    N=10000
    a,b=map(int,input().split())
    cnt=1
    lst=[(0,0)]
    for _ in range(1,N+1):
        s = cnt
        e = 1
        while True:
            if s==0:
                break
            else:
                lst.append((s,e))
                s-=1
                e+=1
        cnt+=1

    idx1=lst[a]
    idx2=lst[b]
    r=idx1[0]+idx2[0]
    c=idx1[1]+idx2[1]
    ans=lst.index((r,c))
    print(f'#{tc}',ans)

'''

# 1949. [모의 SW 역량테스트] 등산로 조성

'''
가장 높은 곳에서 시작

자신 보다 낫다면 진행, DFS방문 처리

'''

# 깍지 않고 최단 경로만 구할때 코드
''''
N,K=map(int,input().split())
arr=[]
ans=0 # 경로 길이
max_num=0 # 정상 높이
for i in range(N):
    arr.append(list(map(int,input().split())))
    max_num=max(max_num,max(arr[i]))

st=[]   # 시작점
for i in range(N):
    for j in range(N):
        if arr[i][j]==max_num:
            st.append((i,j))


# DFS
def dfs(i, j, depth):
    global ans

    if depth > ans:
        ans = depth

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni = i + di
        nj = j + dj
        if 0 <= ni < N and 0 <= nj < N:
            if arr[ni][nj] < arr[i][j] and v[ni][nj] == 0:
                v[ni][nj] = 1
                dfs(ni, nj, depth + 1)
                v[ni][nj] = 0

v = [[0] * N for _ in range(N)]
# main
for i in range(N):
    for j in range(N):
        if arr[i][j]==max_num: # 꼭대기 지점에서 DFS 수행
            v[i][j]=1
            dfs(i,j,1)
            v[i][j]=0


print(ans)

'''

# 깍을 수 있는 코드(최종코드)
'''
T=int(input())
for tc in range(1,T+1):
    N,K=map(int,input().split())
    arr=[]
    ans=0 # 경로 길이
    max_num=0 # 정상 높이


    for i in range(N):
        arr.append(list(map(int,input().split())))
        max_num=max(max_num,max(arr[i]))

    st=[]   # 시작점
    for i in range(N):
        for j in range(N):
            if arr[i][j]==max_num:
                st.append((i,j))


    # DFS
    def dfs(i, j, depth,ch): #ch: 산 깍을 기회
        global ans

        if depth > ans:
            ans = depth

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = i + di
            nj = j + dj
            if 0 <= ni < N and 0 <= nj < N:
                if arr[ni][nj] < arr[i][j] and v[ni][nj] == 0: # 방문 가능 + 낮은 높이로 이동할 때,
                    v[ni][nj] = 1
                    dfs(ni, nj, depth + 1,ch)
                    v[ni][nj] = 0
                elif arr[ni][nj] >= arr[i][j] and v[ni][nj]==0: # 방문 가능 + 같거나 큰 높이로 올라갈 때,
                    if ch: # 산 깍을 기회있고
                        if arr[ni][nj]-K<arr[i][j]: # 현재보다 낮게 깍아진다면
                            ch=False
                            tmp = arr[ni][nj]
                            arr[ni][nj]=arr[i][j]-1 # 가장 큰 높이로 자르기
                            v[ni][nj]=1
                            dfs(ni,nj,depth+1,ch)
                            ch=True
                            v[ni][nj] = 0
                            arr[ni][nj]=tmp # 높이 돌리기


    v = [[0] * N for _ in range(N)]
    # main
    for i in range(N):
        for j in range(N):
            if arr[i][j]==max_num: # 꼭대기 지점에서 DFS 수행
                v[i][j]=1
                dfs(i,j,1,True)
                v[i][j]=0



    print(f'#{tc}',ans)

'''


# 1860 진기의 최고급 붕어빵(복습)
'''
T=int(input())
for tc in range(1,T+1):
    N,K,M=map(int,input().split())
    lst=list(map(int,input().split()))
    max_second=max(lst)

    cnt=0
    ans='Possible'
    for i in range(max_second+1):
        if i!=0 and i%K==0:
            cnt+=M
        if i in lst:
            cnt-=1
            if cnt<0:
                ans='Impossible'
                break

    print(f'{tc}',ans)

'''

# 1220. [S/W 문제해결 기본] 5일차 - Magnetic
'''
N=int(input())
arr=[list(map(int,input().split())) for _ in range(N)]

cnt=0
for j in range(N):
    flg=False
    for i in range(N):
        if arr[i][j]==1:
            flg=True
        elif arr[i][j]==2 and flg==True:
            cnt+=1
            flg=False

print(cnt)

'''

# 2814 최장 경로
'''
def dfs(n,cnt):
    global ans

    v[n] = True

    for x in graph[n]:
        if v[x]!=True:
            dfs(x,cnt+1)

    v[n]=False

    if cnt>ans:
        ans=cnt


T=int(input())
for tc in range(1,T+1):
    N,M=map(int,input().split())
    graph=[[] for _ in range(N+1)]
    ans=0

    for _ in range(M):
        v1,v2=map(int,input().split())
        graph[v1].append(v2)
        graph[v2].append(v1)

    v=[False]*(N+1) # 방문 여부 리스트

    for i in range(1,N+1):
        dfs(i,1)

    print(f'#{tc}',ans)

'''

# 1244. [S/W 문제해결 응용] 2일차 - 최대 상금
'''
두 자리 수를 바꿔가면 교환 횟수가 N이 될때까지 재귀 호출
백트랙킹 원리: 교체 -> 재귀호출 -> 원위치 

'''
'''
def dfs(n,lst):
    global ans

    if n==K:
        num=int(''.join(map(str,lst)))
        ans=max(ans,num)
        return

    for i in range(len(lst)-1):
        for j in range(i+1,len(lst)):
            lst[i],lst[j]=lst[j],lst[i]
            check = int(''.join(map(str, lst)))
            if (n,check) not in v:
                v.append((n,check))
                dfs(n+1,lst)
            lst[i],lst[j]=lst[j],lst[i]

T=int(input())
for tc in range(1,T+1):
    N,K=map(int,input().split())
    lst=list(map(int,str(N)))
    ans=0
    v=[]
    dfs(0,lst)
    print(f'#{tc}',ans)
    
'''

# 3307 최장 증가 부분 수열
'''
T=int(input())
for tc in range(1,T+1):
    N=int(input())
    lst=list(map(int,input().split()))
    dp=[1]*N

    for i in range(1,len(lst)):
        for j in range(0,i):
            if lst[j]<=lst[i]:
                dp[i]=max(dp[j]+1,dp[i])

    print(f'#{tc}',max(dp))

'''

# 6057 그래프의 삼각형
'''
T=int(input())
for tc in range(1,T+1):
    N,M=map(int,input().split())
    graph=[[] for _ in range(N+1)]
    cnt=0

    for _ in range(M):
        v1,v2=map(int,input().split())
        graph[v1].append(v2)
        graph[v2].append(v1)

    for i in range(N-1):
        for j in range(i+1,N):
            for k in range(j+1,N+1):
                if i in graph[j] and j in graph[k] and k in graph[i]:
                    cnt+=1

    print(f'#{tc}',cnt)
    
'''

# 5215 햄버거 다이어트
'''
T=int(input())
for tc in range(1,T+1):
    N,K=map(int,input().split())
    items=[]
    for _ in range(N):
        v,w=map(int,input().split())
        items.append((v,w))

    dp=[0]*(K+1)

    for v,w in items:
        for i in range(K,w-1,-1):
            dp[i]=max(dp[i],dp[i-w]+v)

    print(f'#{tc}',dp[K])

'''

################# SSAFY 12 #################

# 1959 두 개의 숫자열

'''
T=int(input())

for tc in range(1,T+1):
    N,M=map(int,input().split())
    arr1=list(map(int,input().split()))
    arr2=list(map(int,input().split()))
    ans=0

    if len(arr1)>len(arr2):
        arr1,arr2=arr2,arr1

    for i in range(len(arr2)-len(arr1)+1):
        sm=0
        for j in range(len(arr1)):
            # print(arr1[j],arr2[i+j])

            sm+=arr1[j]*arr2[i+j]
        ans=max(sm,ans)

    print(f'#{tc}',ans)

'''

# 2001. 파리 퇴치
'''
T=int(input())

for tc in range(1,T+1):
    n,m=map(int,input().split())
    arr=[]
    ans=0

    for _ in range(n):
        arr.append(list(map(int,input().split())))

    for i in range(n-m+1):
        for j in range(n-m+1):
            sm=0
            for k in range(m):
                for l in range(m):
                    sm+=arr[i+k][j+l]
            ans=max(sm,ans)

    print(f'#{tc}',ans)

'''

# 1961 숫자 배열 회전

'''
def rotate(arr):
    arrR = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            arrR[i][j] = arr[n - 1 - j][i]
    return arrR

T=int(input())

for tc in range(1,T+1):
    n=int(input())
    arr=[list(map(int,input().split())) for _ in range(n)]



    arr1=rotate(arr)
    arr2=rotate(arr1)
    arr3=rotate(arr2)
    print(f'#{tc}')
    for a,b,c in zip(arr1,arr2,arr3):
        print(f'{"".join(map(str,a))} {"".join(map(str,b))} {"".join(map(str,c))}')

'''

# 1486 장훈이의 높은 선반
'''
def dfs(idx,sm):
    global ans

    if sm>=H:
        ans=min(ans,abs(sm-H))
        return
    if idx==N: # 마지막 인덱스여도 H보다 sm이 작을 경우는 ans 갱신안하고 리턴해야함, H이상인 경우였다면 위에서 걸려서 ans 갱신하고 리턴되므로
        return

    dfs(idx+1,sm+arr[idx])
    dfs(idx+1,sm)

T=int(input())
for tc in range(1,T+1):
    N,H=map(int,input().split())
    ans=1e9

    arr=list(map(int,input().split()))

    dfs(0,0)
    print(f'#{tc}',ans)

'''

# 1868. 파핑파핑 지뢰찾기

'''
내 생각
어느 위치를 먼저 클릭하는지에 따라 횟수가 달라짐
>> 클릭한 위치와 인접한 지뢰가 만약 0개라면 인전한 최대 8개의 칸들에 대해서도 자동으로 조사가 함께 조사하기 때문에 (BFS)

Q1. 한번 수행했다고 치면, 나머지에 대해서는 어떻게 수행함? 백트랙킹으로 모든 경우의 수를 따지나? 
>> 지뢰가 아무것도 없는 칸들을 먼저 전체 순회하면서 찾아 업데이트 시켜주고, 마지막에 남은 것들 차례로 해치우기 (빙고)

Q2. 그렇다면, 지뢰가 없는 칸 모두 조사하고 남은 지뢰가 없는 칸들에 대해서 클릭 횟수를 줄이는 방법은 뭐지?
>> 없음. 남은 개수들은 다 하나씩 체크해야함

'''

'''
풀이법

1. 클릭이 가능한 부분('.')을 찾아서 클릭을 할지 말지 결정
- 주변에 지뢰가 하나도 없다면 클릭, 카운트 추가
- 하나라도 있다면 패스

2. 클릭을 했다면, 주변에 대해 BFS탐색 
- 탐색하는 부분의 주변에도 지뢰가 하나도 없다면 BFS 탐색 진행
- 하나라도 있다면 STOP

3. 나머지 클릭 안된 부분들 카운트 더해주기

'''

# 시간초과 발생 ,,

'''
from collections import deque

def check(i,j):
    global cnt
    next_lst=[] # 지뢰 없을 시 다음 bfs 대상 노드들

    for k in range(8):
        ni=i+dx[k]
        nj=j+dy[k]

        if 0<=ni<N and 0<=nj<N:
            if arr[ni][nj]=='*': # 지뢰가 주변에 있다면 바로 탈출
                break
            elif arr[ni][nj]=='.':
                next_lst.append((ni,nj))

    else: # for문 전체 순회시 -> 모든 방면 지뢰가 없다면
        arr[i][j]='o' # 현재 노드 방문 체크
        cnt+=1 # 클릭 수 +1
        bfs(next_lst) # bfs 수행

# bfs 수행시 카운트를 추가안하고 변경해야 하므로, check함수와 조금 다름
def bfs(lst):
    q=deque(lst)

    while q:
        i,j=q.popleft()
        arr[i][j]='o' # 방문 체크
        next_lst=[]
        for k in range(8):
            ni=i+dx[i]
            nj=j+dy[j]
            if 0<=ni<N and 0<=nj<N:
                if arr[ni][nj] == '*':  # 지뢰가 주변에 있다면 바로 탈출
                    break
                elif arr[ni][nj] == '.':
                    next_lst.append((ni, nj))

        else:
            bfs(next_lst) # bfs 수행


def leftober_check(arr):
    global cnt

    for i in range(N):
        for j in range(N):
            if arr[i][j]=='.':
                cnt+=1


T=int(input())

for tc in range(1,T+1):
    N=int(input())
    arr=[list(input()) for _ in range(N)]
    dx = [1, 1, 1, 0, -1, -1, -1, 0]
    dy = [-1, 0, 1, 1, 1, 0, -1, -1]
    cnt=0


    for i in range(N):
        for j in range(N):
            if arr[i][j]=='.':  # 지뢰가 아닌 곳이라면 카운트 조건 확인 및 만족시 업데이트
                check(i,j)

    leftober_check(arr)
    print(f'#{tc}',cnt)


'''

# 부분 수열의 합

'''
# dfs
def dfs(idx,sm):
    global ans

    if sm>M:
        return

    if idx==N:
        if sm==M:
            ans+=1
        return
    
    dfs(idx+1,sm)
    dfs(idx+1,sm+lst[idx])

# main
t=int(input())
for tc in range(1,t+1):
    N,M=map(int,input().split())
    lst=list(map(int,input().split()))
    ans=0
    dfs(0,0)
    print(f'#{tc}',ans)

'''

# 공평한 분배2

'''
T=int(input())
for tc in range(1,T+1):
    N,K=map(int,input().split())
    lst=list(map(int,input().split()))
    lst.sort()
    ans=1e9

    for i in range(N-K+1):
        ans=min(ans,max(lst[i:i+K])-min(lst[i:i+K]))

    print(f'{tc}',ans)

    '''

# 보물상자 비밀번호
'''
T = int(input())
for tc in range(1,T+1):
    N, K = map(int,input().split())
    lst = list(map(str,input()))
    arr = [] # 회전 시킨 후 16진수 번호가 저장 될 리스트

    # 회전 수 
    cnt = N // 4
    for i in range(cnt):
        #리스트 끝 원소를 앞으로 이동 -> 한바퀴 이동
        pop_num = lst.pop() 
        lst.insert(0,pop_num)
        for j in range(0,N,cnt): # 리스트 내에서 각각 떨어져있는 요소를 하나의 번호로 합침 
            a = ''
            # 회전 시키고 각 변의 숫자 문자열로 만들어 arr에 삽입
            for k in range(j,j+cnt):
                a += lst[k]
            arr.append(a)
    set_lst = set(arr) # 중복 제거
    ten_num = [] # 10 진수 변환값 리스트 
    for num in set_lst: 
        ten_num.append(int(num,16))
    
    # 내림차순 정렬 -> 마지막 원소 프린트
    sorted_answer = sorted(ten_num,reverse=True)
    print(f'#{tc}',sorted_answer[K-1])

'''

# 5252 타일 붙이기

'''
T=int(input())
for tc in range(1,T+1):
    N=int(input())
    dp=[0]*(N+1)
    dp[1],dp[2],dp[3]=1,3,6

    if N>3:
        for i in range(4,N+1):
            dp[i]=dp[i-1]+dp[i-2]*2+dp[i-3] 
    print(f'{tc}',dp[N])

'''

#5248 그룹 나누기

'''
# 부모노드 탐색 함수
def find_root(x):
    if x == p_lst[x]:
        return x
    else:
        p_lst[x]=find_root(p_lst[x])
        return p_lst[x]

# 합집합 함수
def union(x,y):
    px=find_root(x)
    py=find_root(y)

    if px<py:
        p_lst[px]=py
    else:
        p_lst[py]=px


# main
T=int(input())
for tc in range(1,T+1):
    # N:노드 개수, M:쌍 개수, lst:입력 리스트, p_lst:루트노드 저장 리스트 
    N,M=map(int,input().split())
    lst=list(map(int,input().split()))
    p_lst=[0]*(N+1)

    # 부모노드 자기 자신으로 초기화
    for i in range(N+1):
        p_lst[i]=i

    # 그룹핑
    idx=0
    for _ in range(M):
        union(lst[idx],lst[idx+1])
        print(lst[idx], lst[idx+1])
        print(p_lst)
        idx+=2

    # 결국 원하는건 그룹의 개수
    # 그룹의 개수 => 대표자의 개수
    for i in range(1, N+1):
        p_lst[i] = find_root(i)

    # 자신 인덱스가 부모 노드값이라면, 루트이므로 cnt+1
    cnt=0
    for i in range(1,N+1):
        if i==p_lst[i]:
            cnt+=1
        

    print('부모 리스트 결과',p_lst)
    #print(f'#{tc}',cnt)

    '''


# 5249 최소 신장 트리

'''
크루스칼

- 모든 간선 오름 차순 정렬
- 차례로 순회하면서, 방문하지 않은 노드와 연결된다면 mst 추가 + 방문체크
- 방문체크 되어 있다면 PASS 

'''

'''
# search root node
def find_p(x):
    if x!=p[x]:
        p[x]=find_p(p[x])
    return p[x]


# union
def union(x,y):
    px=find_p(x)
    py=find_p(y)

    if px!=py:
        if px>py:
            p[py]=px
        else:
            p[px]=py

# main
T=int(input())
for tc in range(1,T+1):
    V,E=map(int,input().split())
    edges=[]
    p=[0]*(V+1)
    mst=0

    # p initial setting
    for i in range(V+1):
        p[i]=i

    # inset edge
    for _ in range(E):
        s,e,w=map(int,input().split())
        edges.append([s,e,w])

    edges.sort(key=lambda x:x[2])

    for edge in edges:
        s,e,w=edge
        print('s,e,w',s,e,w)
        if find_p(s)!=find_p(e):

            union(s,e)
            mst+=w
            # print('parent(s),parent(e)',find_p(s),find_p(e))
            # print('parent',p)
            # print('mst',mst)

    print(f'{tc}',mst)


    '''

# 5256 이항계수

'''
T=int(input())
for tc in range(1,T+1):
    n,a,b=map(int,input().split())

    dp=[[0 for _ in range(n+1)] for _ in range(n+1)]

    for i in range(n+1):
        for j in range(n+1):
            if i==0 and j==0:
                dp[i][j]=1
            else:
                dp[i][j]=dp[i-1][j]+dp[i-1][j-1]
                flag=True

    # for i in range(len(dp)):
    #     print(dp[i])

    print(f'#{tc}',dp[n][b])

    '''



# 1231 중이순회

'''
배운 점

- 순회 과정에서 모든 노드는 상대적인 root노드가 됨
- B노드가 A노드 기준의 오른쪽 자식 노드라고 할때, B노드는 자기 자식 노드들에게는 root 노드임
- 모든 노드들을 순회하면서 root노드를 출력 -> 모든 노드 출력 가능

'''

'''
# 왼쪽 자식 노드 반환 함수
def get_left_child(root_idx):
    left_idx = 2 * root_idx + 1

    # 자식 노드가 트리 안에 있다면 자식 노드 인덱스 반환
    if left_idx < len(lst):
        return left_idx
    
    # 범위 밖의 인덱스라면 None값 반환
    else:
        return None

# 오른쪽 자식 노드 반환 함수
def get_right_child(root_idx):
    right_idx = 2 * root_idx + 2

    if right_idx < len(lst):
        return right_idx
    
    else:
        return None

# 중위 순회
def in_order(root_idx):
    global ans

    if root_idx is not None:
        left_child = get_left_child(root_idx)
        in_order(left_child)  # 왼쪽 자식 방문

        ans+=lst[root_idx][1]
        #print(lst[root_idx][1])  # 루트 노드 출력 - 가장 첫번째 방문 노드: 트리의 오른쪽 가장 끝 노드

        right_child = get_right_child(root_idx)
        in_order(right_child)  # 오른쪽 자식 방문

# main
T=10
for tc in range(1,T+1):
    lst = []
    ans=''
    N = int(input())
    for _ in range(N):
        sub_lst = list(input().split())
        lst.append([int(sub_lst[0]), sub_lst[1]])
        
    # 중위 순회 탐색
    in_order(0)
    print(f'#{tc}',ans)
'''

'''
# 생각보다 더 쉽게 풀 수 있었다고 함 ,,

def inorder(tree,paraent,N):
    if paraent*2 <= N:
        inorder(tree,paraent*2,N)
    print(tree[paraent],end='')
    if paraent*2 +1 <=N:
        inorder(tree,paraent*2 +1,N)
 
T = 10
for test_case in range(1,T+1):
    N = int(input())
    tree = [0]
    for _ in range(N):
        temp = list(map(str,input().split()))
        tree.append(temp[1])
    print(f'#{test_case} ',end='')
    inorder(tree,1,N)
    print()

'''


'''
접근법

- 문제 예시로 step 별로 생각해봄

1. 후위 순회로 계산 (왼,오,중)

2. 오른쪽 자식 노드 부터 탐색

- None값(인덱스 밖) 리턴할 때까지 탐색

- 마지막 오른쪽 노드 끝 값 리턴 (부모 노드는 무조건 연산자)

4. 다시 돌아와서, 왼쪽 자식 노드 쪽으로 탐색 진행

- 연산자라면 후위 연산을 앞에서와 같이 진행

- 최종은 오른쪽 반환, 왼쪽 반환


5. 다시 돌아와서, 리턴 받은 값으로 연산 진행

Quetions
1. 자식 노드 어떻게 찾음? 완전이진트리 x 
>> 인덱스 어떻게 활용해야할까 ..?
>> 어떤 형태로 데이터를 저장해야할까 ..?

'''


'''
# 실패 코드
# 피연산자 계산 함수
def cal(num1,num2,operator):
    if operator=='+':
        return num1+num2
    elif operator=='-':
        return num1-num2
    elif operator=='*':
        return num1*num2
    else:
        return int(num1/num2)


# 후위순회
def postorder_traversal(node_idx):
    if lst[node_idx][1].isdigit():
        left_val=postorder_traversal(lst[node_idx][2]) 
        right_val=postorder_traversal(lst[node_idx][3])  
        return cal(left_val,right_val,lst[node_idx][1])
    else:
        return int(lst[node_idx][1])

# main
N=int(input())

lst=[list(input().split()) for _ in range(N)]


for sub_lst in lst:
    while len(sub_lst)!=4:
        sub_lst.append(0)

lst.insert(0,[0,0,0,0])

# print(lst)
ans=postorder_traversal(1)
print(ans)


'''

'''
# 피연산자 계산 함수
def cal(num1,num2,operator):
    if operator=='+':
        return num1+num2
    elif operator=='-':
        return num1-num2
    elif operator=='*':
        return num1*num2
    else:
        return num1//num2
    

def postorder_traversal(node_num):
    # 리프노드 시 갑 반환
    if (len(tree[node_num])) == 2:   
        return int(tree[node_num][1])
    
    # 아니라면(연산자 노드라면) DFS -> 리프 노드 반환 값 받아오기
    else:
        left_val = postorder_traversal(int(tree[node_num][2])) 
        right_val = postorder_traversal(int(tree[node_num][3])) 
         
        return cal(left_val,right_val,tree[node_num][1])



# main
N=int(input())

tree=[list(input().split()) for _ in range(N)]


for sub_tree in tree:
    while len(sub_tree)!=4:
        sub_tree.append(0)

tree.insert(0,[0,0,0,0])

# print(lst)
ans=postorder_traversal(1)
print(ans)

'''



# 1222 계산기1
'''
# 후위 표기식 변환 함수
def postfix_func(sik):

    # 후위 표기식
    postfix=[]

    # 연산자 스택
    oper_stack=[]

    for i in range(len(sik)):

        # 숫자라면 표기식에 넣기
        if sik[i].isdigit():
            postfix.append(int(sik[i]))

        # 연산자일때,
        # 연산자 스택이 비어있다면 푸쉬
        # 연산자 스택에 기존 연산자가 있다면 꺼내서 후위 표기식에 넣고, 연산자 넣기
        else:
            if len(oper_stack)==0:
                oper_stack.append(sik[i])
            else:
                postfix.append(oper_stack.pop())
                oper_stack.append(sik[i])


    # for문을 전체 순회하고, 남은 연산자 스택에 값이 있다면, 전부 후위 표기식에 넣기
    while True:
        if len(oper_stack)==0:
            break
        else:
            postfix.append(oper_stack.pop())

    return postfix



# 후위 표기식 계산 함수
def post_cal(postfix):

    num_stack=[]
    for i in range(len(postfix)):
        if postfix[i]=='+':
            num1=num_stack.pop()
            num2=num_stack.pop()
            num_stack.append(num1+num2)

        else:
            num_stack.append(postfix[i])

    return num_stack[0]

# main
T=10

for tc in range(1,T+1):

    N=int(input())
    
    # 중위 표기식(입력)
    sik=list(input())
    print(f'{tc}',post_cal(postfix_func(sik)))

'''