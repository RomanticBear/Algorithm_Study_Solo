# 369

'''
num=int(input())

for i in range(1,num+1):
    
    n2=i//10  ## 10의 자리 숫자
    n1=i%10   ## 1의 자리 숫자
    
    if n2 in (3,6,9): # 10의 자리에 3, 6, 9 존재
        print('X', end="")
        if n1 in (3,6,9):
            print("X", end=" ")
        else:
            pass
            print(end=" ")
        
    elif n1 in (3,6,9): # 10의 자리에는 없지만 1의 자리에 3, 6, 9 존재
            print('X', end=" ")
            
    else: # 10, 1의 자리에 존재 x
        print(i, end=" ")   

'''


# 개미 집

'''
# 수동 입력 (예시)
arr=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
,[1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
,[1, 0, 0, 1, 1, 1, 0, 0, 0, 1]
,[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
,[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
,[1, 0, 0, 0, 0, 1, 0, 1, 0, 1]
,[1, 0, 0, 0, 0, 1, 2, 1, 0, 1]
,[1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
,[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
,[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
'''

'''
arr = []

for i in range(10):
    arr.append(list(map(int, input().split())))


# 출발 위치
x=1
y=1

# 동작 순서
# 오른쪽 (막히면) -> 아래쪽 -> 막히거나 찾으면 끝 : 찾을 때 arr[i][j]==2 (while문)

while(arr[x][y]!=2):  # arr값이 2가 아닐때 까지 반복 (즉, while문 종료 시 arr[x][y]==2)
    
    while(arr[x][y]==0): 
        arr[x][y]=9
        y+=1    # 오른쪽 한칸 이동 
        if arr[x][y]==2:   # if 이동값 : 2 -> 9변환
            arr[x][y]=9
            
    y-=1    # 왼쪽으로 한칸 이동
    x+=1    # 아래쪽으로 이동
    
    if arr[x][y]==0:  # 아래쪽 이동 값 : 0 -> 9변환 후 다시 내부 while문 실행  / 0이 아닐 경우 -> 막힘: 종료
        arr[x][y]==9
    else:
        break
   
if arr[x][y]==2:
    arr[x][y]=9
    
        
for i in range(10):
    for j in range(10):
        print(arr[i][j],end=" ")
    print()

'''

# CodeUp_Attendance


'''
# list 크기 선언
lst=[0 for i in range(23)]


# 부를 횟수 입력
num = int(input()) 

# 번호 값 입력 받기(n)
n = list(map(int,input().split()))


# 입력 받은 번호 -> list 횟수 추가
for i in range(num):
    lst[n[i]-1]+=1


# list 출력
for i in range(len(lst)):
    print(lst[i], end=" ")

'''


'''
# 처음 답안 

list=[0 for i in range(26)]

# 부를 횟수
num = int(input()) 

number=map(int,input().split())


# 번호 입력 받아 저장(n)
for i in range(num):
    number=int(input())
    n.append(number)

# 입력 받은 번호 -> list 횟수 추가
for i in range(num):
    list[n[i]-1]+=1

# list 출력
for i in range(len(list)):
    print(list[i], end=" ")
    

# print(n)  # 개행 없이 입력 받고 싶은데 for문 안에서 어떻게 하는거지 ,,?

>> a = list(map(int,input().split()))

'''

# CodeUp_last_6096

'''
# 바둑판
arr = []
for i in range(19):
    arr.append(list(map(int, input().split())))


# 입력 횟수
num=int(input())


# 좌표 값
# Step1. 가로 변경
for i in range(num):
    x,y=map(int,input().split())
    x-=1
    y-=1

    for i in range(19):
        if i==x:
            for j in range(19):
                if arr[x][j]==1:
                    arr[x][j]=0
                else:
                    arr[x][j]=1

# Step2. 세로 변경
    for j in range(19):
        if j==y:
            for i in range(19):
                if arr[i][y]==1:
                    arr[i][y]=0
                else:
                    arr[i][y]=1



# 출력 
for i in range(19):
    for j in range(19):
        print(arr[i][j],end=" ")
    print()

'''


# CodeUp_Rod block

'''
# 바둑판, 바둑판 크기
arr=[]
a, b = map(int,input().split())



# 바둑판 생성
for i in range(a):
    arr.append([])
    for j in range(b):
        arr[i].append(0)


# 입력받을 막대 개수
n = int(input())



# 동작 순서
# n-> 개수 / if d 값 (방향) / for -> ㅣ만큼 반복 : (x,y) 지점에서 start


for num in range(n):   
    l, d, x, y = map(int,input().split())  # 길이(l), 방향(d), 좌표 (x, y)
    x=x-1
    y=y-1

    if d==0:
        for i in range(a):
            for j in range(b):
                if i==x and j==y:
                    for k in range(l):
                        arr[i][j]=1
                        j+=1

    else:
        for i in range(a):
            for j in range(b):
                if i==x and j==y:
                    for k in range(l):
                        arr[i][j]=1
                        i+=1



# 출력
for i in range(a):
    for j in range(b):
        print(arr[i][j], end=" ")
    print()

'''

# CodeUp_The cross

'''

for i in range(19):
    arr.append([])
    for j in range(19):
        arr[i].append(0)


# 입력 횟수
num=int(input())



# 좌표 값
for i in range(num):
    x,y=map(int,input().split())

    for i in range(19):
        if i==x:
            for j in range(19):
                if arr[i][j]==1:
                    arr[i][j]=0
                else:
                    arr[i][j]=1

    for i in range(19):
        if i==y:
            for j in range(19):
                if arr[j][i]==1:
                    arr[j][i]=0
                else:
                    arr[j][i]=1


# 출력 
for i in range(19):
    for j in range(19):
        print(arr[i][j],end=" ")
    print()

'''



'''
## 응용: 좌표를 찍고 해당 되는 좌표의 가로, 세로 좌표 해당되는 줄  모두 한꺼번에 바뀜 (0->1, 1->0)

# 바둑판
arr=[]


for i in range(19):
    arr.append([])
    for j in range(19):
        arr[i].append(0)

# 입력 횟수
num=int(input())



# 좌표 값

arr_x=[]   # x좌표 저장 리스트
arr_y=[]   # y좌표 저장 리스트

# 횟수 만큼 입력한 좌표 값 arr_x, arr_y 각각 저장
for i in range(num):
    a,b=map(int,input().split())
    arr_x.append(a)
    arr_y.append(b)

# 바둑판 좌표 값 변경

# 원소 하나하나씩 돌아가다 arr_x, arr_y 하나라도 값이 존재한다면 값 변환
for i in range(19):
    for j in range(19):
        if i in arr_x or j in arr_y:
            if arr[i][j]==1:
                arr[i][j]=0
            else:
                arr[i][j]=1


# 출력 
for i in range(19):
    for j in range(19):
        print(arr[i][j],end=" ")
    print("\n")

'''
