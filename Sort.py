# 정렬 _Youtube(이코테)

# 선택 정렬

'''
array=[7,5,9,0,3,1,6,2,8,]

for i in range(len(array)):
    min_index=i
    for j in range(i+1,len(array)):
        if array[min_index]>array[j]:
            min_index=j
    array[i],array[min_index]=array[min_index],array[i]

print(array)

'''

# 삽입 정렬

'''
array=[7,5,9,0,3,1,6,2,4,8]


for i in range(1,len(array)): # 첫 번째 데이터는 정렬되어 있다고 판단 → 두 번째 데이터부터 시작 
    for j in range(i,0,-1):
        if array[j]<array[j-1]: # 특정 데이터 왼쪽에 있는 데이터들은 이미 정렬이 된 상태이므로 살펴볼 필요 x
            array[j],array[j-1]=array[j-1],array[j]
        else:
            break
print(array)

'''


# 퀵 정렬

'''
array=[5,7,9,0,3,1,6,2,4,8]

def quick_sort(array, start, end):

    # 원소가 1개인 경우 (부등호 '>'는 왜 들어감 ,,?)
    if start>=end:
        return array
    
    pivot=start 
    left=start+1 
    right=end 

    while left<=right:
        
        # 왼쪽에서 부터 - 피벗보다 큰 데이터를 찾을 때까지 반복
        while left<=end and array[left]<=array[pivot]:
            left+=1
            
        # 오른쪽에서 부터 - 피벗보다 작은 데이터를 찾을 때까지 반복
        while right>start and array[right]>=array[pivot]:
            right-=1

        # 엇갈렸다면 작은 값과 피벗 값 교체
        if left>right:
            array[right],array[pivot]=array[pivot],array[right]

            # 교체만 시켜주고, while문 탈출을 첫번째 while문 루프
            
        # 엇갈리지 않았다면 작은 값과 큰 값 교체
        else:
            array[right],array[left]=array[left],array[right]


        quick_sort(array,start,right-1)
        quick_sort(array,right+1,end)  # pivot은 자신의 왼쪽으로 작은 값, 오른쪽으로 큰 값 -> 즉, 알맞은 위치에 정렬되었으므로 재귀에 포함시키지 x

quick_sort(array,0,len(array)-1)
print(array)

'''       


# 파이썬 장점 살린 퀵 정렬

'''
array=[5,7,9,0,3,1,6,2,4,8]

def quick_sort(array):
    # 원소가 하나인 경우 종료
    if len(array)<=1:
        return array

    pivot=array[0] 
    tail=array[1:] # 피벗을 제외한 리스트


    left_side=[x for x in tail if x<=pivot]
    right_side=[x for x in tail if x>pivot]

    return quick_sort(left_side) + [pivot] + quick_sort(right_side)

print(quick_sort(array))

'''

# 계수 정렬

'''
# 모든 원소의 값이 0보다 크거나 같다고 가정
array=[7,5,9,0,3,1,6,2,9,1,4,8,0,5,2]

# 모든 범위를 포함하는 리스트 선언
count=[0]*(max(array)+1)

for i in range(len(array)):
    count[array[i]]+=1 # 각 데이터에 해당하는 인덱스의 값 증가

for i in range(len(count)): # 각 인덱스에
    for j in range(count[i]): # 개수만큼 
        print(i, end=' ')

'''


# 이진 탐색

# 재귀함수 _ 이진탐색 구현

'''
def binary_search(array,target,start,end):

    if start>end: # 종료 조건: 탐색값 존재 x
        return None
    
    mid=(start+end)//2

    if array[mid]==target:# 탐색 성공
        return mid

    elif array[mid]<target: # 중간값보다 탐색값보다 크다면
        return binary_search(array,target,mid+1,end)

    else: # 중간값 보다 탐색값이 작다면
        return binary_search(array,target,start,mid-1)

        
n,target=list(map(int,input().split()))
array=list(map(int,input().split()))

result=binary_search(array,target,0,n-1)

if result==None:
    print("원소 존재 x")
else:
    print(result+1)

'''

# 반복문 _ 이진탐색 구현

'''
def binary_search(array,target,start,end):
    while start<=end:
        mid=(start+end)//2

        if array[mid]==target:
            return mid
        
        elif array[mid]>target:
            end=mid-1
        else:
            start=mid+1

    return None  # 탈출 조건 -> 원소가 없을 때이므로


n,target=list(map(int,input().split()))
array=list(map(int,input().split()))

result=binary_search(array,target,0,n-1)

if result==None:
    print("원소 존재 x")
else:
    print(result+1)
        
'''



# 부품 찾기(실전 문제)
# 코드1(이진탐색)
'''
def binary_search(array,target,start,end):

    if start>end:
        return False

    mid=(start+end)//2

    if array[mid]==target:
        return True

    elif array[mid]<target:
        return binary_search(array,target,mid+1,end)

    else:
        return binary_search(array,target,start,mid-1)



N=int(input())
st_lst=list(map(int,input().split())) # 가게 보유 리스트
st_lst.sort()

M=int(input())
cu_lst=list(map(int,input().split())) # 손님 확인 요청 리스트

result=0
for i in range(len(cu_lst)):
    target=cu_lst[i]
    result=binary_search(st_lst,target,0,N-1)

    if result==True:
        print('yes',end=' ')
    else:
        print('no',end=' ')

'''

# 코드2(계수정렬)

'''
n=int(input())
array=[0]*1000001 # M(손님) 최대 입력 개수

for i in input().split():
    array[int(i)]=1

m=int(input())
x=list(map(int,input().split()))

for i in x:
    if array[i]==1:
        print('yes', end=' ')
    else:
        print('no', end=' ')

'''

# 코드3(set사용)

'''
# 단순히 유무 판단만 하는 문제 -> set 연산자 사용

n=int(input())
array=set(map(int,input().split()))

m=int(input())
x=list(map(int,input().split()))

for i in x:
    if i in array:
        print('yes', end=' ')
    else:
        print('no', end=' ')
    
'''


# 떡볶이 떡 만들기(실전문제)
# 파라메트릭 서치 문제 유형 - 재귀적 구현 < 반복문 구현

'''
N,T=map(int,input().split())
array=list(map(int,input().split()))


start=0
end=max(array)

while start<=end:

    # 떡 길이 합
    sum=0
    mid=(start+end)//2 # 길이 단위

    # array 인덱스 순차 접근
    for i in range(len(array)):

        if mid<array[i]: # 길이로 접근 -> 중간값보다 길이가 긴 경우만 절단 수행
            sum+=array[i]-mid

    if sum<T: # 주문한 떡보다 작은 경우 -> 좀 더 높이 낮추기 -> end = mid-1
        end=mid-1
    else:
        result=mid # T를 만족 시키면서, 가장 값을 작게 가져갔을 때가 답 -> result에 기록
        start=mid+1
                

print(result)

'''



# 필터 1895

'''
import sys

R,C=map(int,input().split())

# 입력 받을 매트릭스
matrix=[]
for i in range(C):
    matrix.append(list(map(int,sys.stdin.readline().split())))

# 필터링된 결과 저장할 매트릭스
result_mtx=[]

# 필터 매트릭스 -> 중간값 도출 -> result_mtx 추가
filter_mtx=[]

st_row=0
st_col=0

while True:
        
    for i in range(R-2):
        for j in range(C-2):
            filter_mtx.append(matrix[i:i+3][j:j+3])
            filter_mtx.sort()
            result_mtx.append(filter_mtx[len(filter_mtx)//2])
            print(result_mtx)

'''

# 18870 좌표 압축

'''
# 시간 초과

import sys

N=int(input())
lst=list(map(int,sys.stdin.readline().split()))
# print(lst)

copy=[0 for i in range(N)] # 순위 저장을 위한 리스트
lst2=sorted(lst) # 원본값 정렬한 리스트
# print(lst2)

rank=0

copy[0]=0
for i in range(1,N):
    if lst2[i]!=lst2[i-1]:
        rank+=1
        copy[i]=rank
    else:
        copy[i]=rank
      
# print(copy)

for i in range(N):
    for j in range(N):
        if lst[i]==lst2[j]:
            lst[i]=copy[j]
        else:
            pass

print(lst)

'''

# 시간 초과

'''
import sys

N=int(input())
lst=list(map(int,sys.stdin.readline().split()))

# 정렬 수행 -> 중복 제거
arr=sorted(set(lst)) 

# 딕셔너리 키 값 -> 인덱스 순서(val)
dic={arr[i]:i for i in range(len(arr))}

# 키 값 호출 시 val값 출력
# dic 인덱스 자료형 -> 리스트 동일
for i in lst:
    print(dic[i], end=' ')
    
'''



# 1302 베스트 셀러

'''
import sys

N=int(input())

dic={}

for i in range(N):
    name=input()

    # 딕셔너리 키 존재 x -> 추가하고 val->1
    if name not in dic:
        dic[name]=1

    # 키 존재 -> val+=1
    else:
        dic[name]+=1

# val 최대값 저장
max_val=max(dic.values())

lst=[]
for key, val in dic.items():

    # val값이 최대값이라면 리스트에 추가
    if val==max_val:
        lst.append(key)

# 오름 차순으로 기본 정렬 -> 첫번째 원소 출력
lst=sorted(lst)
print(lst[0])

'''

# 1920 수 찾기



# 문제) M개 입력 받은 리스트의 각각의 원소가 N개 입력 리스트에 존재하면 1, 안하면 0 출력

'''
N=int(input())
arr1=list(map(int,input().split()))

# 핵심 -> 이진탐색 조건 : 정렬된 리스트
arr1.sort()

M=int(input())
arr2=list(map(int,input().split()))

# 이진탐색
def binary_search(array,target,start,end):

    while start<=end:
        mid= (start+end)//2

        if array[mid]==target:
            return True

        elif array[mid]>target:
            end=mid-1

        else:
            start=mid+1

    return False



for i in range(len(arr2)):

    # 탐색 값 -> arr2 
    target=arr2[i]
    result=binary_search(arr1,target,0,len(arr1)-1)  # 이진탐색 대상 arr1 -> len 길이 주체

    if result==True:
        print(1)
    else:
        print(0)

'''


# 10816 숫자 카드 2
# 원소 개수 구하기 (노션 -> bisect 함수 정리)
# 실버3 (2023-11-04)

'''
import sys
from bisect import bisect_left, bisect_right

input=sys.stdin.readline


def count_by_range(array,left_val,right_val):
    right_idx=bisect_right(array,right_val)
    left_idx=bisect_left(array,left_val)

    return right_idx-left_idx
    


N=int(input())
arr1=list(map(int,input().split()))

# 핵심 -> 이진탐색 조건 : 정렬된 리스트
arr1.sort()

M=int(input())
arr2=list(map(int,input().split()))

# 이진탐색
def binary_search(array,target,start,end):

    while start<=end:
        mid= (start+end)//2

        if array[mid]==target:
            return True

        elif array[mid]>target:
            end=mid-1

        else:
            start=mid+1

    return False



for i in range(len(arr2)):

    # 탐색 값 -> arr2 
    target=arr2[i]
    result=binary_search(arr1,target,0,len(arr1)-1)  # 이진탐색 대상 arr1 -> len 길이 주체

    if result==True:
        cnt=count_by_range(arr1,target,target)
        print(cnt,end=' ')
    else:
        print(0,end=' ')

'''

# 2805 나무 자르기


'''
import sys
input=sys.stdin.readline

N,K=map(int,input().split())

array=list(map(int,input().split()))


result=0

def binary_cut(arr,target,start,end):

    while start<=end:
        length=0 # 매번 길이가 바뀔때마다 가져가는 길이(length)가 바뀌므로 while문 내 선언
    
        # start=0, end=max(arr)
        mid=(start+end)//2
        
        # 가져갈 수 있는 길이 계산
        for i in range(N):
            if mid<arr[i]:
                length+=arr[i]-mid

        if length==target:
            return mid

        # 목표하는 길이보다 크다면 -> 일단 저장 -> 이진 탐색
        elif length>target:
            result=mid 
            start=mid+1

        # 길이가 부족하다면
        else:
            end=mid-1
            
    # 반복문 탈출 시 임시 저장되어 있던 길이 반환
    return result

result=binary_cut(array,K,0,max(array)-1)
print(result)

'''      

# 24480 알고리즘 수업-깊이 우선 탐색 2 >> 모르겠음












                   
        

        
