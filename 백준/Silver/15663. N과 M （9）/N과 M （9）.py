# 156558 N과 M(9)
# 원소로 접근하면 중복되니 인덱스로 접근 -> visited로 인덱스 방문 표시
# 앞의 숫자를 기억하는 변수를 선언하여 다음 재귀문으로 넘겨주어 조건을 걸어줌

n,m=map(int,input().split())
lst=list(map(int,input().split()))
lst.sort()
visited=[False]*len(lst)
arr=[]



def func():
    remember_num=0
    if len(arr)==m:        
        print(*arr)    
        return

    for i in range(len(lst)):
        if visited[i]!=True and remember_num!=lst[i]:
            arr.append(lst[i])
            visited[i]=True
            remember_num=lst[i] # 다음 재귀문에서의 조건을 나타냄
            func()
            arr.pop()
            visited[i]=False

func()
