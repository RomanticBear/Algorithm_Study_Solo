# 15665 N과 M(11)

n,m=map(int,input().split())
arr=list(map(int,input().split()))
arr.sort()
lst=[]


def func():
    if len(lst)==m:
        print(*lst)
        return
    remember_num=0
    for i in range(n):         
        if arr[i]!=remember_num:
            lst.append(arr[i])
            remember_num=arr[i]
            func()
            lst.pop()


func()