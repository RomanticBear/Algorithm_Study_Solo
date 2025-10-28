# 1914 하노이 탑

def hanoi(n,A,B,C):
    
    global depth
    global lst

    if n==1:
        depth+=1
        lst.append((A,C))
        return
    
    hanoi(n-1,A,C,B)
    depth+=1
    lst.append((A,C)) # n-1 판 해결, n번째 판 이동

    hanoi(n-1,B,A,C) # temp로 이동시킨 n-1판에 대한 이동


# main 
N=int(input())
depth=0
lst=[]


if N<=20:
    hanoi(N,1,2,3)
    print(depth)
    for start,end in lst:
        print(start,end)
else:
    print((1<<N)-1)