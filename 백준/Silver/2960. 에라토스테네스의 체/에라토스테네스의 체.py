# 2960 에라토스테네스의 체

N,K=map(int,input().split())

lst=[i for i in range(2,N+1)]

flag=False # 이중 for문 탈출 
ans=0
check_lst=[]

for i in range(len(lst)):
    if lst[i] not in check_lst and K!=0:
        check_lst.append(lst[i])
        K-=1

        # print('#',lst[i],check_lst,K)
        if K==0:
            ans=lst[i]
            break

        for j in range(i+1,len(lst)):
            if lst[j]%lst[i]==0:
                if lst[j] not in check_lst:
                    check_lst.append(lst[j])
                    K-=1
                if K==0:
                    ans=lst[j]
                    flag=True
                    break
            
            # print(lst[i],lst[j],check_lst,K)

    if flag:
        break
    
    # print(lst[i],lst,K)

print(ans)