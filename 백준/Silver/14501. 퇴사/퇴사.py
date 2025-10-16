# 14501 퇴사
N=int(input())

lst=[(0,0)]
for _ in range(N):
    T,P=map(int,input().split())
    lst.append((T,P))

dp=[0]*(N+1)
#dp[1]=lst[1][1]

#print(lst[1][1])

for i in range(1,N+1):
    cur_t,cur_p=lst[i]

    for j in range(i):
        pre_t,pre_p=lst[j]

        # 현재 하려고 하는 일이, 마감기간 out
        if i+cur_t<=N+1:

            # 날짜가 겹쳐지지 않는다면, dp갱신
            if j+pre_t<=i:
                dp[i]=max(dp[i],dp[j]+cur_p)

    #print(dp)

print(max(dp))
