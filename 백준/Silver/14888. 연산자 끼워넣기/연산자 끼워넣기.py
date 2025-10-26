# 14888 연산자 끼워넣기

N=int(input())
lst=list(map(int,input().split()))
op1,op2,op3,op4=map(int,input().split())

INF = 10**10
max_ans = -INF
min_ans =  INF

def dfs(cur_idx,op1,op2,op3,op4,result):

    global max_ans
    global min_ans
    
    if cur_idx>=len(lst)-1:
        max_ans=max(max_ans,result)
        min_ans=min(min_ans,result)
        return
    
    else:
        if op1>=1:     
            dfs(cur_idx+1,op1-1,op2,op3,op4,result+lst[cur_idx+1])
        if op2>=1:
            dfs(cur_idx+1,op1,op2-1,op3,op4,result-lst[cur_idx+1])
        if op3>=1:
            dfs(cur_idx+1,op1,op2,op3-1,op4,result*lst[cur_idx+1])
        if op4>=1:
            dfs(cur_idx+1,op1,op2,op3,op4-1,int(result/lst[cur_idx+1]))

dfs(0,op1,op2,op3,op4,lst[0])

print(max_ans)
print(min_ans)