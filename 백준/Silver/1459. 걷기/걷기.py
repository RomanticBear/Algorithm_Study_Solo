# 1459 걷기

# 좌표나 시간에 따른 기준으로 분기하기에 복잡 
'''
X,Y,W,S=map(int,input().split()) # W: 직선 시간 / S: 대각선 시간
ans=0

if W>=S*2:
    ans=(X+Y)*S
else:
    L_D,S_D=max(X,Y),min(X,Y)

    if (X+Y)%2==0:
        # CASE1: 대각선 이동 
        CASE1=L_D*W

        # CASE2: 대각선 + 직선 이동  
        CASE2=S_D*W+(L_D-S_D)*S
        ans=min(CASE1,CASE2)

    else:
        pass
print(ans)

'''

# 경로에 따라 분기하자. 

# 1. 직선으로만 조진다.

# 2. 대각선으로만 조진다.
# 2-1. X+Y가 짝수라면 대각선으로만 조진다.
# 2-2. X+Y가 홀수라면 한번 직선으로 가고, 대각선으로 조진다. (최대한 대각선으로 조지는 방법)

# 앞의 방법에서 고려하지 못한 문제
# 3. 대각선으로 초기 이동하되, 지나치지는 말고, 나머지는 직선으로 가자.
# 왜냐하면, 대각선으로만 조지는 케이스는 다른 경우의 수로 계산함


X,Y,S,W=map(int,input().split()) # S: 직선 시간 / W: 대각선 시간

DIST1=(X+Y)*S

if (X+Y)%2==0:
    DIST2=max(X,Y)*W
else:
    DIST2=(max(X,Y)-1)*W+S

DIST3=min(X,Y)*W+(abs(X-Y)*S)

print(min(DIST1,DIST2,DIST3))