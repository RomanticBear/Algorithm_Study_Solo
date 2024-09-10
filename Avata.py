'''
arr=list(map(int,input()))

# 리스트 정수 개수 -> 딕셔너리 출력
def count_num(arr):
    dic=dict()
    for i in arr:
        if i not in dic:
            dic[i]=1
        else:
            dic[i]+=1
    return dic

dic=count_num(arr)

# 가장 큰 value 키
max_key=max(dic, key=dic.__getitem__)

# 가장 큰 vlaue 값
max_val=max(dic.values())



if max_key!=6 and max_key!=9:
    set_num=max_val
else:
    if len(dic)==1: # 전부 6 or 전부 9 
        set_num=round(max_val/2)
        
    else:    
        max_sn=(round(max_val/2))  # 2.5개일 경우 -> 3개 필요        
        dic_t = {key : value for key, value in dic.items() if key != 6 and key!=9}
        
        max_val_t=max(dic_t.values())
        set_num=max(max_sn,max_val_t)

print(set_num)

'''


'''
max_count 키 값

case 1. 
6,9가 아닌 숫자 -> max_count 만큼 세트 필요

case 2.
6.9이라면 -> around(max_count/2) 와 다음으로 많은 숫자 비교 -> 큰 숫자 -> 세트개수

       if, 6개수:4, 4개수:3 -> 필요한 set 개수:3

'''

# 알아낸점 -> 딕셔너리는 for문 삭제가 안됨 (반복 도중 크기 바껴서 에러 발생)
# 문제점 -> 런타임 에러(실패)


# 이코테 구현 (3.게임 개발)

N,M=map(int,input().split())
x,y,d=map(int,input().split())
arr=[]
for _ in range(N):
    arr.append(list(map(int,input().split())))


# 북,동,남,서
dx=[1,0,-1,0]
dy=[0,1,0,-1]
cnt=0
obs=0  # 바다나 막다른 길, 사방이 막히면 종료(종료 조건)

while True:
    if arr[x][y]==0:
        cnt+=1
        arr[x][y]=1
        obs=0
    print(arr[x][y],(x,y))
    d = (d+1)% 4  # 바라 보는 방향
    nx=x+dx[d]
    ny=y+dy[d]

    if obs>=4:
        break

    if 0<=nx<N and 0<=ny<M:
        if arr[nx][ny]==0:  # 대륙내 육지라면 -> 이동
            x=nx
            y=ny
        else:  # 바다라면 장애물 +1
            obs+=1

    else:   # 대륙 밖
        obs+=1




print(cnt)

























