
# [문제]

'''
리스트 내에서 K라는 거리가 주어졌을 때 서로서로 K거리 이내를 만족하는 부분 리스트의 최대 개수(길이)

'''

# [테스트 케이스]

'''
T:3

in: 4 2
     6 4 2 3
out: #1 3

in: 4 3
    1 2 3 4
out: #2 4

in: 4 1
    1 3 7 5
out: #3 1

'''


# [풀이방법]

'''

리스트 lst [6,4,3,2] 가 주어졌을 때

1. 각각의 원소에서 자신을 포함한 K거리를 만족하는 부분 리스트 추출하면서 최대 크기 찾기 (temp list)

-> 4일때 [4,6,3,2] 만족

2. 4를 빼서 결과 리스트(res_lst) 에 넣어주고 (어떻게 되었든 4를 중심으로 한 최대 개수 존재한다는 것은 자명) 나머지 리스트 [6,3,2]를 lst로 초기화하고 1 과정 반복

3. while 탈출 조건: pop하지 못할 때 -> temp list의 크기가 1보다 작을 때

4. len(res_lst) + len(temp_lst) 빼주지 못한 1개 더해주기  *마지막 temp list가 두개 남았을 때 

'''


T=int(input()) # 테스트 케이스 개수

for tc in range(1,T+1):
        
    N,K=map(int,input().split())

    lst=list(map(int,input().split()))


    # 결과 담을 리스트
    res_lst=[]

    while(True):
        
        max_cnt=0 # K이내 거리 만족 최대 개수
        temp_lst=[] # K거리 이내 만족하는 최대 개수 리스트
        
        for i in range(len(lst)):
            cnt=0
            sub_lst=[] # 원소를 매번 담아두기 위한 리스트: max_cnt 만족시 temp_lst 초기화
            sub_lst.append(lst[i]) # 자기 차례 원소 먼저 삽입
            
            for j in range(len(lst)):
                if i==j: 
                    pass
                else:
                    if abs(lst[i]-lst[j])<=K:
                        cnt+=1
                        sub_lst.append(lst[j]) # K거리 이내 만족시 sub_lst 삽입
                    else:
                        pass

            # K거리 이내 만족하는 가장 원소가 많은 sub_lst -> temp_lst                    
            if cnt>max_cnt:
                max_cnt=cnt
                temp_lst=sub_lst



        if len(temp_lst)<=1:  # K거리 이내 리스트의 개수가 1개인 경우 -> 바로 탈출해줘야함 -> IF문 뒤에 POP연산이 와야함
            break
        
        else:
            pop_num=temp_lst.pop(0)
            res_lst.append(pop_num)
            lst=temp_lst

       
    res_cnt=len(res_lst)+1
    print("#{} {}".format(tc,res_cnt))





# [시험]

'''

가장 낮은 CNT 인덱스 부터 리스트에서 제거하고

나머지 사람들 CNT 계산 -> CNT==N이면 BREAK

'''
''''

T=int(input())

for tc in range(1,T+1):
    N,K=map(int,input().split())

    lst=list(map(int,input().split()))


    # while문

    cnt_lst=[]

    for i in range(len(lst)):
        cnt=0
        for j in range(N):
            if i==j: 
                pass
            else:
                if abs(lst[i]-lst[j])<=K:
                    cnt+=1
                else:
                    pass
        cnt_lst.append(cnt)



    # 가장 빈도수 많은 숫자 

    dic=dict()

    for i in cnt_lst:
        if i in dic:
            dic[i]+=1
        else:
            dic[i]=1

    dst=[0]*(max(cnt_lst)+1)

    for i in range(len(cnt_lst)):
        dst[cnt_lst[i]]+=1

    print("#{} {}".format(tc,max(dst)))
        
'''


