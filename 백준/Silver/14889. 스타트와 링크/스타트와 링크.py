
# team_dfs -> for문 변환 

def power_dfs(team):

    P_score=0
    
    for i in range(len(team)-1):
        for j in range(i+1,len(team)):
            P_score+=arr[team[i]][team[j]]+arr[team[j]][team[i]]

    return P_score


def team_dfs(idx,sub_lst):

    global ans

    # 가지치기 : 현재 뽑은 인원으로, 더 진행해봤자 절반에 해당하는 인원을 채울 수 없는 경우
    if len(sub_lst)+(len(lst)-idx)<len(lst)//2:
        return

    if len(sub_lst)==len(lst)//2 :
        if mark not in sub_lst:
            return
        
        else:
            teamA=[num for num in sub_lst]
            teamB=[num for num in lst if num not in sub_lst]
            # print(teamA,'|',teamB)

            P_teamA=power_dfs(teamA)
            P_teamB=power_dfs(teamB)

            ans=min(ans,abs(P_teamA-P_teamB))

        return
    
    team_dfs(idx+1,sub_lst+[lst[idx]])
    team_dfs(idx+1,sub_lst)


# main
N=int(input())
lst=[num for num in range(N)]
arr=[list(map(int,input().split())) for _ in range(N)]
ans=1e9
mark=0
team_dfs(0,[])
print(ans)