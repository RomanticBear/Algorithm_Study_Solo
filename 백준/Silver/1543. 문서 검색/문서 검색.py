
# 1543 문서 검색

str=input()
sub_str=input()

cur_idx=0
cnt=0

while True:
    if cur_idx+len(sub_str)>len(str):
        break

    # 여부 확인
    
    cur_str=str[cur_idx:cur_idx+len(sub_str)]

    if cur_str==sub_str:
        cnt+=1
        cur_idx+=len(sub_str)

    else:
        cur_idx+=1

print(cnt)
