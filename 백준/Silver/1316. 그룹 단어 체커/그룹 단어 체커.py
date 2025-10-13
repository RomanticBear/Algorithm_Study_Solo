# 1316 그룹 단어 체커

def check_word(lst):

    flag=True
    stack=[]

    for word in lst:
        if word not in stack:
            stack.append(word)
        else:
            if word != stack[-1]:
                flag=False
                break
    
    return flag


N=int(input())
cnt=0
for _ in range(N):
    lst=list(input())
    cnt+=check_word(lst)

print(cnt)