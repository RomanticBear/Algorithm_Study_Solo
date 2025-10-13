# 2941 크로아티아 알파벳

'''
dz=, dz 의 경우 count 함수 접근 불가 -> dz 2개로 카운트 됨 

'''

cro_lst=['c=','c-','dz=','d-','lj','nj','s=','z=']

str=input()

cnt=0 # cro_lst 단어 개수 
word_sum=0 # cro_lst 글자 개수 


for word in cro_lst:

    if word=='z=':
        if 'dz=' in str:
            cnt+=(str.count('z=')-str.count('dz='))
            word_sum+=(str.count('z=')-str.count('dz='))*len(word)
            continue      
              
    cnt+=str.count(word)
    word_sum+=len(word)*str.count(word)

left_word=len(str)-word_sum


print(cnt+left_word)
