# 파이썬 문법

# 1. Sequence Type

# 리스트 슬라이싱
'''
lst=[1,2,3,4,5]

print(lst[::-1])    # [5, 4, 3, 2, 1]
print(lst[-1:-6])   # []
print(lst[-1:-6:-1])    # [5, 4, 3, 2, 1]
print(lst[-1:-6:-2])     # [5, 3, 1]


string='hello'

string[1]='H' # 에러 - 문자열 불변 <-> 리스트 가변

'''


# 문자열 조작 메서드
'''
text='Hello, world!'

# replace
new_text=text.replace('world','Python')
print(new_text) # Hello, Python!

# strip
text='  Hello, world!  '

new_text=text.strip()
print(new_text) # Hello, world!


# split
text='Hello, world!'

new_text=text.split(',')
print(new_text) # ['Hello', ' world!']


# join
words=['Hello','world!']

text='-'.join(words)
print(text) # Hello-world!

'''

# 리스트 메서드
'''
my_lst=[1,2,3]

my_lst.extend([5,6])    # [1, 2, 3, 3, 5, 6]
my_lst+=[7,8]   # [1, 2, 3, 5, 6, 7, 8]

'''

# 튜플 메서드
# 튜플: 여러 개의 값을 순서대로 저장하는 변경 불가능한 시퀀스 자료형, 모든 자료형의 데이터 저장 가능
# 튜플의 불변 특성을 사용하여 안전하게 여러개의 값을 전달하기 위해 사용, 그룹화, 다중할당 등
# 개발자가 직접 사용하기 보다 '파이썬 내부 동작'에서 주로 사용

'''
my_tuple=(1,'a',2,'b',3)

my_tuple[1]='z' # 에러, 튜플 불변

'''

###############################################

# 2. Non-Sequence Type


# 딕셔너리 메소드
person={'name':'Alice', 'age':25}

# get
'''
print(person.get('name'))   # Alice
print(person.get('country'))    # None
print(person.get('country','Unknown'))  # Unknown

'''

# keys, values, items
'''
# keys
print(person.keys())   # dict_keys(['name', 'age'])
print(list(person.keys()))   # ['name', 'age']

# values
print(person.values()) # dict_values(['Alice', 25])

# items
print(person.items())   # dict_items([('name', 'Alice'), ('age', 25)])
print(list(person.items()))  # [('name', 'Alice'), ('age', 25)]

'''

# pop
'''
print(person.pop('age')) # 25
print(person) # {'name':'Alice}
print(person.pop('country',None)) # None
print(person.pop('country')) # KeyError
print(person.pop()) # TypeError

'''

# set
# 순서와 중복이 없고, *변경 가능한* 자료형

# 집합 연산
'''
my_set1={1,2,3}
my_set2={3,6,9}

print(my_set1|my_set2)  # {1, 2, 3, 6, 9}
print(my_set1-my_set2)  # {1, 2}
print(my_set1&my_set2)  # {3}

'''

# add,remove
'''
my_set={1,2,3}
my_set.add(4)   # {1, 2, 3, 4}
my_set.add(2)   # {1, 2, 3, 4}

my_set.remove(3) # {1,2,4}
my_set.remove(3) # KeyError
'''

# None
# '값이 없음'을 표현하는 자료형
# 함수의 return이 없는 경우 None 반환
'''
def func():
    print('aaa')

print(func())

# aaa
# None
'''


###############################################
# p.105 ~ 116
# 3. 복사

# 변경 가능 타입: 리스트, 딕셔너리, 집합
# 변경 불가능 타입: 나머지
'''
dic_1={'name':'Alice','age':15}
dic_2=dic_1

dic_2['country']='Busan'
print(dic_1)    # {'name': 'Alice', 'age': 15, 'country': 'Busan'}

del dic_1['name']
print(dic_2)    # {'age': 15, 'country': 'Busan'}

'''

# 복사 유형

# 할당(Assignment)
# 할당 연산자(=)를 통한 복사는 해당 객체에 대한 객체 참조를 복사
'''
lst1=[1,2,3]
lst2=lst1

lst2[0]=4
print(lst1,lst2)    # [4, 2, 3] [4, 2, 3]

'''


# 얕은 복사 
# 슬라이싱으로 생성된 객체는 원본 객체와 독립적으로 존재

'''
a=[1,2,3]
b=a[:]

b[0]=100
print(a,b)  # [1, 2, 3] [100, 2, 3]

'''

# 얕은 복사 _ 한계
# 2차원 리스트와 같이 변경 가능한 객체 안에 변경 가능한 객체 있는 경우, 슬라이싱으로 처리 불가
# a와 b 주소는 다르지만 내부 객체의 주소는 같기 때문에 함께 변경됨
'''
a=[1,2,[1,2]]
b=a[:]

b[0]=100
print(a,b)  # [1, 2, [1, 2]] [100, 2, [1, 2]]

b[2][0]=200
print(a,b)  # [1, 2, [200, 2]] [100, 2, [200, 2]]
'''

# 깊은 복사
# 내부에 중첩된 모든 객체까지 새로운 객체 주소를 참조하도록 함
'''
import copy

a=[1,2,[1,2]]
b=copy.deepcopy(a)

b[2][0]=100
print(a,b)  # [1, 2, [1, 2]] [1, 2, [100, 2]]

'''


###############################################

# 4. Type Conversion

# 암시적 형변환
# Boolean과 Numeric Type에서만 가능
'''
print(3+5.0)    # 8.0 
print(True+3)   # 4
print(True+False)   # 1
'''

# 명시적 형변환
# str->integer
# 형식에 맞는 숫자는 변경 가능
'''
print(int('1'))     # 1
print(int('3.5'))   # ValueError
print(int(3.5))     # 3
print(float('3.5')) # 3.5
'''

# integer -> str
# 모두 가능



###############################################

# 5. Operator
# 비교 연산자 ==, is
'''
- ==는 동등성(equality), is는 식별성(identity)
- ==은 값(데이터)을 비교하는 것이지만, is는 레퍼런스(주소)를 비교(객체 비교)
print(1 is True) # False
print(2 is 2.0) # False

'''

# 단축평가
# 논리 연산에서 두 번째 피연산자를 평가하지 않고 결과를 결정하는 동작
'''
vowels='aeiou'

# 1
print(('a' and 'b') in vowels) # False
print(('b' and 'a') in vowels) # True

# 2
print(('a' or 'b') in vowels) # True
print(('b' or 'a') in vowels) # False

# 3
print(3 and 5) # 5
print(3 and 0) # 0
print(0 and 3) # 0
print(0 and 0) # 0
print(5 or 3) # 5
print(3 or 0) # 3
print(0 or 3) # 3
print(0 or 0) # 0
'''

'''
# 1
- 'a' 와 'b' 모두 vowels에 포함되어 있는지 물어본 것 x
- ('a' and 'b') >> 두 연산자 모두 True 인지 확인 >> True이므로 'b' 반환 >> 'b'는 vowels에 포함 x >> False 반환

# 2
- ('a' or 'b') >> 둘 중 하나만 True이면 True 반환 >> 'a'가 True이므로 뒷부분 'b'평가 안함 >> 'a'는 vowels에 포함 >> True 반환
'''

# 멤버십 연산자
# 특정 값이 시퀀스나 다른 컬렉션에 속하는지 여부를 확인
# in, not in

# 시퀀스 연산자
# + 와 *는 시퀀스 간 연산에서 산술 연산자일때와 다른 역할을 가짐
# +: 결합 연산자
# *: 반복 연산자


###############################################

# 6. 제어문

# 반복 가능한 객체(iterable)
# 시퀀스 객체 뿐만 아니라 dict, set 등도 포함

# List Comprehension
'''
numbers=[1,2,3,4,5]

squared_numbers=[num**2 for num in numbers]    # [1, 4, 9, 16, 25]
print(squared_numbers)
squared_odd_numbers=[num**2 for num in numbers if num%2!=0]    # [1, 4, 9, 16, 25]
print(squared_odd_numbers)
'''

###############################################

# 7. 함수

# 매개변수, 인자
'''
매개변수(parameter)
- 함수 정의 시, 함수가 받을 값을 나타내는 변수

인자(argument)
- 함수를 호출할 때, 실제로 전달되는 값
'''

# 1. 임의의 인자 목록

#정해지지 않은 개수의 인자를 처리하는 인자
#함수 정의 시 매개변수 앞에 '*'를 붙여 사용하며, 여러 개의  인자를 튜플로 처리
'''
def calculate_sum(*args):
    print(args)     # (1, 2, 3)
    print(sum(args)) # 6

calculate_sum(1,2,3)
'''

# 2. 임의의 키워드 인자 목록

# 정해지지 않은 개수의 키워드 인자를 처리하는 인자
# 함수 정의 시 매개변수 앞에 '**'를 붙여 사용하며, 여러 개의 인자를 dictionary로 묶어 처리
'''
def print_info(**kwargs):
    print(kwargs)

print_info(name='EVE',age=30)   # {'name': 'EVE', 'age': 30}
'''

# 3. 함수 인자 권장 작성순서
'''
- 위치 -> 기본 -> 가변 -> 가변 키워드
- 호출 시 인자를 전달하는 과정에서 혼란을 줄이도록 작성

def func(pos1, pos2, default_arg='default', *args, **kwargs)

'''

'''
def func(pos1, pos2, default_arg='default', *args, **kwargs):
    print('pos1: ',pos1)
    print('pos2: ',pos2)
    print('defalut_arg: ',default_arg)
    print('args: ',args)
    print('kwargs: ',kwargs)

func(1,2,3,4,5,6,key1='val1',key2='val2')

pos1:  1
pos2:  2
defalut_arg:  3
args:  (4, 5, 6)
kwargs:  {'key1': 'val1', 'key2': 'val2'}

'''

###############################################

# 8. 패킹, 언패킹

# 패킹
'''
# 변수에 담긴 값들은 튜플 형태로 묶임
pack_val=1,2,3,4,5
print(pack_val)    # (1, 2, 3, 4, 5)


# '*'을 활용한 패킹
# print 함수에서 임의의 가변 인자를 작성할 수 있는 이유
# 인자 개수에 상관 없이 튜플 하나로 패킹되어 내부에서 처리하기 때문
numbers=[1,2,3,4,5]
a,*b,c=numbers
print(a,b,c)    # 1 [2, 3, 4] 5
'''

# 언패킹
'''
# 패킹된 변수의 값을 개별적인 변수로 분리하여 할당하는 것

packed_val=[1,2,3,4,5]
a,b,c,d,e=packed_val    
print(a,b,c,d,e)    # 1 2 3 4 5

# '*'을 활용한 언패킹
def my_func(x,y,z):
    print(x,y,z)

name=['a','b','c']
my_func(*name)  # a b c

# '**'을 활용한 언패킹
# **는 딕셔너리의 키-값 쌍을 언패킹하여 함수의 키워드로 인자 전달

def my_func(x,y,z):
    print(x,y,z)

my_dict={'x':1,'y':2,'z':3}
my_func(**my_dict) # 1 2 3

'''

# 패킹, 언패킹 연산자 정리
'''
'*'
- 패킹 연산자로 사용될 때, 여러 개의 인자를 하나의 튜플로 묶음
- 언패킹 연산자로 사용될 때, 시퀀스나 반복 가능한 객체를 각각의 요소로 언패킹하여 함수의 인자로 전달

'**'
- 언패킹 연산자로 사용될 때, 딕셔너리 키-값 쌍을 언패킹하여 함수의 키워드 인자로 전달

'''

###############################################

# 9. 기타

# lambda 내장함수
'''
numbers=[1,2,3,4,5]
squares=list(map(lambda x:x*x, numbers))    # [1, 4, 9, 16, 25]
print(squares)
'''

# enumerate 내장함수
'''
# iterable 객체의 각 요소에 대해 인덱스와 함께 반환하는 내장함수

fruits=['apple','banana','cherry']

for idx,fruit in enumerate(fruits):
    print(idx,fruit)

# 0 apple
# 1 banana
# 2 cherry

'''

# 컬렉션 정리 p.299

# 함수와 스코프 p.238
# 서술형을 시험내기 너무너무너무 좋다 ,,, ? 
# [참고] https://jinaon.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%B2%94%EC%9C%84-Scope-%EC%A7%80%EC%97%AD%EB%B3%80%EC%88%98%EC%A0%84%EC%97%AD%EB%B3%80%EC%88%98-LEGB%EA%B7%9C%EC%B9%99


