'''
class Person:
    blood_color='red'

    def __init__(self,name):
        self.name=name
    
    def singing(self):
        print(f'{self.name}가 노래합니다.')

singer1=Person('iu')
'''

'''
class Person:
    cnt=0

    def __init__(self,name):
        self.name=name
        Person.cnt+=1

person1=Person('iu')
person2=Person('BTS')
print(Person.cnt)
print(person1.cnt)

'''

# p.30 
# 인스턴스를 통해 클래스 변수 값을 변경하면, 클래스 변수를 공유하고 있는 다른 인스턴스들에게 같이 변경되는거 아님 ,,?

'''
class Circle:
    pi=3.14

    def __init__(self,r):
        self.r=r

c1=Circle(5)
c2=Circle(10)

Circle.pi=5
print(Circle.pi)    # 5
print(c2.pi)    # 5

c2.pi=3  # ** 인스턴스에서 클래스 메서드를 접근하게 되면, 인스턴스의 별도의 공간에 클래스 변수 값을 할당함, 지양해야함 
print(Circle.pi)    # 5
print(c2.pi)    # 3
print(c1.pi)    # 5

Circle.pi=10
print(c2.pi)  #
print(c1.pi)  # 10

'''


# 인스턴스, 클래스, 정적 메서드
'''
1. 인스턴스 메서드
- 클래스로부터 생성된 각 인스턴스에서 호출할 수 있는 메서드
- 인스턴스의 상태를 조작하거나 동작을 수행
- 반드시 첫 번째 매개변수로 인스터스 자신(self)을 전달받음
- 생성자 메서드도 포함됨

'hello'.upper() >> str.upper('hello')
- 'hello'라는 문자열 객체가 단순히 어딘가의 함수로 들어가는 인자로 활용되는 것이 아님
- 객체 스스로 메서드를 호출하여 코드를 동작하는 객체 지향적인 표현이라 볼 수 있음


2. 클래스 메서드
- 클래스가 호출하는 메서드
- 클래스 변수를 조작하거나 클래스 레벨의 동작을 수행
- 호출 시, 첫번째 인자로 해당 메서드를 호출하는 클래스(cls)가 전달됨


3. 정적 메서드
- 클래스와 인스턴스와 상관없이 독립적으로 동작하는 메서드
- 주로 클래스와 관련이 있지만, 인스턴스와 상호작용이 필요하지 않은 경우에 사용
- 호출 시 필수적으로 작성해야 할 매개변수가 있음
'''

##############################
# 상속

# 1. 단일상속
'''
class Person:
    def __init__(self,name,age):
        self.name=name
        self.age=age

    def talk(self): # 메서드 재사용
        print(f'반갑습니다. {self.name}입니다')

class Professor(Person):
    def __init__(self,name,age,department):
        self.name=name
        self.age=age
        self.department=department

class Student(Person):
    def __init__(self,name,age,qpa):
        self.name=name
        self.age=age
        self.qpa=qpa

p1=Professor('박교수',49,'컴공')
s1=Student('김학생',20,3.5)

p1.talk()
s1.talk()

'''

# 2. 다중상속
'''
- 둘 이상의 상위 클래스로부터 여러 행동이나 특징을 상속받을 수 있는 것
- 상속받은 모든 클래스의 요소를 활용 가능함
- 중복된 속성이나 메서드가 있는 경우 상속 순서에 의해 결정됨
'''

'''
class Person:
    def __inti__(self,name):
        self.name=name
    
    def greeting(self):
        return f'안녕, {self.name}'

class Mom(Person):
    gene='XX'

    def swim(self):
        return '엄마가 수영'

class Dad(Person):
    gene='XY'

    def walk(self):
        return '아빠가 걷기'

class FirstChild(Dad,Mom):
    def swim(self):
        return '첫째가 수영'
    
    def cry(self):
        return '첫째가 응애'

baby1=FirstChild('아가')
print(baby1.cry())  # 첫째가 응애
print(baby1.swim()) # 첫째가 수영
print(baby1.walk()) # 아빠가 걷기
print(baby1.gene)   # XY

'''

# MRO 알고리즘 - 다중 상속

'''
MRO 필요성

- 부모 클래스들이 여러 번 엑세스 되지 않도록,
- 각 클래스에서 지정된 왼쪽에서 오른쪽으로 가는 순서를 보존하고,
- 각 부모를 오직 한번만 호출하고,
- 부모들의 우선순위에 영향을 주지 않으면서 서브 클래스를 만드는 단조적인 구조 형성

>> 신뢰성, 확장성 있는 클래스 설계에 도움
>> 클래스 간의 메서드 호출 순서가 예측 가능하게 유지되며, 코드의 재사용성과 유지보수성 향상

'''

'''
super 필요성

1. 단일 상속 구조
- 명시적으로 이름을 지정하지 않아도, 부모 클래스 참조 가능
- 클래스 이름이 변경되거나 부모 클래스가 교체되어도 코드 수정이 더 적게 필요

2. 다중 상속 구조
- MRO에 따른 메서드 호출
- 복잡한 다중 상속 구조에서 발생할 수 있는 문제 방지

'''
'''
class ParentA:
    def __init__(self):
        self.val_a='ParentA'
    
    def show_val(self):
        print(f'Value from ParentA: {self.val_a}')
    

class ParentB:
    def __init__(self):
        self.val_b='ParentB'
    
    def show_val(self):
        print(f'Value from ParentB: {self.val_b}')
    
class Child(ParentA,ParentB):
    def __init__(self):
        super().__init__()  # ParentA 클래스의 __init__ 메서드 호출
        self.val_c='Child'
    
    def show_val(self):
        super().show_val()  # ParentA 클래스의 show_val 메서드
        print(f'Value from Child: {self.val_c}')

'''

# 매직 메서드
'''
class Person:
    def __init__(self,r):
        self.r=r

    def __str__(self):
        return f'원의 반지름: {self.r}'


p1=Person(1)
print(p1)

'''

# p.87 제너레이터 ~