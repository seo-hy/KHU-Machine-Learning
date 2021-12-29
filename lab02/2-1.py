"""
2.리스트, 튜플, 딕셔너리, 셋 
1) 리스트 a, 튜플 b, 딕셔너리 c, 셋 d를 출력화면과 같이 나오도록 코드를 작성하시오.
"""
a=[5,2,3,8,2]
b=(5,2,3,8,2)
c={1: 'book', 5: 'notebook', 3: 'pencil', -3: 'eraser', 'as': 120, 12.2: 50}
d=set([5,2,3,8,2])
print(type(a), type(b), type(c), type(d))
print(a,b,c,d,sep='\n')
