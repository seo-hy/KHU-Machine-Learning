"""
2.리스트, 튜플, 딕셔너리, 셋
3) 주어진 for 문을, 코드 한 줄만 사용하여 출력화면과 같이 나오도록 x2, x3, x4 리스트를 만들도
록 코드를 작성하시오.
"""
x2=[i*i for i in range(10)]
print(x2)

x3=[[i,j*2] for i in [10,2,3,1] for j in [2,4]]
print(x3)

x4=[[i,j*2] for i in [10,2,3,1] for j in [2,4] if i!=j]
print(x4)
