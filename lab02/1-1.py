"""
1.numpy 라이브러리
1) numpy를 사용하여 1차원 배열 a, 2차원 배열 b, 3차원 배열 c를 만들고 출력화면과 같이 나오
도록 하는 코드를 작성하시오.
"""

#pip install nunmpy
import numpy as np
a = np.array([12, 8, 20, 17,15])
print(a)
print(a.shape)

b = np.array([[12, 3, 4],[1,4,5]])
print(b)
print(b.shape)

c = np.array([[[1,3,0,1],[1,1,4,2],[3,3,4,1]],[[2,1,2,1],[1,0,1,0],[1,5,6,2]]])
print(c)
print(c.shape)