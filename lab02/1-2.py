"""
1.numpy 라이브러리
2) numpy를 사용하여 3x2 행렬 a, 2x3 행렬 b의 곱셈을 수행하고 출력화면과 같이 나오도록 하는
코드를 작성하시오.
"""

import numpy as np
a = np.array([[4,2],[2,7],[-2,1]])
b = np.array([[1,-2,3],[5,0,2]])
c = np.matmul(a,b)
print(c)