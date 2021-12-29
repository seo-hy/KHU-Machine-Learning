"""
3. matplotlib를 사용한 차트 그리기
3) 다음의 x,y 리스트가 주어졌을 때 출력화면과 같이 나오도록 subplot을 사용하여 여러 그래프
를 하나의 화면에 그리는 코드를 작성하시오. (matplotlib만 사용)
"""
import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4,5,6,7])
y=np.array([[12,14,40],[20,25,43],[25,30,41],[22,20,35],[20,30,42],[25,30,35],[30,35,38]])
plt.figure(figsize=(15,3))
plt.suptitle('Amount of sales for 3 items')
plt.subplot(1,3,1)
plt.plot(x,y[:,0],'g*-',linewidth=1.5,markersize=8)
plt.xlabel('days')
plt.ylabel('amount (unit: Kg)')
plt.legend(['apple'])
plt.subplot(1,3,2)
plt.plot(x,y[:,1],'ro--',linewidth=1.5,markersize=8)
plt.xlabel('days')
plt.ylabel('amount (unit: Kg)')
plt.legend(['plum'])
plt.subplot(1,3,3)
plt.plot(x,y[:,2],'b+:',linewidth=1.5,markersize=8)
plt.xlabel('days')
plt.ylabel('amount (unit: Kg)')
plt.legend(['strawberry'])
plt.show()