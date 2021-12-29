"""
3. matplotlib를 사용한 차트 그리기
1) 다음의 x,y 리스트가 주어졌을 때 출력화면과 같이 나오도록 실선 그래프와 막대 그래프르
matplotlib으로 그리는 코드를 작성하시오. (matplotlib만 사용)
"""
import matplotlib.pyplot as plt
x=[1,2,3,4,5,6,7] # 일주일
y=[12,20,25,22,20,25,30]
plt.plot(x,y)
plt.bar(x,y)
plt.show()
