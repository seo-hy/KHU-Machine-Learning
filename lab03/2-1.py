"""
2. [프로그램 3-2]를 sepal length 특징을 제외하고 3차원 특징 공간을 그리도록 수정하고 데이터
분포에 대한 분석을 제시하시오.
"""

# pip install plotly
# pip install pandas
import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x='petal_length', y='sepal_width',z='petal_width', color = 'species')
fig.show(renderer="browser")