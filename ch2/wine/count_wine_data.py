import matplotlib.pyplot as plt
import pandas as pd

# 와인 데이터 읽어 들이기
wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

# 품질 데이터별로 그룹을 나누고 수 세어보기
count_data = wine.groupby('quality')["quality"].count()
print(count_data)

# 수를 그래프로 그리기
count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()

