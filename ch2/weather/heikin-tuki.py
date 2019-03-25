import matplotlib.pyplot as plt
import pandas as pd
# CSV 파일 읽어 들이기 ---(*1)
df = pd.read_csv("tem10y.csv", encoding="utf-8")
# 월별 평균 구하기 ---(*2)
g = df.groupby(['월'])['기온']
gg = g.sum() / g.count()
# 결과 출력하기 ---(*3)
print(gg)
gg.plot()
plt.savefig("tem-month-avg.png")
plt.show()

