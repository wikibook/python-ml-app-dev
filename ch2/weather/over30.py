import matplotlib.pyplot as plt
import pandas as pd
# 파일 읽어 들이기
df = pd.read_csv('tem10y.csv', encoding="utf-8")
# 온도가 30도를 넘는 데이터 확인하기 ---(*1)
hot_bool = (df["기온"] > 30)
# 데이터 추출하기 ---(*2)
hot = df[hot_bool]
# 연별로 세기 ---(*3)
cnt = hot.groupby(["연"])["연"].count()
# 출력하기
print(cnt)
cnt.plot()
plt.savefig("tem-over30.png")
plt.show()
