import pandas as pd

# Pandas로 CSV 파일 읽어 들이기 ---(*1)
df = pd.read_csv("tem10y.csv", encoding="utf-8")

# 날짜별 기온을 리스트에 넣기 ---(*2)
md = {}
for i, row in df.iterrows():
    m,  d, v = (int(row['월']), int(row['일']), float(row['기온']))
    key = str(m) + "/" + str(d)
    if not(key in md): md[key] = []
    md[key] += [v]

# 날짜별 평균 구하기 ---(*3)
avs = {}
for key in md:
    v = avs[key] = sum(md[key]) / len(md[key]) # ---(*4)
    print("{0} : {1}".format(key, v))


