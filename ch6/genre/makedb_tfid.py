import os, glob, pickle
import tfidf

# 변수 초기화
y = []
x = []

# 디렉터리 내부의 파일 목록 전체에 대해 처리하기 --- (*1)
def read_files(path, label):
    print("read_files=", path)
    files = glob.glob(path + "/*.txt")
    for f in files:
        if os.path.basename(f) == 'LICENSE.txt': continue
        tfidf.add_file(f)
        y.append(label)

# 기사를 넣은 디렉터리 읽어 들이기 --- ( ※ 2)
read_files('text/100', 0)
read_files('text/101', 1)
read_files('text/103', 2)
read_files('text/105', 3)


# TF-IDF 벡터로 변환하기 --- (*3)
x = tfidf.calc_files()

# 저장하기 --- (*4)
pickle.dump([y, x], open('text/genre.pickle', 'wb'))
tfidf.save_dic('text/genre-tdidf.dic')
print('ok')
