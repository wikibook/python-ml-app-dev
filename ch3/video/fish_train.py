import cv2
import os, glob
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# 이미지 학습 크기와 경로 지정하기
image_size = (64, 32)
path = os.path.dirname(os.path.abspath(__file__))
path_fish = path + '/fish'
path_nofish = path + '/nofish'
x = [] # 이미지 데이터
y = [] # 레이블 데이터

# 이미지 데이터를 읽어 들이고 배열에 넣기 --- (*1)
def read_dir(path, label):
    files = glob.glob(path + "/*.jpg")
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, image_size)
        img_data = img.reshape(-1, ) # 1차원으로 전개하기
        x.append(img_data)
        y.append(label)

# 이미지 데이터 읽어 들이기
read_dir(path_nofish, 0)
read_dir(path_fish, 1)

# 데이터를 학습 전용과 테스트 전용으로 분리하기 --- (*2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 데이터 학습하기 --- (*3)
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

# 정답률 확인하기 --- (*4)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

# 데이터 저장하기 --- (*5)
joblib.dump(clf, 'fish.pkl')
