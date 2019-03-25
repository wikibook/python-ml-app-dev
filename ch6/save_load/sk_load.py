from sklearn import datasets, svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

# 이전에 저장한 학습된 데이터 읽어 들이기
clf = joblib.load('iris.pkl')

# 붓꽃 데이터 읽어 들이기
iris = datasets.load_iris()
# 예측하기
pre = clf.predict(iris.data)
# 정답률 확인하기
print(accuracy_score(iris.target, pre))

