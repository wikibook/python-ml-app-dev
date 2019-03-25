from sklearn import datasets, svm
from sklearn.externals import joblib

# 붓꽃 데이터 읽어 들이기
iris = datasets.load_iris()

# 데이터 학습하기
clf = svm.SVC()
clf.fit(iris.data, iris.target)

# 학습한 데이터 저장하기
joblib.dump(clf, 'iris.pkl', compress=True)

