import pandas as pd
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold
import warnings
from sklearn.model_selection import cross_val_score

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기 
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# classifier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

# K-분할 크로스 밸리데이션 전용 객체 --- (*1)
kfold_cv = KFold(n_splits=5, shuffle=True)

for(name, algorithm) in allAlgorithms:
    if name == 'ClassifierChain' : continue
        elif name == 'CheckingClassifier' : continue
        elif name == 'MultiOutputClassifier' : continue
        elif name == 'OneVsOneClassifier' : continue
        elif name == 'OneVsRestClassifier' : continue
        elif name == 'OutputCodeClassifier' : continue
        elif name == 'StackingClassifier' : continue
        elif name == 'VotingClassifier' : continue
            
    # 각 알고리즘 객체 생성하기
    clf = algorithm()

    # score 메서드를 가진 클래스를 대상으로 하기--- (*2)
    if hasattr(clf,"score"):
        
        # 크로스 밸리데이션--- (*3)
        scores = cross_val_score(clf, x, y, cv=kfold_cv)
        print(name,"의 정답률=")
        print(scores)
