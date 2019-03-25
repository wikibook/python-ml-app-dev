import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import glob

# Unicode 코드 포인트로 출현 빈도 판정하기
def count_codePoint(str):
    # Unicode 코드 포인트를 저장할 배열 준비하기
    counter = np.zeros(65535)

    for i in range(len(str)):
        # 각 문자를 Unicode 코드 포인트로 변환하기
        code_point = ord(str[i])
        if code_point > 65535 :
            continue
        # 출현 횟수 세기
        counter[code_point] += 1

    # 각 요소를 문자 수로 나눠 정규화하기
    counter = counter/len(str)
    return counter

# 학습 데이터 준비하기 --- (*1)
index = 0
x_train = []
y_train = []
for file in glob.glob('./train/*.txt'):
    # 언어 정보를 추출하고 레이블로 지정하기 --- (*2)
    y_train.append(file[8:10])
    
    # 파일 내부의 문자열을 모두 추출한 뒤 빈도 배열로 변환한 뒤 입력 데이터로 사용하기 --- (*3)
    file_str = ''
    for line in open(file, 'r'):
        file_str = file_str + line
    x_train.append(count_codePoint(file_str))

# 학습하기
clf = GaussianNB() 
clf.fit(x_train, y_train)

# 평가 데이터 준비하기 --- (*4)
index = 0
x_test = []
y_test = []
for file in glob.glob('./test/*.txt'):
    # 언어 정보를 추출하고 레이블로 지정하기
    y_test.append(file[7:9])
    
    # 파일 내부의 문자열을 모두 추출한 뒤 빈도 배열로 변환한 뒤 입력 데이터로 사용하기
    file_str = ''
    for line in open(file, 'r'):
        file_str = file_str + line
    x_test.append(count_codePoint(file_str)) 

# 평가하기
y_pred = clf.predict(x_test)
print(y_pred)
print("정답률 = " , accuracy_score(y_test, y_pred))
    