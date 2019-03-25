import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# 저장 경로와 이미지 크기 지정하기 --- (*1)
out_dir = "./png-etl1" # 이미지 데이터가 있는 디렉터리
im_size = 25 # 이미지 크기
save_file = out_dir + "/katakana.pickle" # 저장 경로
plt.figure(figsize=(9, 17)) # 화면에 출력할 이미지 크기 지정하기

# 이미지 추출하기 --- (*2)
kanadir = list(range(177, 220+1))
kanadir.append(166) # 일본어 요(ヲ)를 나타내는 글자
kanadir.append(221) # 일본어 응(ン)을 나타내는 글자
result = []
for i, code in enumerate(kanadir):
    img_dir = out_dir + "/" + str(code)
    fs = glob.glob(img_dir + "/*")
    print("dir=",  img_dir)
    # 이미지를 읽어 들이고, 그레이스케일로 변환하고, 크기 변경하기 --- (*3)
    for j, f in enumerate(fs):
        img = cv2.imread(f)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img_gray, (im_size, im_size))
        result.append([i, img])
        # Jupyter Notebook에 이미지 출력하기
        if j == 3:
            plt.subplot(11, 5, i + 1)
            plt.axis("off")
            plt.title(str(i))
            plt.imshow(img, cmap='gray')
# 레이블 데이터와 이미지 데이터 저장하기 --- (*4)
pickle.dump(result, open(save_file, "wb"))
plt.show()
print("ok")
