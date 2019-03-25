# 이미지 파일을 읽어 들이고 Numpy 형식으로 변환하기
import numpy as np
from PIL import Image
import os, glob, random

outfile = "image/photos.npz" # 저장할 파일 이름
max_photo = 100 # 사용할 장 수
photo_size = 32 # 이미지 크기
X = [] # 이미지 데이터
y = [] # 레이블 데이터

# path 내부의 이미지를 최대 max_photo 만큼만 읽기 --- (*1)
def glob_files(path, label, max_photo):
    files = glob.glob(path + "/*.jpg")
    random.shuffle(files)
    # 파일 처리하기
    num = 0
    for f in files:
        if num >= max_photo: break
        num += 1
        # 이미지 파일 읽어 들이기
        img = Image.open(f)
        img = img.convert("RGB") # 색공간 변환하기
        img = img.resize((photo_size, photo_size))
        X.append(image_to_data(img))
        y.append(label)
        # 각도를 조금씩 변경한 이미지 추가하기 --- (*2)
        for angle in range(-20, 21, 5):
            # 각도 변경
            if angle != 0:
                img_angle = img.rotate(angle)
                X.append(image_to_data(img_angle))
                y.append(label)
            # 반전
            img_r = img_angle.transpose(Image.FLIP_LEFT_RIGHT)
            X.append(image_to_data(img_r))
            y.append(label)

def image_to_data(img): # 이미지 데이터 정규화하기
    data = np.asarray(img)
    data = data / 256
    data = data.reshape(photo_size, photo_size, 3)
    return data

# 디렉터리 읽어 들이기 --- (*3)
glob_files("./image/sushi", 0, max_photo)
glob_files("./image/salad", 1, max_photo)
glob_files("./image/tofu", 2, max_photo)

# 파일로 저장하기 --- (*4)
X = np.array(X, dtype=np.float32)
np.savez(outfile, X=X, y=y)
print("저장했습니다:" + outfile)
