# 이미지들을 NumPy 형식으로 변환하기
import numpy as np
from PIL import Image
import os, glob, random

outfile = "image/photos.npz" # 저장할 파일 이름
max_photo = 100 # 사용할 장 수
photo_size = 32 # 이미지 크기
x = [] # 이미지 데이터
y = [] # 레이블 데이터

def main():
    # 디렉터리 읽어 들이기 --- (*1)
    glob_files("./image/sushi", 0)
    glob_files("./image/salad", 1)
    glob_files("./image/tofu", 2)
    # 파일로 저장하기 --- (*2)
    np.savez(outfile, x=x, y=y)
    print("저장했습니다:" + outfile, len(x))

# path 내부의 이미지 읽어 들이기 --- (*3)
def glob_files(path, label):
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
        img = img.resize((photo_size, photo_size)) # 크기 변경하기
        img = np.asarray(img)
        x.append(img)
        y.append(label)

if __name__ == '__main__':
    main()
