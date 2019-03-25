# ETL1 파일 읽어 들이기
import struct
from PIL import Image, ImageEnhance
import glob, os

# 출력 디렉터리
outdir = "png-etl1/"
if not os.path.exists(outdir): os.mkdir(outdir)

# ETL1 디렉터리 아래의 파일 처리하기 --- (*1)
files = glob.glob("ETL1/*")
for fname in files:
    if fname == "ETL1/ETL1INFO": continue # 정보 파일 무시하기
    print(fname)
    # ETL1 데이터 파일 열기 --- (*2)
    f = open(fname, 'rb')
    f.seek(0)
    while True:
        # 메타데이터와 이미지 데이터 조합을 하나씩 읽어 들이기 --- (*3)
        s = f.read(2052)
        if not s: break
        # 바이너리 데이터이므로 Python에서 사용할 수 있는 형태로 읽어 들이기 --- (*4)
        r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
        code_ascii = r[1]
        code_jis = r[3]
        # 이미지 데이터로 추출하기 --- (*5)
        iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
        iP = iF.convert('L')
        # 이미지를 선명하게 해서 저장하기
        dir = outdir + "/" + str(code_jis)
        if not os.path.exists(dir): os.mkdir(dir)
        fn = "{0:02x}-{1:02x}{2:04x}.png".format(code_jis, r[0], r[2])
        fullpath = dir + "/" + fn
        #if os.path.exists(fullpath): continue
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(16)
        iE.save(fullpath, 'PNG')
print("ok")
