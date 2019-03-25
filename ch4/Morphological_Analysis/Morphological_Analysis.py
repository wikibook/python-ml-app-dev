from konlpy.tag
import Okt2 
# Okt 객체 생성
okt = Okt()
# 형태소 분석
malist = okt.pos("아버지 가방에 들어가신다.", norm=True, stem=True) 
print(malist)