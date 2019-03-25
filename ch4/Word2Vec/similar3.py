from konlpy.tag import Okt
from gensim.models import word2vec
# Word2Vec 모델 읽어 들이고 형태소 분석 준비하기
model = word2vec.Word2Vec.load("./wiki.model") 
okt = Okt()
def print_emargency(text): print(text)
  # 전달된 문장을 형태소 분석하기
  node = okt.pos(text, norm=True, stem=True)
  for word, form in node:
    # 필요한 형태소만 추출하기
    if form == 'Noun' or form == 'Verb' or form == 'Adjective' or form == 'Adverb':
      # 급하다와 비슷한 단어
      print("-", word, ":", model.wv.similarity(word, '급하다'))

print_emargency("컴퓨터에 문제가 생겼어요. 빨리 해결해야 하는 문제가 있어서 지원 요청합니다.")
print_emargency("사용 방법을 잘 모르겠습니다.")