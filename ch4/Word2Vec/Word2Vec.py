from gensim.models import word2vec
# 코퍼스 읽어 들이기 --- (※ 1)
sentences = word2vec.Text8Corpus('./wiki_wakati.txt')
# 모델 만들기 --- (※ 2)
model = word2vec.Word2Vec(sentences, sg=1, size=100, window=5)
# 모델 저장하기 --- (※ 3)
model.save("./wiki.model")
