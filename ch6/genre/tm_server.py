import json
import flask
from flask import request
import my_text

# 포트 번호 --- (*1)
TM_PORT_NO = 8085
# HTTP 서버 실행하기
app = flask.Flask(__name__)
print("http://localhost:" + str(TM_PORT_NO))

# 루트에 접근할 경우 --- (*2)
@app.route('/', methods=['GET'])
def index():
    with open("index.html", "rb") as f:
        return f.read()

# /api에 접근할 경우
@app.route('/api', methods=['GET'])
def api():
    # URL 매개 변수 추출하기 --- (*3)
    q = request.args.get('q', '')
    if q == '':
      return '{"label": "내용을 입력해주세요", "per":0}'
    print("q=", q)
    # 텍스트 카테고리 판별하기 --- (*4)
    label, per, no = my_text.check_genre(q)
    # 결과를 JSON으로 출력하기
    return json.dumps({
      "label": label, 
      "per": per,
      "genre_no": no
    })
    
if __name__ == '__main__':
    # 서버 실행하기
    app.run(debug=False, port=TM_PORT_NO)

