# Flickr로 사진 검색해서 다운로드하기
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# AP 키 지정하기--- ( ※ 1)
key = "<자신의 것을 입력해주세요>"
secret = "<자신의 것을 입력해주세요>"
wait_time = 1 # 대기 시간(초)

# 키워드와 디렉터리 이름 지정해서 다운로드하기 --- ( ※ 2)
def main():
    go_download('초밥', 'sushi')
    go_download('샐러드', 'salad')
    go_download('마파두부', 'tofu')

# Flickr API로 사진 검색하기 --- (*3)
def go_download(keyword, dir):
    # 저장 경로 지정하기
    savedir = "./image/" + dir
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    # API를 사용해서 다운로드하기 --- (*4)
    flickr = FlickrAPI(key, secret, format='parsed-json')
    res = flickr.photos.search(
      text = keyword,     # 키워드
      per_page = 300,     # 검색할 개수
      media = 'photos',   # 사진 검색
      sort = "relevance", # 키워드 관련도 순서
      safe_search = 1,    # 안전 검색
      extras = 'url_q, license')
    # 결과 확인하기
    photos = res['photos']
    pprint(photos)
    try:
      # 1장씩 다운로드하기 --- (*5)
      for i, photo in enumerate(photos['photo']):
        url_q = photo['url_q']
        filepath = savedir + '/' + photo['id'] + '.jpg'
        if os.path.exists(filepath): continue
        print(str(i + 1) + ":download=", url_q)
        urlretrieve(url_q, filepath)
        time.sleep(wait_time)
    except:
      import traceback
      traceback.print_exc()

if __name__ == '__main__':
    main()
