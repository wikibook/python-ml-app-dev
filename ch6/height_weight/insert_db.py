import sqlite3
import random

dbpath = "./hw.sqlite3"

def insert_db(conn):
    # 더미 데이터 만들기 --- (*1)
    height = random.randint(130, 180)
    weight = random.randint(30, 100)
    # 더미 데이터를 기반으로 체형 데이터 생성하기 --- (*2)
    type_no = 1
    bmi = weight / (height / 100) ** 2
    if bmi < 18.5:
        type_no = 0
    elif bmi < 25:
        type_no = 1
    elif bmi < 30:
        type_no = 2
    elif bmi < 35:
        type_no = 3
    elif bmi < 40:
        type_no = 4
    else:
        type_no = 5
    # 데이터베이스에 저장하기 --- (*3)
    sql = '''
      INSERT INTO person (height, weight, typeNo) 
      VALUES (?,?,?)
    '''
    values = (height,weight, type_no)
    print(values)
    conn.executemany(sql,[values])

# 100개의 데이터 삽입하기
with sqlite3.connect(dbpath) as conn:
    # 데이터 100개 삽입하기 --- (*4)
    for i in range(100):
        insert_db(conn)
    # 확인하기 --- (*5)
    c = conn.execute('SELECT count(*) FROM person')
    cnt = c.fetchone()
    print(cnt[0])
