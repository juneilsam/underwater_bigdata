# -*- coding: utf-8 -*-
import numpy as np # 넘파이
import pandas as pd
import json
from math import sqrt # math
from tqdm import tqdm
import time
import datetime as dt
from dateutil.parser import parse
import cx_Oracle as cx
import os
import requests

# tm_gd60101의 spot_id, PM_GD60201의 quilty_id, 업로드 RESULT_ID, 기상관측망(TM_GD30704의 기후관측소코드)
spot_ids = {'02' : [219, 1, 14789, 211], # 양구방산
            '04' : [94, 1, 14995, 101], # 화천화천
            '06' : [225, 1, 16218, 114], # 양평양동
            '07' : [32, 1, 14717, 119], # 평택안중
            '08' : [383, 1, 14700, 95], # 포천대회산
            '10' : [310, 1, 14687, 288], # 밀양가곡
            "11" : [191, 1, 14730, 289], # 산청산청
            "12" : [195, 1, 16189, 279], # 성주벽진
            '14' : [253, 1, 14764, 278], # 의성안계
            '22' : [38, 1, 14749, 235], # 태안안면
            '24' : [152, 1, 16197, 285], # 의령낙서(충적)
            '25' : [152, 2, 14758, 285] } # 의령낙서(암반)

# 카피하는 거
# 밖 - 중간 - 내부
def DFcopy():
    # cx 위치
    LOCATION = r"C:\Users\junei\.conda\envs\ll\instantclient_21_3"
    
    # 환경변수 등록
    os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록
    
    # 밖 connect
    conn_from = cx.connect()
    
    # 내부 connect
    conn_to = cx.connect()
    
    # 내부 cursor
    curs_to = conn_to.cursor()
    
    # 중간값 확인 - 지난 업로드의 최종 시퀀스 넘버
    sn_sql = f"SELECT LAST_CNTC_NO FROM TM_GD70004 WHERE CNTC_IEM_CODE = 03"
    start_SN = pd.read_sql(sn_sql, conn_from).iat[0, 0]
    
    # 밖 테이블 불러오기 - 위에서 확인한 중간값 이후의 시퀀스 넘버부터 불러옴
    sql_from = f"SELECT WETHER_OBSRVT_CODE, FRST_REGIST_DT, DE_AMTPRCP_VALUE, \
    AVRG_RLTIV_HD_RT, AVRG_ARCSR_VALUE, SUM_SSH_TIME_VALUE, MESURE_DE FROM TM_GD30704 WHERE SN >= {start_SN}"
    
    # 가장 최근에 API로 받은 마지막 시퀀스 넘버 - 쿼리
    sql_to = "SELECT SN FROM TM_GD30704"
    
    # 밖 테이블
    result = pd.read_sql(sql_from, conn_from)
    result['FRST_REGIST_DT'] = result['FRST_REGIST_DT'].astype(str)
    
    # 밖 테이블에서 불러온 거 내부에 올리기
    rows = [tuple(i) for i in result.to_records(index=False)]
    up_rows = []
    
    for j in tqdm(result.index):
        r_1 = str(rows[j][0])
        r_2 = parse(rows[j][1])
        r_3 = float(rows[j][2])
        r_4 = float(rows[j][3])
        r_5 = float(rows[j][4])
        r_6 = float(rows[j][5])
        r_7 = str(rows[j][6])
        up_rows.append(tuple([r_1, r_2, r_3, r_4, r_5, r_6, r_7]))

    # insert 쿼리
    curs_to.executemany("INSERT INTO TM_GD30705 \
                     (WETHER_OBSRVT_CODE, FRST_REGIST_DT, DE_AMTPRCP_VALUE, \
                     AVRG_RLTIV_HD_RT, AVRG_ARCSR_VALUE, SUM_SSH_TIME_VALUE, MESURE_DE) \
                     VALUES (:1, :2, :3, :4, :5, :6, :7)", up_rows)
    
    # 연결 테이블의 시퀀스 넘버 업데이트
    sql_to = "SELECT SN FROM TM_GD30704"
    last_SN = int(pd.read_sql(sql_to, conn_to).values.max()) + 1
    curs_to.execute(f"UPDATE TM_GD70004 SET LAST_CNTC_NO = {last_SN} WHERE CNTC_IEM_CODE = 03")
    
    conn_to.commit()
    curs_to.close()
    conn_to.close()
    print("DB Inserted")
    # 끝^^

# 최초 업로드
# LOCATION = r"C:\Users\junei\.conda\envs\ll\instantclient_21_3"
# # 환경변수 등록
# os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록

# conn = cx.connect()

# curs = conn.cursor()

# # up_rows = [('0', '2022/04/18', '03')]
# curs.executemany("INSERT INTO TM_GD70004 (LAST_CNTC_NO, FRST_REGIST_DT, CNTC_IEM_CODE) VALUES (:1, :2, :3)", [('0', '2022/04/18', '03')])
                 
# conn.commit()
# curs.close()
# conn.close()
