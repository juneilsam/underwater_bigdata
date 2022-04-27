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
import pyodbc
import os
import requests

# 데이터 DB 업로드
def DFToDB(df):
    try:
        # DB 접속정보
        conn = pyodbc.connect("UID=;PWD=;host=;port=;DSN=")
        curs = conn.cursor()

        # 데이터 프레임 행을 튜플로 변경
        rows = [tuple(i) for i in df.to_records(index=False)]
        up_rows = []
        for j in tqdm(df.index):
            r_1 = str(rows[j][0])
            r_2 = parse(rows[j][1])
            r_3 = float(rows[j][2])
            r_4 = float(rows[j][3])
            r_5 = float(rows[j][4])
            r_6 = float(rows[j][5])
            r_7 = str(rows[j][6])
            up_rows.append(tuple([r_1, r_2, r_3, r_4, r_5, r_6, r_7]))

        print("Upload started")
        # insert 쿼리
        curs.executemany("INSERT INTO TD304 \
                         (SN, WETHER_OBSRVT_CODE, FRST_REGIST_DT, DE_AMTPRCP_VALUE, \
                         AVRG_RLTIV_HD_RT, AVRG_ARCSR_VALUE, SUM_SSH_TIME_VALUE, MESURE_DE) \
                         VALUES (SEQ_TM_GD30704.NEXTVAL, :1, :2, :3, :4, :5, :6, :7)", up_rows) # 자동 시퀀스 입력
        conn.commit()
        curs.close()
        conn.close()
        print("DB Inserted")
        
    except Exception as e:
        print("데이터를 DB에 업로드하는 데 문제가 있습니다.\n", e)

# 테이블의 마지막 날짜 확인하기
def DBlast():
    # dsn 생성 후 DB연결
    conn = pyodbc.connect("UID=;PWD=;host=;port=;DSN=")

    curs = conn.cursor()
    
    # 밖 테이블 불러오기
    sql = f"SELECT FRST_REGIST_DT FROM TM_GD30704 WHERE SUM_SSH_TIME_VALUE != -1"

    # 밖에서 불러온 테이블의 마지막 시퀀스 넘버
    try:
        last_seq = pd.to_datetime(pd.read_sql(sql, conn).values.max())
        
    # 테이블이 비어있는 경우
    except:
        last_seq = None

    conn.commit()
    curs.close()
    conn.close()
    return last_seq

# 전체 관측소 지점코드
stdIDS = ['90', '93', '95', '98', '99', '100', '101', '102', '104', '105', '106', '108', '112', '114', '115', '212', '216', '217',
         '221', '226', '232', '235', '236', '238', '239', '243', '244', '245', '247', '248', '119', '121', '127', '129', '130',
         '131', '133', '135', '136', '137', '138', '140', '143', '146', '152', '155', '156', '159', '251', '252', '253', '254',
         '255', '257', '258', '259', '260', '261', '262', '263', '264', '266', '268', '271', '272', '273', '162', '165', '168',
         '169', '170', '172', '174', '177', '184', '185', '188', '189', '192', '201', '202', '203', '211', '276', '277', '278',
         '279', '281', '283', '284', '285', '288', '289', '294', '295']

# 임시 - 이전 날짜 모두 다운받기
print('weather_start')        
# 모든 날짜의 날씨 데이터 용량이 너무 크고, 백업을 해야하기 때문
former = pd.read_csv('26_기상자료전체.csv')

DBDate = DBlast()

if DBDate is None:
    s_date = str((pd.to_datetime(former.FRST_REGIST_DT.values.max()) + dt.timedelta(days=1)).date())
else:
    s_date = str((DBDate + dt.timedelta(days=1)).date())

# 어제 날짜를 string(object)로 변경
f_date = str((dt.datetime.now() - dt.timedelta(days=1)).date())

# 받아야 하는 기간
dtRange = [str(i.date()).replace('-', '') for i in pd.date_range(start=s_date, end=f_date, freq='D')]

temp = []
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
for stdid in tqdm(stdIDS):
    for Dt in dtRange:
        # API 접속정보
        params = {'serviceKey' : '',
                 'pageNo' : "1",
                 'numOfRows' : '24', 'dataType' : 'json', 'dataCd' : 'ASOS', 'dateCd' : 'HR',
                 'startDt' : f'{Dt}', 'startHh' : '00',
                 'endDt' : f'{Dt}', 'endHh' : '23',
                 'stnIds' : f'{stdid}'}

        response = requests.get(url, params=params)
        downD = json.loads(response.content)["response"]["body"]["items"]["item"]

        # 다운받은 데이터를 프레임에 저장
        temp.append(pd.json_normalize(downD))

if len(temp) != 0:
    temp_df = pd.concat(temp, axis=0, ignore_index=True)
    temp_df['tm'] = pd.to_datetime(temp_df['tm'])

    # numeric 타입으로 변환
    # 'stnId' : 지점코드, 'tm' : 일자, 'rn' : 강수량, 'hm' : 습도, 'pa' : 현지기압, 'ss' : 일조
    temp_df = temp_df[['tm', 'stnId', 'rn', 'hm', 'pa', 'ss']]
    temp_df[['rn', 'hm', 'pa', 'ss']] = temp_df[['rn', 'hm', 'pa', 'ss']].apply(pd.to_numeric, errors='coerce')

    # FRST_REGIST_DT : 일시, DE_AMTPRCP_VALUE : 관측소 지점, DE_AMTPRCP_VALUE : 강수량, AVRG_RLTIV_HD_RT : 습도,
    # AVRG_ARCSR_VALUE : 기압, SUM_SSH_TIME_VALUE:일조시간, MESURE_DE:측정일자 
    temp_df.columns = ['FRST_REGIST_DT', 'WETHER_OBSRVT_CODE', 'DE_AMTPRCP_VALUE', 'AVRG_RLTIV_HD_RT',
                       'AVRG_ARCSR_VALUE', 'SUM_SSH_TIME_VALUE']
    temp_df.set_index('FRST_REGIST_DT', inplace=True)

    final_df = pd.concat([temp_df.groupby(['WETHER_OBSRVT_CODE', temp_df.index.date])['DE_AMTPRCP_VALUE'].sum(),
               temp_df.groupby(['WETHER_OBSRVT_CODE', temp_df.index.date])['AVRG_RLTIV_HD_RT'].mean(),
               temp_df.groupby(['WETHER_OBSRVT_CODE', temp_df.index.date])['AVRG_ARCSR_VALUE'].mean(),
               temp_df.groupby(['WETHER_OBSRVT_CODE', temp_df.index.date])['SUM_SSH_TIME_VALUE'].sum()], axis=1).reset_index()
    del temp_df

    final_df.rename(columns={'level_1':'FRST_REGIST_DT'}, inplace=True)
    final_df.set_index(pd.to_datetime(final_df['FRST_REGIST_DT'])).sort_index(ascending=False, inplace=True, ignore_index=True)

    final_df['MESURE_DE'] = str(dt.datetime.now().date()).replace('-', '')
    final_df.fillna(0, inplace=True)
    final_df['FRST_REGIST_DT'] = final_df['FRST_REGIST_DT'].astype('str')
    DFToDB(final_df.copy())

    new_df = pd.concat([former, final_df], axis=0, ignore_index=True)
    del final_df
    new_df = new_df.set_index(pd.to_datetime(new_df['FRST_REGIST_DT'])).sort_index().reset_index(drop=True)
    new_df.to_csv('./26_기상자료전체.csv', header=True, index=False, encoding='utf-8')
    new_df.to_csv('./26_기상자료전체_back.csv', header=True, index=False, encoding='utf-8')

# 테이블이 비어있으면 기존 자료 업로드
elif (len(temp) == 0) & (DBDate is None):
    DFToDB(former.copy())

else:
    pass
    
print('weather_finished')

# 단기 예측 자료 다운
url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'

# 기준일자 설정
b_date = ''.join(filter(str.isdigit, str(dt.datetime.now().date())))

# 관측지 좌표 [x, y]
spots = {'211' : ['80', '138'], '101' : ['72', '139'], '114' : ['73', '124'], '119' : ['59', '114'],
         '95' : ['65', '238'], '288' : ['92', '82'], '289' : ['76', '80'], '279' : ['82', '92'],
         '278' : ['86', '102'], '235' : ['49', '104'], '285' : ['85', '82']}

# ['WETHER_OBSRVT_CODE', 'FRST_REGIST_DT', 'DE_AMTPRCP_VALUE', 'AVRG_RLTIV_HD_RT', 'AVRG_ARCSR_VALUE', 'SUM_SSH_TIME_VALUE']
new_df = pd.DataFrame(columns=['WETHER_OBSRVT_CODE', 'FRST_REGIST_DT', 'DE_AMTPRCP_VALUE', 'AVRG_RLTIV_HD_RT'])

for spot in tqdm(spots.keys()):
    X = spots[spot][0]
    Y = spots[spot][1]
    params = {'serviceKey' : '',
             'pageNo' : '1', 'numOfRows' : '1000', 'dataType' : 'json', 
             'base_date' : f'{b_date}',
             'base_time' : '0200',
             'nx' : X, 'ny' : Y}
    
    response = requests.get(url, params=params)
    
    aa = json.loads(response.content)["response"]["body"]["items"]["item"]

    temp_dict = {}
    
    for i in range(len(aa)):
        if (aa[i]['category'] == 'PCP'):
            temp_dict['WETHER_OBSRVT_CODE'] = spot
            temp_dict['FRST_REGIST_DT'] = aa[i]['fcstDate']
            if (aa[i]['fcstValue'] == '강수없음'):
                temp_dict['DE_AMTPRCP_VALUE'] = 0.0
            elif aa[i]['fcstValue'] == '30.0~50.0mm':
                temp_dict['DE_AMTPRCP_VALUE'] = 30.0
            elif aa[i]['fcstValue'] == '50.0mm 이상':
                temp_dict['DE_AMTPRCP_VALUE'] = 50.0
            else:
                temp_dict['DE_AMTPRCP_VALUE'] = float(aa[i]['fcstValue'][:-2])
        elif (aa[i]['category'] == 'REH'):
            temp_dict['AVRG_RLTIV_HD_RT'] = float(aa[i]['fcstValue'])      
            new_df = new_df.append(temp_dict, ignore_index=True)
            temp_dict = {}
            
final_df = pd.concat([new_df.groupby(['WETHER_OBSRVT_CODE', 'FRST_REGIST_DT'])['DE_AMTPRCP_VALUE'].sum(),
                      new_df.groupby(['WETHER_OBSRVT_CODE', 'FRST_REGIST_DT'])['AVRG_RLTIV_HD_RT'].mean()], axis=1).reset_index()

final_df['AVRG_ARCSR_VALUE'] = -1
final_df['SUM_SSH_TIME_VALUE'] = -1
final_df['MESURE_DE'] = b_date
final_df = final_df.set_index(pd.to_datetime(final_df['FRST_REGIST_DT'])).sort_index().reset_index(drop=True)

DFToDB(final_df.copy())
