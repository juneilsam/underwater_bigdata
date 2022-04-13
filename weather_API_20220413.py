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

print('weather_start')

# 데이터 DB 업로드
def DFToDB(df):
    try:
        # cx_oracle 경로 
        LOCATION = r".\instantclient_21_3"
        os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록

        # DB 접속정보
        conn = cx.connect(user="", password="", \
                dsn=cx.makedsn(host="", port="", service_name=""))
        curs = conn.cursor()

        # 데이터 프레임 행을 튜플로 변경
        rows = [tuple(i) for i in df.to_records(index=False)]
        up_rows = []
        for j in tqdm(df.index):
            r_1 = str(rows[j][0])
            r_2 = parse(dt.datetime.now())
            r_3 = float(rows[j][2])
            r_4 = float(rows[j][3])
            r_5 = float(rows[j][4])
            r_6 = float(rows[j][5])
            r_7 = parse(rows[j][1])
            up_rows.append(tuple([r_1, r_2, r_3, r_4, r_5, r_6, r_7]))

        # insert 쿼리
        curs.executemany("INSERT INTO TM_GD30704 \
                         (WETHER_OBSRVT_CODE, FRST_REGIST_DT, DE_AMTPRCP_VALUE, \
                         AVRG_RLTIV_HD_RT, AVRG_ARCSR_VALUE, SUM_SSH_TIME_VALUE, 날짜) \
                         VALUES (:1, :2, :3, :4, :5, :6, :7)", up_rows)
        conn.commit()
        curs.close()
        conn.close()
        print("DB Inserted")
        
    except Exception as e:
        print("데이터를 DB에 업로드하는 데 문제가 있습니다.\n", e)
        
# 전체 관측소 지점코드
stdIDS = ['90', '93', '95', '98', '99', '100', '101', '102', '104', '105', '106', '108', '112', '114', '115', '212', '216', '217',
         '221', '226', '232', '235', '236', '238', '239', '243', '244', '245', '247', '248', '119', '121', '127', '129', '130',
         '131', '133', '135', '136', '137', '138', '140', '143', '146', '152', '155', '156', '159', '251', '252', '253', '254',
         '255', '257', '258', '259', '260', '261', '262', '263', '264', '266', '268', '271', '272', '273', '162', '165', '168',
         '169', '170', '172', '174', '177', '184', '185', '188', '189', '192', '201', '202', '203', '211', '276', '277', '278',
         '279', '281', '283', '284', '285', '288', '289', '294', '295']

# 어제자
# 'stnId' : 지점코드, 'tm' : 일자, 'rn' : 강수량, 'hm' : 습도, 'pa' : 현지기압, 'ss' : 일조
new_df = pd.DataFrame(columns=['stnId', 'tm', 'rn', 'hm', 'pa', 'ss'])

url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
# 어제 날짜를 string(object)로 변경
s_date = ''.join(filter(str.isdigit, str((dt.datetime.now() - dt.timedelta(days=1)).date())))

for stdid in tqdm(stdIDS):
    # API 접속정보
    params = {'serviceKey' : '',
             'pageNo' : "1",
             'numOfRows' : '24', 'dataType' : 'json', 'dataCd' : 'ASOS', 'dateCd' : 'HR',
             'startDt' : f'{s_date}', 'startHh' : '00',
             'endDt' : f'{s_date}', 'endHh' : '23',
             'stnIds' : f'{stdid}'}
    
    response = requests.get(url, params=params)
    downD = json.loads(response.content)["response"]["body"]["items"]["item"]

    # 다운받은 데이터를 프레임에 저장
    for j in range(len(downD)):
        temp_dict = {}
        temp_list = []
        temp_dict['stnId'] = downD[j]['stnId']
        temp_dict['tm'] = downD[j]['tm']
        if (downD[j]['rn'] == ''):
            temp_dict['rn'] = 0.0
        else:
            temp_dict['rn'] = downD[j]['rn']
        temp_dict['hm'] = downD[j]['hm']
        temp_dict['pa'] = downD[j]['pa']
        temp_dict['ss'] = downD[j]['ss']
        new_df = new_df.append(temp_dict, ignore_index=True)

new_df['tm'] = pd.to_datetime(new_df['tm'])
# numeric 타입으로 변환
new_df[['rn', 'hm', 'pa', 'ss']] = new_df[['rn', 'hm', 'pa', 'ss']].apply(pd.to_numeric, errors='coerce')

# 데이터 일 변환
# 'stnId' : 지점코드, 'tm' : 일자, 'rn' : 강수량, 'hm' : 습도, 'pa' : 현지기압, 'ss' : 일조
temp = pd.DataFrame(columns=['stnId', 'tm', 'rn', 'hm', 'pa', 'ss', 'work_date'])
king_dict = {}
# 시간 단위 데이터를 일변환
for e,f in new_df.groupby([new_df['stnId'], new_df['tm'].dt.date]):
    king_dict['stnId'] = e[0]
    king_dict['tm'] = e[1]
    # 일총합
    king_dict['rn'] = sum(f['rn'])
    # 일 평균
    king_dict['hm'] = f['hm'].mean()
    king_dict['pa'] = f['pa'].mean()
    king_dict['ss'] = f['ss'].mean()
    temp = temp.append(king_dict, ignore_index=True)

# 작업일자
temp['work_date'] = dt.datetime.now()
temp.columns = ['지점', '일시', '일강수량(mm)', '평균 상대습도(%)', '평균 현지기압(hPa)', '합계 일조시간(hr)', '작업일자']
temp.sort_values(['지점', '일시'], ascending=False, inplace=True)
temp1 = temp.copy()
print('finished')

# DB 업로드
DFToDB(temp.copy())
# 어제 날짜 거까지 다 받은 다음 합쳐서 백업본도 따로 저장하기
former = pd.read_csv('./26_기상자료전체.csv')
pd.concat([former, temp1], axis=0).to_csv('./26_기상자료전체.csv', header=True, index=False, encoding='utf-8')
# # 백업용
pd.concat([former, temp1], axis=0).to_csv('./26_기상자료전체.csv', header=True, index=False, encoding='utf-8')


# 단기 예측 자료 다운
url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'

# 기준일자 설정
b_date = ''.join(filter(str.isdigit, str(dt.datetime.now().date())))

# 관측지 좌표 [x, y]
spots = {'02' : ['80', '138'], '04' : ['72', '139'], '06' : ['73', '124'], '07' : ['59', '114'], '08' : ['65', '238'], 
         '10' : ['92', '82'], '11' : ['76', '80'], '12' : ['82', '92'], '14' : ['86', '102'], '22' : ['49', '104'], 
         '24' : ['85', '82'], '25' : ['85', '82']}


new_df = pd.DataFrame(columns=['지점', '일시', '일강수량(mm)', '평균 상대습도(%)'])

for spot in tqdm(spots.keys()):
    X = spots[spot][0]
    Y = spots[spot][1]
    params = {'serviceKey' : '',
             'pageNo' : '1', 'numOfRows' : '1000', 'dataType' : 'json', 
             'base_date' : b_date,
             'base_time' : '0200',
             'nx' : X, 'ny' : Y}
    
    response = requests.get(url, params=params)
    
    aa = json.loads(response.content)["response"]["body"]["items"]["item"]

    temp_dict = {}
    
    for i in range(len(aa)):
        if (aa[i]['category'] == 'PCP'):
            temp_dict['지점'] = spot
            temp_dict['일시'] = aa[i]['fcstDate']
            if (aa[i]['fcstValue'] == '강수없음'):
                temp_dict['일강수량(mm)'] = 0.0
            elif aa[i]['fcstValue'] == '30.0~50.0mm':
                temp_dict['일강수량(mm)'] = 30.0
            elif aa[i]['fcstValue'] == '50.0mm 이상':
                temp_dict['일강수량(mm)'] = 50.0
            else:
                temp_dict['일강수량(mm)'] = float(aa[i]['fcstValue'][:-2])
        elif (aa[i]['category'] == 'REH'):
            temp_dict['평균 상대습도(%)'] = float(aa[i]['fcstValue'])      
            new_df = new_df.append(temp_dict, ignore_index=True)
            temp_dict = {}
            
final_df = pd.concat([new_df.groupby(['지점', '일시'])['일강수량(mm)'].sum(),
                      new_df.groupby(['지점', '일시'])['평균 상대습도(%)'].mean()], axis=1).reset_index()

final_df['평균 현지기압(hPa)'] = np.nan
final_df['합계 일조시간(hr)'] = np.nan
final_df['작업일자'] = b_date
