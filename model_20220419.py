#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras import backend as K
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import time
from dateutil.parser import parse
import os
import pandas as pd
import datetime as dt
import cx_Oracle as cx
import warnings
warnings.filterwarnings('ignore')

# 기준일
stand_date = pd.Timestamp(dt.datetime.now().date())

# Hyperparameters
###
# 랜덤 시드 설정
tf.set_random_seed()
# 이게 뭐지?
n_pst = 
# 학습 데이터의 분리 비율
# r_trn = 
r_trn = 
# LTS parameters
n_lts_itr = 
###
r_lts = 
min_lv = 
r_lw = 
r_hg = 
# Optimization parameters
dec_step = 
###
n_itr = 
###
toler1 = 
###
toler2 = 
# Design parameters
ref_int = 

# tm_gd60101의 spot_id, PM_GD60201의 quilty_id, 업로드 RESULT_ID, 기상관측망(TM_GD30704의 기후관측소코드)
spot_ids = {'02' : [219, 1, 14789, 211, '양구방산'], # 양구방산
            '04' : [94, 1, 14995, 101, '화천화천'], # 화천화천
            '06' : [225, 1, 16218, 114, '양평양동'], # 양평양동
            '07' : [32, 1, 14717, 119, '평택안중'], # 평택안중
            '08' : [383, 1, 14700, 95, '포천대회산'], # 포천대회산
            '10' : [310, 1, 14687, 288, '밀양가곡'], # 밀양가곡
            "11" : [191, 1, 14730, 289, '산청산청'], # 산청산청
            "12" : [195, 1, 16189, 279, '성주벽진'], # 성주벽진
            '14' : [253, 1, 14764, 278, '의성안계'], # 의성안계
            '22' : [38, 1, 14749, 235, '태안안면'], # 태안안면
            '24' : [152, 1, 16197, 285, '의령낙서(충적)'], # 의령낙서(충적)
            '25' : [152, 2, 14758, 285, '의령낙서(암반)'] } # 의령낙서(암반)

##################################################################################
########################## 주의!!!! DB 데이터 모두 삭제 ##########################
##################################################################################
# 데이터 업로드 전 출력값 데이터 모두 삭제
# LOCATION = r".\instantclient_21_3"
# os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록

# conn = cx.connect()
# curs = conn.cursor()
# curs.execute("DELETE FROM PM_GD60205")
# conn.commit()
# curs.close()
# conn.close()
##################################################################################
########################## 주의!!!! DB 데이터 모두 삭제 ##########################
##################################################################################

# 지하수 DB에서 조건에 맞는 것들 DF로 다운로드
def WTDBToDF(spot_id):
    # spot_id 값
    spot_id_value = spot_ids[spot_id][0]
    # spot_quality 값
    spot_qlt_value = spot_ids[spot_id][1]
    # cx 위치
    LOCATION = r".\instantclient_21_3"
    # 환경변수 등록
    os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록
    # dsn 생성 후 DB연결
    conn = cx.connect()
    curs = conn.cursor()
    # sql을 통해 DB select, 조건에 맞는 데이터 추출
    sql = f"SELECT RESULT_ID FROM tm_gd60101 WHERE SPOT_ID = {spot_id_value} AND OBSR_IEM_ID = 5 AND TIME_UNIT_ID = 4"
    # 결과값 추출
    result_id = pd.read_sql(sql, conn).iat[0, 0]
    # sql을 통해 필요한 데이터 받아오기(날짜, id조건)
    sql2 = f"select OBSR_DTA_VALUE, OBSR_DT from PM_GD60201 where result_id = {result_id} AND QLITY_ID = {spot_qlt_value}"
    water_df = pd.read_sql(sql2, conn)
    water_df.set_index('OBSR_DT', inplace=True)
    return water_df

# API 기후 날씨 DB에서 조건에 맞는 것들 DF로 다운로드 - 모델에 쓰는 것
# ['WETHER_OBSRVT_CODE', 'FRST_REGIST_DT', 'DE_AMTPRCP_VALUE', 'AVRG_RLTIV_HD_RT', 'AVRG_ARCSR_VALUE', 'SUM_SSH_TIME_VALUE']
def CMDBToDF(spot_id):
    # cd_id 값
    cd_id = str(spot_ids[spot_id][3])
    # cx 위치
    LOCATION = r".\instantclient_21_3"
    # 환경변수 등록
    os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록
    # dsn 생성 후 DB연결
    conn = cx.connect()
    curs = conn.cursor()
    # sql을 통해 DB select, 조건에 맞는 데이터 추출
    sql = f"SELECT FRST_REGIST_DT, DE_AMTPRCP_VALUE, AVRG_RLTIV_HD_RT, AVRG_ARCSR_VALUE, SUM_SSH_TIME_VALUE FROM TM_GD30705 \
    WHERE WETHER_OBSRVT_CODE={cd_id} AND SUM_SSH_TIME_VALUE != -1"
    
    # 결과값 추출
    result = pd.read_sql(sql, conn)
    result.set_index('FRST_REGIST_DT', inplace=True)
    return result

# API 기후 날씨 DB에서 조건에 맞는 것들 DF로 다운로드 - 예측에 쓰는 것
# ['WETHER_OBSRVT_CODE', 'FRST_REGIST_DT', 'DE_AMTPRCP_VALUE', 'AVRG_RLTIV_HD_RT', 'AVRG_ARCSR_VALUE', 'SUM_SSH_TIME_VALUE']
def PRDCMToDF(spot_id):
    # cd_id 값
    cd_id = str(spot_ids[spot_id][3])
    # cx 위치
    LOCATION = r".\instantclient_21_3"
    # 환경변수 등록
    os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록
    # dsn 생성 후 DB연결
    conn = cx.connect()
    curs = conn.cursor()
    # sql을 통해 DB select, 조건에 맞는 데이터 추출
    sql = f"SELECT FRST_REGIST_DT, DE_AMTPRCP_VALUE, AVRG_RLTIV_HD_RT FROM TM_GD30705 \
    WHERE WETHER_OBSRVT_CODE={cd_id} AND SUM_SSH_TIME_VALUE = -1"
    
    # 결과값 추출
    result = pd.read_sql(sql, conn)
    result.set_index('FRST_REGIST_DT', inplace=True)
    result = result[result.index.date >= (dt.datetime.now().date() + pd.Timedelta('1d'))]
    return result

# 출력된 결과물 DF를 DB에 업로드
def DFToDB(df):
    try:
        # cx 주소
        LOCATION = r".\instantclient_21_3"
        # cx 환경변수 등록
        os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록
        
        # cx 접속
        conn = cx.connect()
        curs = conn.cursor()

        # 데이터프레임을 튜플-리스트 형식으로 변환
        rows = [tuple(i) for i in df.to_records(index=False)]
        up_rows = []
        
        # 각 튜플을 업로드
        for j in df.index:
            # 각 열별 데이터 타입 변환
            r_1 = int(rows[j][0])
            r_2 = int(rows[j][1])
            r_3 = str(rows[j][2])
            # 날짜 타입
            r_4 = parse(rows[j][3])
            r_5 = int(rows[j][4])
            # 업로드
            up_rows.append(tuple([r_1, r_2, r_3, r_4, r_5]))

        curs.executemany("INSERT INTO PM_GD60205 \
                         (OBSR_DTA_ID, RESULT_ID, OBSR_DTA_VALUE, \
                         OBSR_DT, QLITY_ID) \
                         VALUES (:1, :2, :3, :4, :5)", up_rows)
        conn.commit()
        curs.close()
        conn.close()
        print("DB Inserted")
        
    except Exception as e:
        print("데이터를 DB에 업로드하는 데 문제가 있습니다.\n", e)
        
def decode(Y_trn, Y_prd1, Y_tst, Y_prd2, n_trn):
    return Y_trn, Y_prd1, Y_tst, Y_prd2
    
def RNN_model(X):
    return Y_trn, Y_prd1, Y_tst, Y_prd2

# Normalization (0 ~ 1) of both explanatory and response data
def normalization(dat, r1, r2):
    minv = np.min(dat, axis=0)
    maxv = np.max(dat, axis=0)
    dat = (dat - minv) / (maxv - minv) * (r2 - r1)
    np.nan_to_num(dat, copy=False)
    dat = dat + r1
    return dat, minv, maxv


# Detrending and deseasonalization by LTS LTS에 의한 비계절화(?)와 추세 제거
def lts_preproc(dat, flg_season):
    return Y_prd

def mk_temp_year(temp_df):
    temp_df1 = temp_df[(temp_df.index <= pd.Timestamp(str(stand_date.year - 1)+'-01-10'))]
    temp_df1.reset_index(drop=False, inplace=True)
    return temp_df1

def mk_temp_month(temp_df):
    try:
        temp_df = temp_df[(temp_df.index > (stand_date - pd.Timedelta('365d') - pd.Timedelta('90d')))
                         & (temp_df.index <= (stand_date - pd.Timedelta('365d') + pd.Timedelta('100d')))]
        temp_df['OBSR_DT'] = temp_df.index + pd.Timedelta('365d')

    except:
        print('except')
        temp_df = temp_df[(temp_df.index > (stand_date - pd.Timedelta('730d') - pd.Timedelta('90d')))
             & (temp_df.index <= (stand_date - pd.Timedelta('730d') + pd.Timedelta('100d')))]
        temp_df['OBSR_DT'] = temp_df.index + pd.Timedelta('730d')
    temp_df.reset_index(drop=True, inplace=True)
    return temp_df    

def up_data(water_spot, n):
    water_spot['OBSR_DTA_ID'] = n
    water_spot['RESULT_ID'] = spot_ids['{0:02d}'.format(n)][2]
    water_spot['수위'] = round(water_spot['OBSR_DTA_VALUE'], 10)
    water_spot['일시'] = water_spot['OBSR_DT'].astype(str)
    water_spot['QLITY_ID'] = 1 if len(water_spot) <= 300 else 2
    print(len(water_spot), water_spot['QLITY_ID'][0])
    water_spot = water_spot[['OBSR_DTA_ID', 'RESULT_ID', '수위', '일시', 'QLITY_ID']]
    tf.reset_default_graph()
    # 데이터 업로드
    return DFToDB(water_spot)

def start_pro(spot):
    tf.reset_default_graph()
    spot_id = list(spot_ids.keys())[spot]
    n = int(spot_id)
    place = spot_ids[spot_id][-1]
    print('='*10, place, '='*10)
    water_df = WTDBToDF(spot_id)
    weather_df = CMDBToDF(spot_id)
    df = pd.concat([water_df, weather_df], axis=1)
    df.reset_index(inplace=True)
    df.columns = ['OBSR_DT', 'OBSR_DTA_VALUE', '일강수량(mm)', '평균 상대습도(%)', '평균 현지기압(hPa)', '합계 일조시간(hr)']
    df['OBSR_DTA_VALUE'] = 0.0
    
    global sess_save_nam, out_file_nam
    
    sess_save_nam = f'./{n}_final_save_LSTM'

    global strt_lr, r_dec, n_stat_rnn, n_lay, n_neu
    
    if n in [4, 6, 10]:
        f = 3
        strt_lr = 0.00005
        r_dec = 0.95
        n_stat_rnn = 16
        n_lay = 4
    elif n == 12:
        f = 3
        strt_lr = 0.00005
        r_dec = 0.75
        n_stat_rnn = 32
        n_lay = 4
    elif n == 14:
        f = 0
        strt_lr = 0.00005
        r_dec = 0.8
        n_stat_rnn = 32
        n_lay = 2
    elif n == 2:
        f = 0
        strt_lr = 0.00005
        r_dec = 0.95
        n_stat_rnn = 16
        n_lay = 4
    elif n in [7, 8, 11]:
        f = 2
        strt_lr = 0.00005
        r_dec = 0.95
        n_stat_rnn = 16
        n_lay = 4
    elif n == 22:
        f = 0
        strt_lr = 0.00005
        r_dec = 0.95
        n_stat_rnn = 32
        n_lay = 2
        
    n_neu = n_stat_rnn

    df_temp_df = df.copy()
    sav_res = 0
    temp_df = df_temp_df.copy()
    temp_df.set_index('OBSR_DT', inplace=True)
    tf.reset_default_graph()
    Y_prd = process(temp_df.copy(), f, sav_res)
    print("model trained")  

    sav_res = 1
    temp_df = df_temp_df.copy()
    temp_df.set_index('OBSR_DT', inplace=True)
    tf.reset_default_graph()
    year_df = mk_temp_year(temp_df.copy())
    Y_prd = process(year_df, f, sav_res)  
    tf.reset_default_graph()
    temp_df.reset_index(inplace=True)
    water_spot = pd.concat([year_df['OBSR_DT'].iloc[:-10], pd.DataFrame(Y_prd, columns=['OBSR_DTA_VALUE'])], axis=1)
    print(water_spot)
    # up_data(water_spot, n)

    temp_df = df_temp_df.copy()
    temp_df.set_index('OBSR_DT', inplace=True)
    api_pred = PRDCMToDF(spot_id)
    api_index = list(api_pred.index)
    tf.reset_default_graph()
    month_df = mk_temp_month(temp_df.copy())
    month_df.loc[api_index[0]:api_index[-1]]['일강수량(mm)'] = api_pred.loc[api_index[0]:api_index[-1]]['DE_AMTPRCP_VALUE']
    month_df.loc[api_index[0]:api_index[-1]]['평균 상대습도(%)'] = api_pred.loc[api_index[0]:api_index[-1]]['AVRG_RLTIV_HD_RT']
    Y_prd = process(month_df, f, sav_res)
    month_df.reset_index(inplace=True)
    water_spot = pd.concat([month_df['OBSR_DT'].iloc[:-10], pd.DataFrame(Y_prd, columns=['OBSR_DTA_VALUE'])], axis=1)
    # up_data(water_spot, n)
    
def data_split(dataset):
    # split into train and test sets
    train_size = int(len(dataset) * 0.5)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    train_X, train_Y = train[:, :-1], train[:, -1]
    test_X, test_Y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

    return train_X, train_Y, test_X, test_Y

def model_train(train_X, train_Y):
    # create and fit the LSTM network
    callback = [EarlyStopping(monitor='loss', patience=3)]

    model = Sequential()
    model.add(LSTM(16, input_shape=(1, 3)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_Y, epochs=100, batch_size=15, verbose=2, callbacks=callback)

    model.save(sess_save_nam + '.h5')

    return model

def model2_pred(tmp, n):
    mm = tmp[['일강수량(mm)', 'OBSR_DTA_VALUE', '일강수량(mm)1']].values
    K.clear_session()
    model = load_model(sess_save_nam + '.h5')
    testPredict = model.predict(mm.reshape(mm.shape[0], 1, mm.shape[1]))
    tmp['temp'] = testPredict

    tmp['OBSR_DTA_VALUE'] = (tmp['OBSR_DTA_VALUE'] + tmp['OBSR_DTA_VALUE']).shift(1)
    tmp.at[0, 'OBSR_DTA_VALUE'] = first_value
    K.clear_session()
    print(tmp)
    # up_data(tmp, n)

for spot in range(0, 10):
    start_pro(spot)
    
for spot in range(10, 12):
    tf.reset_default_graph()
    spot_id = list(spot_ids.keys())[spot]
    n = int(spot_id)
    place = spot_ids[spot_id][-1]
    print('='*10, place, '='*10)
    water_df = WTDBToDF(spot_id)
    weather_df = CMDBToDF(spot_id)
    df = pd.concat([water_df, weather_df], axis=1)
    df.reset_index(inplace=True)
    df.columns = ['OBSR_DT', 'OBSR_DTA_VALUE', '일강수량(mm)', '평균 상대습도(%)', '평균 현지기압(hPa)', '합계 일조시간(hr)']
    
    df = df[['OBSR_DT', '일강수량(mm)', 'OBSR_DTA_VALUE']]
    
    for i in range(len(df)):
        mov_df = df.loc[i:i+11, 'OBSR_DTA_VALUE']
        mov_mean = float(mov_df.mean())
        mov_std = float(mov_df.std())
        mov_df[mov_df > (mov_mean + mov_std*1.0)] = np.nan
        mov_df[mov_df < (mov_mean - mov_std*1.0)] = np.nan
        mov_df[mov_df.isnull()] = mov_mean

    global scaler
    scaler = MinMaxScaler()
    scaled_df = df.copy()
    scaled_df = pd.concat([df[['OBSR_DT']], pd.DataFrame(scaler.fit_transform(df.copy()[['일강수량(mm)', 'OBSR_DTA_VALUE']]))], axis=1)
    scaled_df.columns = ['OBSR_DT', '일강수량(mm)', 'OBSR_DTA_VALUE']
    
    scaled_df['일강수량(mm)1'] = scaled_df['일강수량(mm)'].shift(-1)
    scaled_df['수위1'] = scaled_df['OBSR_DTA_VALUE'].shift(-1)
    scaled_df['diff'] = scaled_df['수위1'] - scaled_df['OBSR_DTA_VALUE']
    df_temp_df = scaled_df.copy()

    sess_save_nam = f'C:/Users/junei/Desktop/soda_ai_data/codes/save/{n}_final_save_LSTM'
    df_columns = df_temp_df.copy()

    dataset = df_columns[['일강수량(mm)', 'OBSR_DTA_VALUE', '일강수량(mm)1', 'diff']]

    train_X, train_Y, test_X, test_Y = data_split(dataset.values)
    K.clear_session()
    model = model_train(train_X, train_Y)
    temp_df = df_temp_df.copy()
    temp_df.set_index('OBSR_DT', inplace=True)

    tmp = mk_temp_year(temp_df.copy())
    first_value = tmp.at[0, 'OBSR_DTA_VALUE']
    model2_pred(tmp, n)

    tmp = mk_temp_month(temp_df.copy())
    api_pred = PRDCMToDF(spot_id)
    api_index = list(api_pred.index)
    tmp['일강수량(mm)'] = 0
    tmp['일강수량(mm)1'] = 0
    aa.loc[list(aa.index)[0]:list(aa.index)[-1]]
    tmp.loc[api_index[0]:api_index[-1]]['일강수량(mm)'] = api_pred.loc[api_index[0]:api_index[-1]]['DE_AMTPRCP_VALUE']
    tmp.loc[api_index[0]:api_index[-2]]['일강수량(mm)1'] = api_pred.iloc[api_index[1]:api_index[-1]]['DE_AMTPRCP_VALUE']
    tmp = tmp[['OBSR_DT', '일강수량(mm)', 'OBSR_DTA_VALUE', '일강수량(mm)1']]
    model2_pred(tmp, n)
