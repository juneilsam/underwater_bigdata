# 디렉토리의 주소
ddi = './data/pre/'
dir_list = os.listdir(ddi)

# 날씨 불러오기
weather_all = pd.read_csv((ddi + dir_list[-1]), parse_dates=['일시'])
weather_all.rename(columns = {'일시' : 'OBSR_DT'}, inplace = True)
weather_all.set_index('OBSR_DT', inplace=True)

# 기준일
stand_date = pd.Timestamp(dt.datetime.now().date())

# Hyperparameters
###
# 랜덤 시드 설정
tf.set_random_seed()
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

# tm_gd60101의 spot_id, PM_GD60201의 quilty_id, 업로드 RESULT_ID, 기상관측망
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

##################################################################################
########################## 주의!!!! DB 데이터 모두 삭제 ##########################
##################################################################################
# 데이터 업로드 전 출력값 데이터 모두 삭제
# LOCATION = r".\instantclient_21_3"
# os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록

# conn = cx.connect("", "", cx.makedsn("..", "", ""))
# curs = conn.cursor()
# curs.execute("DELETE FROM GD602")
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
    conn = cx.connect("", "", cx.makedsn("..", "", ""))
    curs = conn.cursor()
    # sql을 통해 DB select, 조건에 맞는 데이터 추출
    sql = f"SELECT RESULT_ID FROM gd601 WHERE SPOT_ID = {spot_id_value} AND OBSR_IEM_ID = 5 AND TIME_UNIT_ID = 4"
    # 결과값 추출
    result_id = pd.read_sql(sql, conn).iat[0, 0]
    # sql을 통해 필요한 데이터 받아오기(날짜, id조건)
    sql2 = f"select result_id, OBSR_DTA_VALUE, OBSR_DT from PM_GD60201 where result_id = {result_id} and OBSR_DT BETWEEN to_Date('20220101', 'RRRRMMDD') AND to_Date('20220301', 'RRRRMMDD') AND QLITY_ID = {spot_qlt_value}"
    water_df = pd.read_sql(sql2, conn)
    water_df.set_index('OBSR_DT', inplace=True)
    return water_df

# 출력된 결과물 DF를 DB에 업로드
def DFToDB(df):
    try:
        # cx 주소
        LOCATION = r".\instantclient_21_3"
        # cx 환경변수 등록
        os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"] #환경변수 등록
        
        # cx 접속
        conn = cx.connect("", "", cx.makedsn("..", "", ""))
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

        curs.executemany("INSERT INTO GD602 \
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
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_stat_rnn, activation=tf.nn.tanh)
    return out

def n_array(dat_res_raw, dat_exp_raw, dat_io, sav_res):
    n_dat = np.shape(dat_res_raw)[0] - n_pst
    n_exp = np.shape(dat_exp_raw)[1] 
    Y_trn, Y_prd1, Y_tst, Y_prd2 = decode(Y_trn, Y_prd1, Y_tst, Y_prd2, n_trn)
    return Y_trn, Y_prd1, Y_tst, Y_prd2

# Normalization (0 ~ 1) of both explanatory and response data
def normalization(dat, r1, r2):
    minv = np.min(dat, axis=0)
    maxv = np.max(dat, axis=0)
    dat = (dat - minv) / (maxv - minv) * (r2 - r1)
    np.nan_to_num(dat, copy=False)
    dat = dat + r1
    return dat, minv, maxv


# Detrending and deseasonalization by LTS LTS에 의한 비계절화와 추세 제거
def lts_preproc(dat, flg_season):
    _Y = dat
    n_raw_dat = np.shape(dat)[0]
    return _Y, _X, _W

def process(df1, f, sav_res):
    df1['일강수량(mm)'] = 0.0
    df1.reset_index(drop=True, inplace=True)

    try:
        dat_exp_raw1 = df1[['일강수량(mm)', '평균 상대습도(%)', '평균 현지기압(hPa)', '합계 일조 시간(hr)']].values
    except:
        dat_exp_raw1 = df1[['일강수량(mm)', '평균 상대습도(%)', '평균 현지기압(hPa)', '합계 일조시간(hr)']].values

    dat_res_raw1 = df1['OBSR_DTA_VALUE'].values

    dat_io = np.ones([np.shape(dat_res_raw1)[0], 1])
    for ii in range(np.shape(dat_res_raw1)[0]):
        if dat_res_raw1[ii] <= min_lv:
            dat_io[ii] = 0

    global _X, _W
    dat_res_raw, _X, _W = lts_preproc(dat_res_raw1, f)
    dat_exp_raw, _, _ = normalization(dat_exp_raw1, r_lw, r_hg)

    global minv_r, maxv_r
    dat_res_raw, minv_r, maxv_r = normalization(dat_res_raw, r_lw, r_hg)
    Y_trn, Y_prd1, Y_tst, Y_prd2 = n_array(dat_res_raw, dat_exp_raw, dat_io, sav_res)
    tf.reset_default_graph()
    
    # Save estimation results
    Y_prd=np.zeros(np.shape(Y_prd1)[0]+np.shape(Y_prd2)[0])
    Y_prd[: np.shape(Y_prd1)[0]] = Y_prd1
    Y_prd[np.shape(Y_prd1)[0]:] = Y_prd2
    tf.reset_default_graph()
    return Y_prd

# 연 단위의 동일시기 기후 데이터를 불러온다.
def mk_temp_year(temp_df):
    temp_df1 = temp_df[(temp_df.index <= pd.Timestamp(str(stand_date.year - 1)+'-01-10'))]
    temp_df1.reset_index(drop=False, inplace=True)
    return temp_df1

# 작년 동일시기 혹은 재작년 동일시기의 기후 데이터를 불러온다.
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
    n = int(dir_list[spot][:2])
    place = dir_list[spot]
    print('='*10, place[:-11], '='*10)
    water_df = WTDBToDF(dir_list[spot][:2])
    weather_df = weather_all[weather_all['지점'] == spot_ids[dir_list[spot][:2]][3]]
    df = pd.concat([water_df, weather_df], axis=1)
    df.reset_index(inplace=True)
    
    global sess_save_nam, out_file_nam
    
    sess_save_nam = f'./codes/save/{n}_final_save_LSTM'

    global strt_lr, r_dec, n_stat_rnn, n_lay, n_neu
    
    # 각 지역의 모델별 하이퍼파라미터 
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

    # 모델 학습
    df_temp_df = df.copy()
    sav_res = 0
    temp_df = df_temp_df.copy()
    temp_df.set_index('OBSR_DT', inplace=True)
    tf.reset_default_graph()
    Y_prd = process(temp_df.copy(), f, sav_res)
    print("model trained")  

    # 학습된 모델을 기준으로 새 데이터 적용
    sav_res = 1
    temp_df = df_temp_df.copy()
    temp_df.set_index('OBSR_DT', inplace=True)
    tf.reset_default_graph()
    
    # 연 단위
    year_df = mk_temp_year(temp_df.copy())
    Y_prd = process(year_df, f, sav_res)  
    tf.reset_default_graph()
    temp_df.reset_index(inplace=True)
    water_spot = pd.concat([year_df['OBSR_DT'].iloc[:-10], pd.DataFrame(Y_prd, columns=['OBSR_DTA_VALUE'])], axis=1)
    up_data(water_spot, n)

    temp_df = df_temp_df.copy()
    temp_df.set_index('OBSR_DT', inplace=True)
    tf.reset_default_graph()
    
    # 월 단위
    month_df = mk_temp_month(temp_df.copy())
    Y_prd = process(month_df, f, sav_res)
    month_df.reset_index(inplace=True)
    water_spot = pd.concat([month_df['OBSR_DT'].iloc[:-10], pd.DataFrame(Y_prd, columns=['OBSR_DTA_VALUE'])], axis=1)
    up_data(water_spot, n)
    
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
    up_data(tmp, n)
    
for spot in range(10, 13):
    n = int(dir_list[spot][:2])
    place = dir_list[spot]
    print('='*10, place[:-9], '='*10)
    water_df = WTDBToDF(dir_list[spot][:2])
    weather_df = weather_all[weather_all['지점'] == spot_ids[dir_list[spot][:2]][3]]
    df = pd.concat([water_df, weather_df], axis=1)
    df.reset_index(inplace=True)
    
    df = df[['OBSR_DT', '일강수량(mm)', 'OBSR_DTA_VALUE']]
    
    for i in range(len(df)):
        mov_df = df.loc[i:i+11, 'OBSR_DTA_VALUE']
        mov_mean = float(mov_df.mean())
        mov_std = float(mov_df.std())
        mov_df[mov_df > (mov_mean + mov_std*1.0)] = np.nan
        mov_df[mov_df < (mov_mean - mov_std*1.0)] = np.nan
        mov_df[mov_df.isnull()] = mov_mean
    
    df['일강수량(mm)1'] = df['일강수량(mm)'].shift(-1)
    df['수위1'] = df['OBSR_DTA_VALUE'].shift(-1)
    df['diff'] = df['수위1'] - df['OBSR_DTA_VALUE']
    df_temp_df = df.copy()

    sess_save_nam = f'./codes/save/{n}_final_save_LSTM'

    df_columns = df_temp_df.copy()

    dataset = df_columns[['일강수량(mm)', 'OBSR_DTA_VALUE', '일강수량(mm)1', 'diff']]

    train_X, train_Y, test_X, test_Y = data_split(dataset.values)
    K.clear_session()
    model = model_train(train_X, train_Y)
    temp_df = df_temp_df.copy()
    temp_df.set_index('OBSR_DT', inplace=True)

    tmp = mk_temp_year(temp_df)
    first_value = tmp.at[0, 'OBSR_DTA_VALUE']
    model2_pred(tmp, n)

    tmp = mk_temp_month(temp_df)
    tmp['일강수량(mm)'] = 0
    tmp['일강수량(mm)1'] = 0
    tmp = tmp[['OBSR_DT', '일강수량(mm)', 'OBSR_DTA_VALUE', '일강수량(mm)1']]   
    model2_pred(tmp, n)
