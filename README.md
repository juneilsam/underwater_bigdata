# underwater_bigdata

한국 수자원 공사 물관리 디지털 트윈 & 빅데이터 구축 프로젝트

### 📆_기간

- 2021년 11월 ~ 2022년 02월(4개월)

### 🖍_주제 및 동기

- 지하수 자원의 효율적인 관리
- 수자원공사에 축적된 지하수 관련 데이터를 디지털 트윈 & 빅데이터 플랫폼을 통해 대중에게 제공하고자 함
- 수자원 공사 빅데이터 플랫폼에 서비스 중

      https://www.gims.go.kr/bdp/portal/ai-prediction-intro.do

### 📃_프로젝트 요약

- **디지털 트윈**을 이용하여 한국수자원공사에 **다년간 축적된 지하수 수자원 데이터와 미래의 데이터**를 효율적으로 **관리**함과 동시에  대중에게 효과적으로 **전달**할 수 있는 플랫폼을 구축하는 프로젝트이다.
- 지하수 자원은 일정한 수위를 유지하는 것이 매우 중요하여, **수위 변동에 시의적절한 대응**이 필요하다. 따라서 한국 수자원공사와 협업하여 지하수 수자원 **현황을 제공**하고, LSTM 모델을 이용하여 추측된 과거의 **결측값**과 **예측된 수위**를 **활용**하는 프로젝트를 추진하였다.

### 🎭_주요 역할

- 관측되지 않은 지하수 수위를 **기계학습(LSTM)을 통해 추측**
- **새로운 LSTM 모델을 구축**하고 **논문(Jeong et al.)에 소개된 모델 튜닝**을 한다.
- 기계학습을 통한 지하수 수위 **예측으로 이상 기록을 탐지**
- 기상청의 기압, 강수량, 습도, 일조량 데이터(이하 기후 데이터)와 수자원공사의 지하수위 데이터를 **전처리하여 DB에 연동**
- 전처리한 데이터를 시각화 하여 **전처리 전후 데이터를 비교**하고 **예측값을 그래프와 도표로 표현**한다.

### 🌞_세부 역할

- **데이터**
    - **기상청 API**로부터 수집한 **전국 12개 관측공 지역의 기후 데이터**를 이용한다.
    - **한국수자원공사**로부터 제공받은 **전국 12개 관측공의 수위 데이터**를 이용한다.
- **전처리**
    - 기후 데이터는 **시계열 순**으로 l**ine plot, scatter plot 등**으로 표현하여 넓은 범위에서 **이상치와 결측치, 추세와 계절성을 관측**한다.
    - 시간 단위의 데이터를 **다운샘플링**하여 일 단위로 변환하였다.
    - 이상치는 7일 단위의 **이동평균법**을 이용하여 판별하고, **선형보간**을 이용하여 대체한다.
    - 결측치는 강수량을 제외하고 선형보간을 이용하여 대체한다. 강수량은 **0으로 대체**한다.
- **알고리즘/모델**
    - 사용된 알고리즘은 시계열 데이터를 예측하는 데 탁월한 성능을 보이는 **LSTM** 알고리즘이다.
    - **첫 번째 모델**은 **기후 데이터를 입력하여 지하수 수위를 출력**하는 구조로 되어있고, 하나의 lstm층, 하나의 dense층으로 구성되어 있다(Jina Jeong, Eungyu Park, "Comparative applications of data-driven models representing water table fluctuations", Journal of Hydrology 572 (2019) 261–273.의 모델 하이퍼파라미터 튜닝).
    - **두 번째 모델**은 **기후 데이터와 이전 지하수 수위들의 차분(diff) 데이터를 입력**하여 이후 **지하수 수위를 출력**하는 구조로 되어있고, 하나의 lstm층, 하나의 dense층으로 구성되어 있다.
- **실서비스의 적용**
    - 구축된 모델과 기상청 기후 데이터, 수자원공사의 수위 데이터를 이용, **매일 익일부터 3개월간의 지하수 수위를 예측**한다.
    - 기상청 API로 제공받는 익일 포함 **3일간의 기상청 데이터를 이용**하되, 그 이후의 데이터는 **작년 동일 시기의 데이터**로 대체한다.
    - 사용되는 데이터와 예측 데이터는 수자원공사의 **Oracle DB에 업로드** 하고, 웹 상에서 그래프와 표, 3D 지도 등 그래픽으로 확인할 수 있도록 한다.
    - 예측된 수위와 실제 수위가 **80cm이상 차이를 보일 경우, 알림**이 나타나도록 한다.
    

### 🛠_사용 기술

- 사용 언어 - Python3
- Tool - Pycharm, Jupyter Notebook, Anaconda, Oracle DB
- 사용 라이브러리 - Tensorflow1, Keras, Pandas, Numpy, Xc Oracle 등
- OS - Window, Linux

### 🚣‍♀️_팀 구성

- 총 인원 : 약 20명
    - 총 책임자 : 1명
    - 중간 관리자 : 1명
    - 웹 개발자 :  7명
    - 검색 엔진 개발자 : 2명
    - DB 개발자 : 5명
    - 퍼블리셔 : 1명
    - 디자이너 : 1명
    - 3D 그래픽 : 1명
    - **인공지능 : 1명**

### ‼_이슈

- 수자원 공사의 **서버 컴퓨터**는 **보안 문제**로 인해 인터넷 연결이 불가능하였다. 데이터를 옮기기 위해서는 엄격한 보안 절차를 거쳐야 했기 때문에 설치도 쉽지 않았다. 심지어 **Tensorflow1은 파이썬 버전과 기타 라이브러리의 버전을 모두 맞추어야** 했고, 대부분의 라이브러리와 패키지들이 whl로만 설치가 가능했다. 하지만 이 조차도 **보안 SSL 충돌**이 발생하여 불가능했다.
    - 작업 컴퓨터에 **Anaconda 가상환경**을 구축하고, 모든 환경을 맞추었다. 그리고 해당 가상환경을 통째로 **tar 파일로 압축하여 수자원 공사 서버 컴퓨터로 옮겼다**. 이 방법으로 복잡한 설치 없이, Anaconda 설치와 **tar 파일 압축해제**만으로 같은 환경을 유지할 수 있도록 하였다.
- Oracle DB는 numpy와 pandas의 **정수, 날짜 타입**을 지원하지 않아, **업로드 에러**를 발생시켰다.
    - 업로드 대상 테이블의 행이나 열 단위로 업로드가 불가능하므로, 데이터 **업로드 직전 각각의 값을 일반적인 python 정수값이나 실수값, python 자체 날짜로 변환**하여 업로드 하는 것으로 해결하였다.

### 💡_배운 것

- **보안 환경에서 개발 환경**을 어떻게 구축해야 하는지 다양한 접근법을 연구해볼 수 있는 기회가 되었다. 여러 라이브러리와 패키지들의 버전 충돌 없이 하나의 프로젝트를 위한 환경 설정이 중요하다는 것을 배울 수 있었다.
    
    : **Anaconda 가상환경을 tar로 통째로 옮기는 것**을 익힐 수 있었다.
    
- 기존 프로젝트들의 LSTM 모델에서는 주로 **강수량 데이터와 shift한 이전 수위 데이터를 이용하여 미래의 수위를 예측**하였다. 하지만 이 프로젝트에서는 **오직 기후 데이터를 입력값으로 하고 수위 데이터는 출력값**으로만 활용하였는데, 처음 시도해보는 형태였다. 이러한 방식으로도 수위 변동을 예측할 수 있고, 시계열 데이터를 활용할 수 있다는 것을 알게 되었다.
- 시작은 30개 관측공을 전처리하는 것이었는데, 데이터의 특성에 맞는 전처리 방법을 연구할 수 있었고, **시계열 데이터 전처리** 부분에서 큰 자신감을 얻게 되었다.
    
    : **ScatterPlot, 이동평균법, 다운샘플링, 차분**을 이용하는 것을 익힐 수 있었다.
    
    : **비계절화, 비추세화**의 영향에 대해 배울 수 있었다.
    
- **Oracle DB** 파일 업로드는 처음이었는데, 기존 사용하였던 PostgreSQL, MySQL 기반의 DB와는 달리 업로드 가능한 **데이터 타입이 달랐다**.
