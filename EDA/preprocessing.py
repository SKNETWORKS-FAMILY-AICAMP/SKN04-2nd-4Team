import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# 극단치를 특정 조건에 따라 제거하는 함수 정의
def remove_outliers(df, column, threshold):
    cleaned_df = df[df[column] <= threshold]
    deleted_count = df.shape[0] - cleaned_df.shape[0]
    print(f'{column} : {cleaned_df.shape[0]}, 삭제된 수 : {deleted_count}')
    return cleaned_df

def preprocessing(df):
    # 고객 당 nan column의 수 확인
    total_column_len = len(df.columns)
    df['nan_column_counts'] = df.isna().sum(axis=1)
    df['nan_columns_percentage'] = df['nan_column_counts'] / total_column_len
    
    df = df[df['nan_column_counts'] <= 2]
    df = df.drop(columns=['nan_column_counts', 'nan_columns_percentage'])
    
    # nan 값 채우기 (정연)
    df['ServiceArea'] = df['ServiceArea'].fillna('NODATA000')
    df['AgeHH1'] = df['AgeHH1'].fillna(0)
    df['AgeHH2'] = df['AgeHH2'].fillna(0)
    df['PercChangeRevenues'] = df['PercChangeRevenues'].fillna(0)
    df['PercChangeMinutes'] = df['PercChangeMinutes'].fillna(0)
    
    df['CreditRating'] = df['CreditRating'].apply(lambda x: int(x[0]))
    
    print("nan-value filled")
    
    # 각각 하나씩 확인후 극단치 특정 조건으로 삭제 진행 (상집)
    # 극단치에 대한 상세 정보

    # 각 컬럼에 대해 극단치 제거 및 결과 저장
    train_cleaned = df  # 초기 데이터프레임 설정

    # 극단치 조건을 가진 컬럼과 임계값 정의
    outlier_conditions = {
        'MonthlyRevenue': 1000,
        'MonthlyMinutes': 6000,
        'TotalRecurringCharge': 300,
        'DirectorAssistedCalls': 80,
        'OverageMinutes': 2000,
        'RoamingCalls': 200,
        'PercChangeMinutes': 4000,
        'DroppedCalls': 150,
        'BlockedCalls': 300,
        'UnansweredCalls': 600,
        'CustomerCareCalls': 250,
        'ThreewayCalls': 40,
        'ReceivedCalls': 2000,
        'OutboundCalls': 500,
        'InboundCalls': 300,
        'PeakCallsInOut': 1500,
        'OffPeakCallsInOut': 1000,
        'DroppedBlockedCalls': 200,
        'CallForwardingCalls': 10,
        'CallWaitingCalls': 150,
        'MonthsInService': 60,
        'UniqueSubs': 20,
        'ActiveSubs': 20,
        'Handsets': 20,
        'HandsetModels': 12,
        'CurrentEquipmentDays': 1750,
        'RetentionCalls': 1,
        'RetentionOffersAccepted': 1,
        'ReferralsMadeBySubscriber': 5,
        'AdjustmentsToCreditRating': 15,
    }

    # 각 컬럼에 대해 극단치 제거
    for column, threshold in outlier_conditions.items():
        train_cleaned = remove_outliers(train_cleaned, column, threshold)

    # 최종적으로 청소된 데이터프레임 출력
    print(f'최종 데이터 크기 : {train_cleaned.shape[0]}')
    
    return train_cleaned