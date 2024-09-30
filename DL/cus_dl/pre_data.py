import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

def convert_category_into_integer(df: pd.DataFrame, columns: list):
    #주어진 DataFrame의 특정 열들을 범주형에서 정수형으로 변환합니다.
    label_encoders = {}  # 각 열의 LabelEncoder 객체를 저장할 딕셔너리입니다.
    for column in columns:
        # 각 열에 대해 LabelEncoder 객체를 생성합니다.
        label_encoder = LabelEncoder()
        
        # LabelEncoder를 사용하여 해당 열의 범주형 데이터를 정수형으로 변환합니다.
        df.loc[:, column] = label_encoder.fit_transform(df[column])
        
        # 변환된 LabelEncoder 객체를 딕셔너리에 저장합니다.
        label_encoders.update({column: label_encoder})
    
    # 변환된 데이터프레임과 LabelEncoder 객체를 포함하는 딕셔너리를 반환합니다.
    return df, label_encoders

# 극단치를 특정 조건에 따라 제거하는 함수 정의
def remove_outliers(df: pd.DataFrame, column, threshold):
    cleaned_df = df[df[column] <= threshold]
    deleted_count = df.shape[0] - cleaned_df.shape[0]
    return cleaned_df

def preprocessing(df: pd.DataFrame): 
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
    #remove_outliers 사용
    # 각 컬럼에 대해 극단치 제거
    for column, threshold in outlier_conditions.items():
        train_cleaned = remove_outliers(train_cleaned, column, threshold)


    #CustomerID 컬럼 제거
    train_cleaned.drop(columns=['CustomerID'], inplace=True)
    
    return train_cleaned


# 학습 데이터와 테스트 데이터에 동일한 PCA 변환을 적용하는 함수
def pca_merge_correlated_columns(X_train, X_test, positive_threshold=0.9, negative_threshold=-0.9):
    # 상관계수 행렬 계산 (컬럼 간 상관관계)
    corr_matrix = X_train.corr()

    # 양의 상관관계 그룹과 음의 상관관계 그룹을 저장할 리스트
    positive_correlated_groups = []
    negative_correlated_groups = []

    # 이미 병합된 피처들을 추적하기 위한 세트
    used_columns = set()

    for i in range(len(corr_matrix.columns)):
        if corr_matrix.columns[i] not in used_columns:
            # i번째 컬럼과 상관계수가 positive_threshold 이상인 피처 그룹 추출 (양의 상관관계)
            high_positive_corr_cols = corr_matrix.columns[corr_matrix.iloc[i] >= positive_threshold].tolist()
            high_positive_corr_cols = [col for col in high_positive_corr_cols if col not in used_columns]

            # i번째 컬럼과 상관계수가 negative_threshold 이하인 피처 그룹 추출 (음의 상관관계)
            high_negative_corr_cols = corr_matrix.columns[corr_matrix.iloc[i] <= negative_threshold].tolist()
            high_negative_corr_cols = [col for col in high_negative_corr_cols if col not in used_columns]

            # 양의 상관관계 그룹 추가
            if len(high_positive_corr_cols) > 1:
                positive_correlated_groups.append(high_positive_corr_cols)
                used_columns.update(high_positive_corr_cols)

            # 음의 상관관계 그룹 추가
            if len(high_negative_corr_cols) > 1:
                negative_correlated_groups.append(high_negative_corr_cols)
                used_columns.update(high_negative_corr_cols)

    # PCA 적용 및 변환
    X_train_pca = X_train.copy()
    X_test_pca = X_test.copy()
    pca_models = {}

    # 양의 상관관계 그룹에 대해 PCA 적용
    for group in positive_correlated_groups:
        pca = PCA(n_components=1)
        X_train_pca[f'{"_".join(group)}_pca_pos'] = pca.fit_transform(X_train[group])
        X_test_pca[f'{"_".join(group)}_pca_pos'] = pca.transform(X_test[group])  # 동일한 PCA 변환을 테스트 데이터에도 적용
        X_train_pca.drop(columns=group, inplace=True)
        X_test_pca.drop(columns=group, inplace=True)
        pca_models[f'{"_".join(group)}_pca_pos'] = pca

    # 음의 상관관계 그룹에 대해 PCA 적용
    for group in negative_correlated_groups:
        pca = PCA(n_components=1)
        X_train_pca[f'{"_".join(group)}_pca_neg'] = pca.fit_transform(X_train[group])
        X_test_pca[f'{"_".join(group)}_pca_neg'] = pca.transform(X_test[group])
        X_train_pca.drop(columns=group, inplace=True)
        X_test_pca.drop(columns=group, inplace=True)
        pca_models[f'{"_".join(group)}_pca_neg'] = pca

    return X_train_pca, X_test_pca