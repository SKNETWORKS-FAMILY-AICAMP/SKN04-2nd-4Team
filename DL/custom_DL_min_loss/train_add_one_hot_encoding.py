from src.data import ChurnDataset, ChurnDataModule
from sklearn.utils.class_weight import compute_class_weight

from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import ChurnModule

import pandas as pd
import numpy as np
import random
import json
import nni
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns


def main(configs):
    seed = 0
    
    # 데이터셋을 로드
    #data = pd.read_csv('../day36/data/nan_filtered_train_data.csv', index_col=0)
    data = pd.read_csv('../preprocessing_코드정리/drop_nan_vals_data.csv', index_col=0)
    
    data = data.drop(columns=['CustomerID'])

    # categorical 로 사용하지 않고, unknown을 0으로 대체 후 num_col로
    data['HandsetPrice'] = data['HandsetPrice'].replace('Unknown', 0)
    data['HandsetPrice'] = data['HandsetPrice'].astype(int)

    # categorical_cols = ['Churn', 'ServiceArea', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 
    #                     'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 
    #                     'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 
    #                     'NotNewCellphoneUser', 'OwnsMotorcycle', 'MadeCallToRetentionTeam', 'PrizmCode', 'Occupation', 
    #                     'MaritalStatus']

    # # 범주형 열을 정수형으로 변환
    # data, _ = convert_category_into_integer(data, categorical_cols)
    
    
    categorical_cols = ['Churn', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 
                    'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 
                    'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 
                    'NotNewCellphoneUser', 'OwnsMotorcycle']

    nominal_cols = ['ServiceArea', 'MadeCallToRetentionTeam', 'Occupation', 'MaritalStatus', 'PrizmCode']

    # 범주형 열을 정수형으로 변환
    encoded_data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)  # drop_first=False means we keep all categories
    encoded_data, _ = convert_category_into_integer(encoded_data, categorical_cols)
    encoded_cols = encoded_data.columns.difference(data.columns)

    # 데이터프레임을 float32로 변환
    encoded_data = encoded_data.astype(np.float32)

    # 데이터셋을 학습용과 임시 데이터로 분할
    train, temp = train_test_split(encoded_data, test_size=0.4, random_state=0)
    # 임시 데이터를 검증용과 테스트용 데이터로 분할
    valid, test = train_test_split(temp, test_size=0.5, random_state=0)

    standard_scaler = StandardScaler()

    # 훈련 데이터의 열을 표준화
    need_scale_cols = encoded_data.columns.difference(categorical_cols + ['IncomeGroup'] + list(encoded_cols))
    
    train.loc[:, need_scale_cols] = \
        standard_scaler.fit_transform(train.loc[:, need_scale_cols])

    # 검증 데이터와 테스트 데이터의 열을 훈련 데이터의 통계로 표준화
    valid.loc[:, need_scale_cols] = \
        standard_scaler.transform(valid.loc[:, need_scale_cols])

    test.loc[:, need_scale_cols] = \
        standard_scaler.transform(test.loc[:, need_scale_cols])

    # 데이터셋 객체로 변환
    train_dataset = ChurnDataset(train)
    valid_dataset = ChurnDataset(valid)
    test_dataset = ChurnDataset(test)
    
    ### weight 관련 추가 
    # y_train = train['Churn'].values
    # class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = None

    # 데이터 모듈 생성 및 데이터 준비
    churn_data_module = ChurnDataModule(batch_size=configs.get('batch_size'))
    churn_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 생성
    configs.update({'input_dim': len(encoded_data.columns)-1})
    model = Model(configs)

    # LightningModule 인스턴스 생성
    churn_module = ChurnModule(
        model=model,
        configs=configs,
        class_weights=class_weights
    )

    # Trainer 인스턴스 생성 및 설정
    del configs['output_dim'], configs['seed']
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=10)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'churn/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=churn_module,
        datamodule=churn_data_module,
    )
    trainer.test(
        model=churn_module,
        datamodule=churn_data_module,
    )


if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda', 그렇지 않으면 'cpu' 사용
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update(nni_params)
        print(nni_params)

    # seed 설정
    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA 설정
    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(configs)