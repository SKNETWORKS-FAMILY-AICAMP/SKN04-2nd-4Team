from sklearn.utils import compute_class_weight
from src.data import XyDataset, XyDataModule
from src.utils import convert_category_into_integer
from src.model.churn_mlp_4layer import Model
from src.training_acc import XyModule
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np
import random
import json
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns
from torch.nn.utils.rnn import pad_sequence

import torchvision.datasets as ds 
import torchvision.transforms as transforms
import nni

def main(configs):

    # data = pd.read_csv(
    #     './data/train.csv',
    #     encoding='cp949',
    # )
    data = pd.read_csv(
        './data/train_cleaned_outlier_del.csv',
        encoding='cp949',
    )
    data = data.drop(columns=['Unnamed: 0', 'CustomerID',])
    object_columns = data.select_dtypes(include=['object'])
    convert_data, _ = convert_category_into_integer(data, object_columns)
    drop_data = convert_data.dropna()
    
    x_data = drop_data.drop(columns=['Churn'])
    y_data = drop_data.Churn.astype(int)
    
        # 클래스 가중치 계산,smote사용시는 적용x
    class_weights = compute_class_weight('balanced', classes=np.unique(y_data), y=y_data)
    class_weights = torch.tensor([class_weights[1]], dtype=torch.float)

    # 데이터셋을 학습용과 임시 데이터로 분할
    train_x, temp_x, train_y, temp_y = train_test_split(
        x_data, y_data,
        test_size=0.4,
        random_state=seed,
        stratify=y_data  # y가 적절히 분포되도록
    )

    # 임시 데이터를 검증용과 테스트용 데이터로 분할
    valid_x, test_x, valid_y, test_y = train_test_split(
        temp_x, temp_y,
        test_size=0.5,
        random_state=seed,
        stratify=temp_y  # y가 적절히 분포되도록
    )
    standard_scaler = StandardScaler()
    scaled_train_x = standard_scaler.fit_transform(train_x)
    scaled_valid_x = standard_scaler.transform(valid_x)
    scaled_test_x = standard_scaler.transform(test_x)

    # # SMOTE 적용
    # smote = SMOTE(random_state=seed)
    # train_x_resampled, train_y_resampled = smote.fit_resample(scaled_train_x, train_y)
    # # 타입 변환 (넘파이 배열을 PyTorch 텐서로 변환)

    ###################################################################################

    # 데이터셋 객체로 변환
    train_dataset = XyDataset(scaled_train_x, train_y)
    valid_dataset = XyDataset(scaled_valid_x, valid_y)
    test_dataset = XyDataset(scaled_test_x, test_y)

    # 데이터 모듈 생성 및 데이터 준비
    xy_data_module = XyDataModule(batch_size=configs.get('batch_size'))
    xy_data_module.prepare(train_dataset, valid_dataset, test_dataset)
    
    # 모델 생성
    configs.update({
        'input_dim': len(convert_data.columns)-1,
        'class_weights': class_weights,
        })
    model = Model(configs)

    # LightningModule 인스턴스 생성
    xy_module = XyModule(
        model=model,
        configs=configs
    )

    # Trainer 인스턴스 생성 및 설정
    #del configs['output_dim'], configs['seed'], configs['epochs'], configs['seq_len'], configs['input_dim']
    exp_name = ','.join([f'{key}={value}' for key, value in list(configs.items())[:5]])
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            # EarlyStopping(monitor='loss/val_loss', mode='min', patience=10)
            EarlyStopping(monitor='val_loss', mode='min', patience=10)
        ],

        'logger': TensorBoardLogger(
            'tensorboard',
            f'customer/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({
            'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=xy_module,
        datamodule=xy_data_module,
    )
    trainer.test(
        model=xy_module,
        datamodule=xy_data_module,
    )

if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda', 그렇지 않으면 'cpu' 사용
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    with open('./configs_acc.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    if configs.get('nni'):
        nni_params = nni.get_next_parameters()
        # nni_params.get('batch_size', configs.get('batch_size'))
        # nni_params.get('hidden_dim1', configs.get('hidden_dim1'))
        # nni_params.get('hidden_dim2', configs.get('hidden_dim2'))
        # nni_params.get('learning_rate', configs.get('learning_rate'))
        # nni_params.get('dropout_ratio', configs.get('dropout_ratio'))
        # nni_params.get('epochs', configs.get('epochs'))
        # configs를 nni_params로 업데이트
        configs.update(nni_params)


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