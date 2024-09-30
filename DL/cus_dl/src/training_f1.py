import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from sklearn.metrics import classification_report, confusion_matrix
import nni
import numpy as np
from sklearn.metrics import f1_score

class XyModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,      
        configs: dict,
    ):
        super().__init__()
        self.model = model        
        self.configs = configs
        self.learning_rate = configs.get('learning_rate')
        if configs.get('device') == 'gpu':
            self.pos_weight=self.configs.get('class_weights').to('cuda')
            self.pos_weight= torch.tensor(1)
        else:
            self.pos_weight=self.configs.get('class_weights')
    def training_step(self, batch, batch_idx):
        # 학습 단계에서 호출되는 메서드
        
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴
        
        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.flatten(output)
        
        # self.train_loss = F.binary_cross_entropy(sig_out, y)  

        self.train_loss = F.binary_cross_entropy_with_logits(squeeze_output, y)
        sig_out = F.sigmoid(squeeze_output)
        y_pred = (sig_out >= 0.5).float()
        self.train_acc = (y_pred==y).float().mean()
        self.train_f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='binary')

        return self.train_loss

    def on_train_epoch_end(self, *args, **kwargs):
        self.log_dict(
            {
                'train_loss': self.train_loss,
                'train_acc': self.train_acc,
                'train_f1': self.train_f1
            }, # type: ignore
            on_epoch=True,  # 에폭마다 기록
            prog_bar=True,  # 프로그레스 바에 표시
            logger=True     # 로그 파일에도 기록
        )

    
    def validation_step(self, batch, batch_idx):

        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴

        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.flatten(output)
        
        # self.val_loss = F.binary_cross_entropy(sig_out, y)  
        self.val_loss = F.binary_cross_entropy_with_logits(squeeze_output, y)
        sig_out = F.sigmoid(squeeze_output)
        y_pred = (sig_out >= 0.5).float()
        self.val_acc = (y_pred==y).float().mean()
        self.val_f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='binary')

        return {'val_loss': self.val_loss, 'val_acc': self.val_acc, 'val_f1': self.val_f1}

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                'val_loss': self.val_loss,
                'val_acc': self.val_acc,
                'val_f1': self.val_f1
            }, # type: ignore
            on_epoch=True,  # 에폭마다 기록
            prog_bar=True,  # 프로그레스 바에 표시
            logger=True     # 로그 파일에도 기록
        )
        nni.report_intermediate_result(self.val_f1) 
        
    def test_step(self, batch, batch_idx):
        # 테스트 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴

        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.flatten(output)
        
        # self.test_loss = F.binary_cross_entropy(sig_out, y)  
        self.test_loss = F.binary_cross_entropy_with_logits(squeeze_output, y)
        sig_out = F.sigmoid(squeeze_output)
        y_pred = (sig_out >= 0.5).float()
        self.test_acc = (y_pred==y).float().mean()
        self.test_f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='binary')

        return {'test_loss': self.test_loss, 'test_acc': self.test_acc, 'test_f1': self.test_f1}
    
    def on_test_epoch_end(self):
        self.log_dict(
            {
                'test_loss': self.test_loss,
                'test_acc': self.test_acc,
                'test_f1': self.test_f1
            }, # type: ignore
            on_epoch=True,  # 에폭마다 기록
            prog_bar=True,  # 프로그레스 바에 표시
            logger=True     # 로그 파일에도 기록
        )
        if self.configs.get('nni'):
            nni.report_final_result(self.test_f1)



    def configure_optimizers(self):
        # 옵티마이저와 스케줄러를 설정하는 메서드
        optimizer = optim.Adam(
            self.model.parameters(),  # 모델의 파라미터를 옵티마이저에 전달
            lr=self.learning_rate,    # type: ignore
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',               
            factor=0.5,               
            patience=5,              
        )

        return {
            'optimizer': optimizer,   # 옵티마이저 반환
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1',
                'interval': 'epoch',
                'frequency': 1,
            }
        }