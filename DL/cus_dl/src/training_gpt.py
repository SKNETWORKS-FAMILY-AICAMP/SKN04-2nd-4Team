import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from sklearn.metrics import classification_report, confusion_matrix
import nni
import numpy as np

class XyModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,          # 모델 객체 (nn.Module을 상속받은 모델)
        configs: dict,
    ):
        super().__init__()
        self.model = model         # 모델을 초기화
        self.configs = configs
        self.learning_rate = configs.get('learning_rate')
        
        self.train_accs=[]
        self.test_accs=[]
        self.val_accs=[]

    def training_step(self, batch, batch_idx):
        # 학습 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴
        
        output = self.model(X)  # 모델을 통해 예측값을 계산
        output = torch.squeeze(output)
        sig_out = F.sigmoid(output)
        self.train_loss = F.binary_cross_entropy(sig_out, y)  

        y_pred = (sig_out >= 0.5).float()
        self.train_acc = (y_pred == y).type_as(y).float().mean()
        self.train_accs.append(self.train_acc)
  
        return self.train_loss  # 계산된 손실 반환
    
    def on_train_epoch_end(self, *args, **kwargs):
        # 학습 에포크가 끝날 때 호출되는 메서드
        avg_train_acc = torch.mean(torch.tensor(self.train_accs))
        
        self.log_dict(
            {'loss/train_loss': self.train_loss,
             'acc/train_acc': avg_train_acc},  # 학습 손실 및 정확도 기록
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        self.train_accs.clear()  # 에포크 종료 시 정확도 리스트 초기화
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_accs.clear()
        # 검증 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴

        output = self.model(X)  # 모델을 통해 예측값을 계산
        output = torch.squeeze(output)
        sig_out = F.sigmoid(output)
        self.val_loss = F.binary_cross_entropy(sig_out, y)  

        y_pred = (sig_out >= 0.5).float()
        self.val_acc = (y_pred == y).type_as(y).float().mean()
        
        self.val_accs.append(self.val_acc)
        
        return self.val_loss  # 검증 손실 반환
    
    def on_validation_epoch_end(self):
        # 검증 에포크가 끝난 후 평균 정확도 및 손실 계산
        avg_val_acc = torch.mean(torch.tensor(self.val_accs))
        avg_val_loss = self.val_loss.item()
        
        self.log_dict(
            {'loss/val_loss': avg_val_loss,
             'acc/val_acc': avg_val_acc,
             'learning_rate': self.learning_rate},  # 검증 손실, 정확도 및 학습률 기록
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        # 스케줄러에 검증 손실 전달
        self.lr_schedulers().step(avg_val_loss)
        
        if self.configs.get('nni'):
            nni.report_intermediate_result(avg_val_acc)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_accs.clear()
        # 테스트 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴
        
        output = self.model(X)  # 모델을 통해 예측값을 계산
        output = torch.squeeze(output)
        sig_out = F.sigmoid(output)
        self.test_loss = F.binary_cross_entropy(sig_out, y)  

        y_pred = (sig_out >= 0.5).float()
        self.test_acc = (y_pred == y).type_as(y).float().mean()
        
        self.test_accs.append(self.test_acc)        
        return output
    
    def on_test_epoch_end(self):
        # 테스트 에포크가 끝난 후 평균 정확도 계산
        avg_test_acc = torch.mean(torch.tensor(self.test_accs))
        
        self.log_dict({
            'acc/test_acc': avg_test_acc
        }, prog_bar=True)
        
        if self.configs.get('nni'):
            nni.report_final_result(avg_test_acc)  # type: ignore

    def configure_optimizers(self):
        # 옵티마이저와 스케줄러를 설정하는 메서드
        optimizer = optim.Adam(
            self.model.parameters(),  # 모델의 파라미터를 옵티마이저에 전달
            lr=self.learning_rate,    # type: ignore
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',               # 손실이 감소할 때 학습률을 줄임
            factor=0.5,               # 학습률 감소 비율
            patience=5,               # 손실이 감소하지 않을 때 대기 에포크 수
        )

        return {
            'optimizer': optimizer,   # 옵티마이저 반환
            'scheduler': scheduler,   # 학습률 스케줄러 반환
        }
