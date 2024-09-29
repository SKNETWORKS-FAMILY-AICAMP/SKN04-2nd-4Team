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

        self.train_losses = []
        self.val_losses = []  # 검증 손실을 저장할 리스트
        self.test_losses = []  # 검증 손실을 저장할 리스트

    def training_step(self, batch, batch_idx):
        # 학습 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴
        
        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.squeeze(output)
        sig_out = F.sigmoid(squeeze_output)
        train_loss = F.binary_cross_entropy(sig_out, y)  # 손실 계산

        # 손실을 배치마다 로그에 기록
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_losses.append(train_loss.cpu().detach().numpy())
        
        return {'train_loss': train_loss}

    def validation_step(self, batch, batch_idx):
        # 검증 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴

        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.squeeze(output)
        sig_out = F.sigmoid(squeeze_output)
        val_loss = F.binary_cross_entropy(sig_out, y)  # 손실 계산

        # 손실을 배치마다 로그에 기록
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_losses.append(val_loss.cpu().detach().numpy())

        return {'val_loss': val_loss}

    def on_validation_epoch_end(self):
        # 검증 에포크가 끝난 후 NNI에 중간 결과로 손실값을 보고
        if self.configs.get('nni'):
            avg_val_loss = self.trainer.callback_metrics.get('val_loss', None).item()
            if avg_val_loss is not None:
                print(f"Reporting val_loss to NNI: {avg_val_loss}")  # 디버깅용 출력
                nni.report_intermediate_result(avg_val_loss)
            else:
                print("val_loss not found in callback_metrics")

    def test_step(self, batch, batch_idx):
        # 테스트 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴

        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.squeeze(output)
        sig_out = F.sigmoid(squeeze_output)
        test_loss = F.binary_cross_entropy(sig_out, y)  # 손실 계산

        # 손실을 배치마다 로그에 기록
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_losses.append(test_loss.cpu().detach().numpy())

        return {'test_loss': test_loss}
    
    def on_test_epoch_end(self):
        # 테스트 에포크가 끝난 후 NNI에 최종 결과로 손실값을 보고
        if self.configs.get('nni'):
            avg_test_loss = self.trainer.callback_metrics.get('test_loss', None)
            if avg_test_loss is not None:
                print(f"Reporting final test_loss to NNI: {avg_test_loss}")
                nni.report_final_result(avg_test_loss)
            else:
                print("test_loss not found in callback_metrics")

    def configure_optimizers(self):
        # 옵티마이저와 스케줄러를 설정하는 메서드
        optimizer = optim.Adam(
            self.model.parameters(),  # 모델의 파라미터를 옵티마이저에 전달
            lr=self.learning_rate,    # 학습률
        )

         # ReduceLROnPlateau 스케줄러 정의
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',               # 손실이 감소할 때를 모니터링
            factor=0.5,               # 학습률을 50%로 감소
            patience=5,               # 손실이 개선되지 않을 때 대기할 에포크 수
            threshold=0.0001,         # 손실 감소를 판정할 기준
            cooldown=0,               # 학습률 감소 후 대기하는 기간
            min_lr=1e-6,              # 학습률의 최저값
        )

        return {
            'optimizer': optimizer,   # 옵티마이저 반환
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # 검증 손실을 모니터링하여 스케줄러 작동
                'interval': 'epoch',    # 에포크 단위로 실행
                'frequency': 1,         # 매 에포크마다 확인
            }
        }
