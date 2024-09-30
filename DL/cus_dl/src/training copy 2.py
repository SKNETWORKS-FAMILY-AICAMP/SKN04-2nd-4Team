import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from sklearn.metrics import classification_report, confusion_matrix
import nni
import numpy as np
import torchmetrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # self.train_accs=[]
        # self.test_accs=[]
        # self.val_accs=[]

    def training_step(self, batch, batch_idx):
        # 학습 단계에서 호출되는 메서드
        
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴
        
        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.squeeze(output)
        sig_out = F.sigmoid(squeeze_output)
        self.train_loss = F.binary_cross_entropy(sig_out, y)  

        y_pred = (sig_out >= 0.5).float()
        self.train_acc = (y_pred==y).float().mean()
        # self.train_accs.append(self.train_acc)

        # 손실과 정확도를 배치마다 로그에 기록
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
        return self.train_loss

    # def on_train_epoch_end(self, *args, **kwargs):
    #     # 학습 에포크가 끝날 때 호출되는 메서드
    #     self.log_dict(
    #         {'loss/train_loss': np.mean([loss.cpu().numpy() for loss in self.train_losses]),
    #          'acc/train_acc': np.mean([acc.cpu().numpy() for acc in self.train_accs])}, 
    #         on_epoch=True,
    #         prog_bar=True,  # 진행 막대에 표시
    #         logger=True,    # 로그에 기록
    #     )

    def on_train_epoch_end(self, *args, **kwargs):
        # 훈련 에포크가 끝날 때마다 평균 훈련 정확도와 손실을 로그로 기록
        avg_train_acc = self.trainer.callback_metrics.get('train_acc')
        avg_train_loss = self.trainer.callback_metrics.get('train_loss')

        if avg_train_acc is not None:
            self.log('avg_train_acc', avg_train_acc)
        
        if avg_train_loss is not None:
            self.log('avg_train_loss', avg_train_loss)

        print(f"Epoch {self.current_epoch} - Train Acc: {avg_train_acc}, Train Loss: {avg_train_loss}")

        # if self.configs.get('nni'):
        #     # 자동으로 기록된 정확도를 NNI에 보고
        #     avg_train_acc = self.trainer.callback_metrics.get('train_acc', None)
        #     if avg_train_acc is not None:
        #         avg_train_acc_value = avg_train_acc.item()
        #         print(f"Reporting train_acc to NNI: {avg_train_acc_value}")  # 디버깅용 출력
        #         nni.report_intermediate_result(avg_train_acc_value)
        #     else:
        #         print("train_acc not found in callback_metrics")  # 값이 없을 경우 출력

        # 로그 기록
        # self.log_dict({
        #     'loss/train_loss': avg_train_loss,  # 에포크별 손실 로그
        #     'acc/train_acc': avg_train_acc      # 에포크별 정확도 로그
        # }, on_epoch=True, prog_bar=True, logger=True)

        # 에포크가 끝나면 리스트 초기화 (다음 에포크를 위해)
        # self.train_losses.clear()
        # self.train_accs.clear()
    
    def validation_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     self.val_accs.clear()
        # 검증 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴

        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.squeeze(output)
        sig_out = F.sigmoid(squeeze_output)
        self.val_loss = F.binary_cross_entropy(sig_out, y)  

        y_pred = (sig_out >= 0.5).float()
        self.val_acc = (y_pred==y).float().mean()
        # self.val_accs.append(self.val_acc)

        # 손실과 정확도를 배치마다 로그에 기록
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'val_loss': self.val_loss, 'val_acc': self.val_acc}
    
    # def on_validation_epoch_end(self):
    #     # 검증 에포크가 끝날 때 호출되는 메서드
    #     self.log_dict(
    #         {'loss/train_loss': np.mean([loss.cpu().numpy() for loss in self.val_losses]),
    #          'acc/val_acc':np.mean([acc.cpu().numpy() for acc in self.val_accs]),
    #          'learning_rate': self.optimizers().param_groups[0]['lr'] },  # 학습률도 로그에 기록 # type: ignore
    #         on_epoch=True,
    #         prog_bar=True,  # 진행 막대에 표시
    #         logger=True,    # 로그에 기록
    #     )
    #     if self.configs.get('nni'):
    #         nni.report_intermediate_result(np.mean([acc.cpu().numpy() for acc in self.val_accs])) # type: ignore

    # def on_validation_epoch_end(self):
    #     if self.configs.get('nni'):
    #         avg_val_acc = self.trainer.callback_metrics['val_acc'].cpu().numpy()
    #         nni.report_intermediate_result(avg_val_acc)  # NNI에 중간 결과 보고

    def on_validation_epoch_end(self):
        if self.configs.get('nni'):
            # 자동으로 기록된 정확도를 NNI에 보고
            avg_val_acc = self.trainer.callback_metrics.get('val_acc', None)
            if avg_val_acc is not None:
                avg_val_acc_value = avg_val_acc.item()
                print(f"Reporting val_acc to NNI: {avg_val_acc_value}")  # 디버깅용 출력
                nni.report_intermediate_result(avg_val_acc_value)
            else:
                print("val_acc not found in callback_metrics")  # 값이 없을 경우 출력

    def test_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     self.test_accs.clear()
        # 테스트 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴

        output = self.model(X)  # 모델을 통해 예측값을 계산
        squeeze_output = torch.squeeze(output)
        sig_out = F.sigmoid(squeeze_output)
        self.test_loss = F.binary_cross_entropy(sig_out, y)  

        y_pred = (sig_out >= 0.5).float()
        self.test_acc = (y_pred==y).float().mean()
        # self.test_accs.append(self.test_acc)     

        # 손실과 정확도를 배치마다 로그에 기록
        self.log('test_loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True) 

        return {'test_loss': self.test_loss, 'test_acc': self.test_acc}
    
    # def on_test_epoch_end(self):
    #     # if self.configs.get('nni'):
    #     #     nni.report_final_result(np.mean([acc.cpu().numpy() for acc in self.test_accs])) # type: ignore
    #     if self.configs.get('nni'):
    #         avg_test_acc = self.trainer.callback_metrics['test_acc'].cpu().numpy()  # 자동 기록된 값을 사용
    #         nni.report_final_result(avg_test_acc)  # NNI에 최종 결과 보고

    def on_test_epoch_end(self):
        if self.configs.get('nni'):
            # 자동으로 기록된 테스트 정확도를 보고
            avg_test_acc = self.trainer.callback_metrics.get('test_acc', None)
            if avg_test_acc is not None:
                avg_test_acc_value = avg_test_acc.item()
                print(f"Reporting final test_acc to NNI: {avg_test_acc_value}")
                nni.report_final_result(avg_test_acc_value)
            else:
                print("test_acc not found in callback_metrics")


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
        # CyclicLR 스케줄러 정의
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer, 
        #     base_lr=0.0001,  # 최소 학습률
        #     max_lr=0.01,   # 최대 학습률
        #     step_size_up=1000,  # 1000 스텝 동안 학습률이 올라감
        #     mode='triangular2'  # 학습률이 삼각형 모양으로 변동, 매 사이클마다 절반으로 줄어듬
        # )

        return {
            'optimizer': optimizer,   # 옵티마이저 반환
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }
        # return {
        #     'optimizer': optimizer,   # 옵티마이저 반환
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'interval': 'step',
        #         'frequency': 1,
        #     }
        # }
