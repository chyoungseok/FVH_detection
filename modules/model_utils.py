import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import models
from torchsummary import summary
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.metrics import classification_report

from modules.plot_utils import save_eval_plots

# =========================================================
# 모델: ResNet18 (1채널 입력)
# =========================================================
def build_resnet_1ch(model_depth: int = 18, num_classes: int = 1) -> nn.Module:
    """
    torchvision resnet18을 흑백 의료영상(slice-level 이진 분류)에 맞게 수정한 모델 생성 함수.
    - conv1: 기본 ResNet은 RGB(3채널) 입력용 → 의료영상은 1채널이므로 in_channels=1로 교체
    - fc: 기본 ResNet은 ImageNet 1000클래스용 → 이진 분류이므로 출력=1(logit)로 교체
    
    ** 기존 resnet 구조를 확인하고 싶은 경우
    from torchsummary import summary
    from torchvision import models

    model_resnet = models.resnet18(weights=None).to(device)
    summary(model_resnet, input_size=(3, 672, 672), device=device)
    print(model_resnet)
    
    """
    # 기본 ResNet18 불러오기 (사전학습 weight 사용하지 않음: 의료영상 특성 상 ImageNet pretrain과 도메인 차이가 큼)
    if model_depth == 18:
        model = models.resnet18(weights=None)
    elif model_depth == 34:
        model = models.resnet34(weights=None)
    elif model_depth == 50:
        model = models.resnet50(weights=None)
    elif model_depth == 101:
        model = models.resnet101(weights=None)
    elif model_depth == 152:
        model = models.resnet152(weights=None)

    # 첫 번째 convolution 레이어 교체
    # - 원래 정의: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # - 여기서 3은 RGB 입력 채널 수 → 흑백 영상은 채널이 1이므로 in_channels=1로 변경
    # - out_channels=64는 ResNet 표준 구조를 맞추기 위한 값 (바꾸면 뒤 레이어와 mismatch 발생)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 마지막 fully-connected 레이어 교체
    # - 원래 정의: nn.Linear(512, 1000) → ImageNet 1000-class
    # - 여기서는 이진 분류(FHV(+), FHV(-)) → logit 하나만 출력하도록 Linear(..., 1)
    # - BCEWithLogitsLoss와 함께 사용하면, 이 1개의 logit에 sigmoid를 씌워 확률[0,1]로 변환
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def model_loss_optimizer_resnet(resnet_depth, device, weight_pos, lr, show_model_summary):
    # 1채널 입력(흑백 FLAIR slice)에 맞춘 ResNet18 모델 생성
    # - conv1: in_channels=1 (기본은 3, RGB용)
    # - fc: out_features=1 (이진 분류 → 단일 logit 출력)
    # - to(device): 모델 파라미터를 GPU 또는 CPU로 이동 (입력 텐서와 같은 디바이스여야 함)
    model = build_resnet_1ch(model_depth=resnet_depth, num_classes=1).to(device)

    # 클래스 불균형 보정을 위한 가중치
    # - BCEWithLogitsLoss에서 양성 클래스 손실에 곱해짐
    # - 원본 데이터(불균형)를 그대로 쓰면 pos_weight ≈ N_neg / N_pos 로 설정
    # - BUT: 이미 1:1 언더샘플링을 적용했으므로 pos_weight=1.0 으로 두면 충분
    pos_weight = torch.tensor([weight_pos], device=device)

    # 손실 함수: BCE with Logits
    # - 모델 출력(logit)을 그대로 입력 (sigmoid 따로 쓰지 말 것)
    # - 내부에서 sigmoid + binary cross entropy를 안정적으로 계산
    # - pos_weight 옵션을 통해 양성 클래스 손실 기여도를 조절
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 옵티마이저: Adam
    # - model.parameters(): 모델의 학습 가능한 모든 파라미터를 업데이트 대상으로 지정
    # - lr: 학습률, 보통 1e-3에서 시작
    # - weight_decay: L2 정규화 항 (가중치가 과도하게 커지는 것을 막아 과적합 방지)
    #   → 너무 크면 underfitting, 너무 작으면 효과 없음. 일반적으로 1e-4 권장
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if show_model_summary:
        summary(model, input_size=(1, 672, 672))

    return model, criterion, optimizer

def init_lr_scheduler(
    scheduler_name: str,
    optimizer,
    lr_step_size: int,
    lr_gamma: float,
    lr_milestones: str,
    epochs: int,
    lr_T_max: int,
    lr_eta_min: float,
    plateau_mode: str,
    plateau_factor: float,
    plateau_patience: int,
):
    """
    학습률 스케줄러 초기화 유틸.
    - 왜 필요한가?
      · 고정 lr로 학습하면 특정 시점 이후 수렴이 더뎌질 수 있음 → lr을 점진적으로 줄여 일반화/수렴을 돕는다.
    - 어떤 스케줄러를 언제 쓰나?
      · "step"      : 일정 epoch 간격으로 lr *= gamma (단순·직관적)
      · "multistep" : 지정한 epoch 리스트에서 lr *= gamma (전형적인 분기점 학습)
      · "cosine"    : 코사인 곡선으로 lr 감소(마지막에 eta_min으로 수렴; warm-restart 없이 단일 주기)
      · "plateau"   : 모니터하는 지표(val_loss 등)가 개선되지 않으면 lr *= factor (자동 감쇠)
      · "none"      : 스케줄러 미사용 (None 반환)

    ⚠️ 주의: 호출 타이밍이 스케줄러마다 다름
      - StepLR/MultiStepLR/CosineAnnealingLR  → epoch 말미에  scheduler.step()
      - ReduceLROnPlateau                    → epoch 말미에  scheduler.step(val_loss)  # 모니터 값 필수

    Args:
        scheduler_name   : {"none","step","multistep","cosine","plateau"}
                           (기존 코드의 lr_scheduler와 이름이 겹쳐 혼동되므로 명확히 변경)
        optimizer        : torch.optim.Optimizer 인스턴스
        lr_step_size     : StepLR에서 몇 epoch마다 감쇠할지 (예: 5 → 매 5 epoch마다)
        lr_gamma         : 감쇠 비율 (예: 0.1 → 10분의 1로)
        lr_milestones    : MultiStepLR용, "10,20,30" 형태의 콤마 구분 문자열
        epochs           : 총 학습 epoch 수 (기본값 생성/검증 시 참조)
        lr_T_max         : CosineAnnealingLR 주기 길이(에폭). 0/음수면 epochs 전체를 주기로 사용
        lr_eta_min       : CosineAnnealingLR 최저 학습률
        plateau_mode     : {"min","max"} · val_loss 모니터 시 "min", val_auc 모니터 시 "max"
        plateau_factor   : ReduceLROnPlateau에서 lr를 곱할 감쇠 계수(예: 0.5 → 절반)
        plateau_patience : 개선 없다고 판단하기 전까지 기다릴 epoch 수

    Returns:
        torch.optim.lr_scheduler 객체 또는 None ("none"인 경우)
    """
    # 이름 충돌/가독성 차원에서 torch.optim.lr_scheduler는 _lrs 별칭으로 사용
    scheduler = None

    if scheduler_name == "none" or scheduler_name is None:
        # 스케줄러 미사용: 호출부에서 scheduler가 None인지 체크하고 step()을 부르지 않도록 처리
        return None

    if scheduler_name == "step":
        # 매 lr_step_size epoch마다 lr = lr * lr_gamma
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    elif scheduler_name == "multistep":
        # "10,15" 같은 문자열을 파싱 → [10, 15]
        milestones = [int(m) for m in lr_milestones.split(",") if m.strip().isdigit()]
        # 사용자가 안 넣었을 때의 안전한 디폴트(대략 60%, 85% 지점에서 감쇠)
        if len(milestones) == 0:
            milestones = [max(1, int(epochs * 0.6)), max(2, int(epochs * 0.85))]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)

    elif scheduler_name == "cosine":
        # T_max: 코사인 한 주기의 길이(에폭). 0/음수면 전체 epochs를 한 주기로 사용.
        T_max = lr_T_max if lr_T_max and lr_T_max > 0 else epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_eta_min)

        # 💡 Tip: warmup이 필요하면 별도 Warmup 스케줄러를 앞단에 구성하거나
        # OneCycleLR 같은 대안을 고려.

    elif scheduler_name == "plateau":
        # 모니터하는 지표가 개선되지 않을 때 lr *= plateau_factor 로 감소
        # mode="min": 낮을수록 좋다(예: val_loss), mode="max": 높을수록 좋다(예: val_auc)
        # patience: 개선이 없다고 판단하기 전까지 기다릴 epoch 수
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
        )

        # ⚠️ 호출부에서 scheduler.step(val_metric) 형태로 '모니터 값'을 반드시 전달해야 함.
        # 예) val_loss를 모니터한다면:  scheduler.step(val_loss)

    else:
        # 오타/미지원 옵션 방지
        raise ValueError(f"Unknown scheduler_name: {scheduler_name}")

    return scheduler

def set_scaler(amp, device):
    """
    Initialize a gradient scaler for Automatic Mixed Precision (AMP) training.

    Args:
        amp (bool): If True, enable AMP support.
        device (str): Target device string ("cuda" or "cpu").

    Returns:
        torch.cuda.amp.GradScaler or None:
            - If amp=True and device="cuda": returns a GradScaler object
            - Otherwise: returns None (AMP not available on CPU)

    Notes:
        - AMP mixes FP16 and FP32 operations to improve speed and reduce memory usage,
          but may cause gradient underflow/overflow.
        - GradScaler dynamically scales the loss to stabilize FP16 training.
        - Usage in training loop:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    """
    
    scaler = torch.amp.GradScaler('cuda') if (amp and device == "cuda") else None
    
    return scaler

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Run one training epoch on the given dataset.

    Args:
        model (torch.nn.Module): 학습할 모델 (예: ResNet18 변형)
        loader (DataLoader): 학습 데이터셋을 배치 단위로 제공하는 DataLoader
        criterion (torch.nn.Module): 손실 함수 (BCEWithLogitsLoss 등)
        optimizer (torch.optim.Optimizer): 모델 파라미터를 업데이트할 옵티마이저 (Adam, SGD 등)
        device (str): 실행할 디바이스 ("cuda" 또는 "cpu")
        scaler (torch.cuda.amp.GradScaler, optional): 
            AMP(혼합 정밀) 학습 시 gradient scaling을 위한 GradScaler.
            None이면 일반 학습 수행.

    Returns:
        float: epoch 전체 데이터셋에 대한 평균 손실 값

    Notes:
        - 학습 단계는 다음 순서로 진행:
            1. gradient 초기화 (zero_grad)
            2. forward pass → loss 계산
            3. backward pass (gradient 계산)
            4. optimizer step (파라미터 업데이트)
            5. 배치 손실 누적
        - AMP 사용 시, autocast + GradScaler를 통해 FP16 연산의 안정성을 확보.
        - 반환되는 값은 전체 데이터셋 평균 손실로, 학습 로그 모니터링에 활용됨.
    """
    model.train()  # 모델을 학습 모드로 설정 (Dropout, BatchNorm 등 훈련 전용 동작 활성화)
    total_loss = 0.0  # epoch 전체 손실을 누적할 변수

    # DataLoader에서 배치 단위로 이미지와 라벨을 꺼냄
    for imgs, labs in loader:
        # 배치 데이터를 GPU/CPU로 이동
        imgs = imgs.to(device, non_blocking=True)  
        labs = labs.to(device, non_blocking=True).view(-1, 1)  # (B,1) 형태로 맞춤

        # 이전 step에서 남아있던 gradient 초기화
        optimizer.zero_grad(set_to_none=True)

        # 혼합 정밀(AMP) 학습을 사용하는 경우
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)         # forward pass (logit 출력)
                loss = criterion(logits, labs)  # 손실 계산
            scaler.scale(loss).backward()   # scaled gradient 계산
            scaler.step(optimizer)          # optimizer step (스케일링 반영)
            scaler.update()                 # 스케일링 인자 갱신
        else:
            # AMP를 사용하지 않는 일반 학습 루프
            logits = model(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()

        # 배치 손실 누적
        total_loss += loss.item() * imgs.size(0)

    # 전체 데이터셋 평균 손실 반환
    return total_loss / len(loader.dataset)

@torch.no_grad() # 평가 모드에서는 gradient를 계산하지 않음 (메모리/속도 최적화)
def evaluate(model, loader, device, criterion: Optional[nn.Module] = None):
    """
    모델을 검증/테스트 모드에서 평가하는 함수.

    Args:
        model (torch.nn.Module): 학습된 모델
        loader (DataLoader): 검증 또는 테스트 데이터셋을 배치 단위로 제공하는 DataLoader
        device (str): 실행 장치 ("cuda" 또는 "cpu")
        criterion (nn.Module, optional): 손실 함수. 제공 시 평균 손실도 계산하여 반환

    Returns:
        dict: 평가 지표와 관련 데이터가 담긴 딕셔너리
            - "acc"  : Accuracy (정확도)
            - "prec" : Precision (정밀도)
            - "rec"  : Recall (재현율)
            - "f1"   : F1-score
            - "auc"  : AUC (Area Under ROC Curve)
            - "cm"   : Confusion Matrix (혼동 행렬)
            - "y_prob": 모델의 예측 확률 (numpy array)
            - "y_true": 실제 라벨 (numpy array)
            - "loss" : 평균 손실 (criterion이 주어졌을 경우만 포함)
    """

    model.eval()
    all_probs, all_labels = [], []  # 배치별 확률과 라벨을 모아둘 리스트
    total_loss, total_n = 0.0, 0    # 손실 합계와 샘플 수 (평균 손실 계산용)

    # DataLoader를 통해 배치 단위로 추론
    for imgs, labs in loader:
        # 입력 이미지와 라벨을 디바이스로 이동
        imgs = imgs.to(device, non_blocking=True)
        labs_f = labs.to(device, non_blocking=True).view(-1, 1)  # (B,) → (B,1) 변환 (손실 계산 호환용)

        # forward pass (gradient 없음)
        logits = model(imgs)

        # 손실 함수를 제공받은 경우 → 손실도 계산하여 누적
        if criterion is not None:
            loss = criterion(logits, labs_f)
            total_loss += loss.item() * imgs.size(0)  # 배치 손실 합계
            total_n += imgs.size(0)                   # 전체 샘플 수 누적

        # sigmoid → 확률로 변환, shape (B,)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.append(probs)           # 배치 확률 저장
        all_labels.append(labs.numpy())   # 배치 라벨 저장 (CPU tensor → numpy)

    # 배치별 리스트를 하나의 array로 결합
    y_prob = np.concatenate(all_probs)           # 예측 확률
    y_true = np.concatenate(all_labels).astype(int)  # 실제 라벨
    y_pred = (y_prob >= 0.5).astype(int)         # 0.5 임계값으로 이진 분류 결과 생성

    # 평가 지표 계산
    acc  = accuracy_score(y_true, y_pred)                     # 정확도
    prec = precision_score(y_true, y_pred, zero_division=0)   # 정밀도
    rec  = recall_score(y_true, y_pred, zero_division=0)      # 재현율
    f1   = f1_score(y_true, y_pred, zero_division=0)          # F1-score

    # ROC AUC 계산 (단, 양성/음성이 모두 존재하지 않으면 예외 발생 → NaN 처리)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)

    # 결과 딕셔너리 구성
    out = {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "y_prob": y_prob,
        "y_true": y_true
    }

    # 손실 포함 여부 (criterion을 제공했을 때만)
    if criterion is not None and total_n > 0:
        out["loss"] = total_loss / total_n  # 평균 손실

    return out

def train_loop(
    out_dir,
    device,
    epochs,
    dl_train,
    dl_val,
    model,
    criterion,
    optimizer,
    scheduler,
    lr_scheduler,
    scaler,
    args,
    ):
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_auc": [], "val_auc": [], "lr": []}
    best_f1, best_ckpt_path, best_metrics = -1.0, os.path.join(out_dir, "best_resnet18.pt"), None

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, criterion, optimizer, device, scaler=scaler)
        tr_metrics = evaluate(model, dl_train, device, criterion)
        val_metrics = evaluate(model, dl_val, device, criterion)
        history["train_loss"].append(tr_metrics.get("loss", tr_loss))
        history["val_loss"].append(val_metrics.get("loss", float('nan')))
        history["train_acc"].append(tr_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_auc"].append(tr_metrics.get("auc", float('nan')))
        history["val_auc"].append(val_metrics.get("auc", float('nan')))
        
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        print(f"[Epoch {epoch:02d}] lr={current_lr:.3e} | tr_loss={tr_metrics.get('loss', tr_loss):.4f} | "
              f"val_loss={val_metrics.get('loss', float('nan')):.4f} | VAL: ACC={val_metrics['acc']:.3f} "
              f"PRE={val_metrics['prec']:.3f} REC={val_metrics['rec']:.3f} F1={val_metrics['f1']:.3f} "
              f"AUC={val_metrics['auc']:.3f}")

        if scheduler is not None:
            if lr_scheduler == "plateau":
                scheduler.step(val_metrics.get('loss', None))  # val loss 기준
            else:
                scheduler.step()
        
        # Save best by validation F1
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
                "seed": args.seed
            }, best_ckpt_path)
            
    
    return history, best_ckpt_path, best_metrics

def evaluation_on_test(best_ckpt_path, device, model, dl_test, criterion, out_dir):
    ckpt = torch.load(
        best_ckpt_path,
        map_location=device if isinstance(device, str) else None,
        weights_only=False 
        ) if os.path.exists(best_ckpt_path) else None
    
    if ckpt and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    test_metrics = evaluate(model, dl_test, device, criterion)
    
    # Save predictions and metrics
    np.save(os.path.join(out_dir, "test_y_true.npy"), test_metrics["y_true"]) 
    np.save(os.path.join(out_dir, "test_y_prob.npy"), test_metrics["y_prob"]) 

    # Plots: ROC, PR, Confusion Matrix
    save_eval_plots(test_metrics["y_true"], test_metrics["y_prob"], out_dir, prefix="test", thr=0.5)
    
    return test_metrics


