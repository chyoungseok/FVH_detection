"""
FHV(+) / FHV(-) 이진 분류 (slice-level) - ResNet18 통합 스크립트
- 데이터 트리: data/subjectX/{slice_img.npy, slice_label.npy}
- 입력: 각 subject의 slice_img.npy: (N_slices, 672, 672), slice_label.npy: (N_slices,) 또는 (N_slices,1)
- 출력: 학습된 모델(.pt), 테스트 리포트(.txt), 콘솔 메트릭

학습 전략
- subject-wise 정보는 사용하지 않음(요청사항 반영)
- slice-wise 전체 concat → 7:3 stratified split
- 클래스 불균형: 음성(0) 언더샘플링 (양성:음성 = 1:R)
- 손실: BCEWithLogitsLoss(+ pos_weight 옵션)
- 추론 함수: infer_one_slice() (확률 반환)
"""

'''
python main_v01.py --data_root ./01_data/04_flair_preproc_slices --out_dir ./runs/fhv_resnet18_with_cam --neg_ratio 1.0 --test_ratio 0.3 --val_ratio 0.1 --epochs 100 --batch_size 16 --lr 1e-3 --lr_scheduler plateau --plateau_mode min --plateau_factor 0.5 --plateau_patience 2 --weight_pos 1.2 --amp --num_workers 0 --export_cam --cam_layer layer4 --cam_target pos
'''

import os, argparse

import torch
from torchsummary import summary

from modules.data_utils import *
from modules.model_utils import *

# =========================================================
# Argparse helpers
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FHV slice-level binary classification (ResNet18)"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="subject 폴더들이 모여있는 루트 (subject/slice_img.npy, slice_label.npy)")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="모델/리포트 저장 폴더")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="언더샘플링 비율: 양성 1에 대해 음성 R (기본 1:1)")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="train:test 비율 (기본 7:3 → 0.3)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="val ratio split from TRAIN (default 0.1)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_pos", type=float, default=1.0, help="BCE pos_weight (양성 가중치)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="자동 혼합 정밀도 사용")
    parser.add_argument("--select_N", type=int, default=100, help="처음 몇개의 데이터를 사용할 것인가")
    parser.add_argument("--show_model_summary", type=lambda x: (str(x).lower() == 'true'), default=True, help="show model structure")

    # LR scheduler options
    parser.add_argument("--lr_scheduler", type=str, default="none",
                        choices=["none", "step", "multistep", "cosine", "plateau"],
                        help="learning rate scheduler type")
    parser.add_argument("--lr_step_size", type=int, default=5, help="StepLR: step_size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="StepLR/MultiStepLR: gamma")
    parser.add_argument("--lr_milestones", type=str, default="",
                        help="MultiStepLR: comma-separated epochs, e.g., 10,15")
    parser.add_argument("--lr_T_max", type=int, default=15, help="CosineAnnealingLR: T_max (epochs)")
    parser.add_argument("--lr_eta_min", type=float, default=0.0, help="CosineAnnealingLR: eta_min")
    parser.add_argument("--plateau_mode", type=str, default="min", choices=["min", "max"],
                        help="ReduceLROnPlateau mode")
    parser.add_argument("--plateau_factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--plateau_patience", type=int, default=3, help="ReduceLROnPlateau patience (epochs)")

    # CAM options
    parser.add_argument("--export_cam", action="store_true", help="Export Grad-CAM overlays for TEST set")
    parser.add_argument("--cam_layer", type=str, default="layer4", choices=["layer4", "layer3"], help="Target layer for CAM")
    parser.add_argument("--cam_target", type=str, default="pos", choices=["pos", "neg"], help="Which logit to explain (pos: z, neg: -z)")
    parser.add_argument("--cam_dir", type=str, default=None, help="Output dir for CAM PNGs (default: out_dir/cam_test)")
    return parser

def parse_args(cli_args=None):
    parser = build_arg_parser()
    return parser.parse_args(cli_args)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    
    dl_train, dl_val, dl_test = set_dataloader(data_root=args.data_root,
                                               select_N=args.select_N,
                                               neg_ratio=args.neg_ratio,
                                               test_ratio=args.test_ratio,
                                               val_ratio=args.val_ratio,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               out_dir=args.out_dir,
                                               seed=args.seed)
    
    
    # 5) 모델/손실/최적화
    # 1채널 입력(흑백 FLAIR slice)에 맞춘 ResNet18 모델 생성
    # - conv1: in_channels=1 (기본은 3, RGB용)
    # - fc: out_features=1 (이진 분류 → 단일 logit 출력)
    # - to(device): 모델 파라미터를 GPU 또는 CPU로 이동 (입력 텐서와 같은 디바이스여야 함)
    model = build_resnet18_1ch(num_classes=1).to(device)

    # 클래스 불균형 보정을 위한 가중치
    # - BCEWithLogitsLoss에서 양성 클래스 손실에 곱해짐
    # - 원본 데이터(불균형)를 그대로 쓰면 pos_weight ≈ N_neg / N_pos 로 설정
    # - BUT: 이미 1:1 언더샘플링을 적용했으므로 pos_weight=1.0 으로 두면 충분
    pos_weight = torch.tensor([args.weight_pos], device=device)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.show_model_summary:
        summary(model, input_size=(1, 672, 672))

if __name__=='__main__':
    main()