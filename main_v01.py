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

import os, argparse

import torch

from modules.data_utils import *

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
    
    # 1) 데이터 로드
    X, y = load_all_slices_from_tree(args.data_root, args.select_N)
    print(f"[Loaded] total={len(y)} | pos={(y==1).sum()} | neg={(y==0).sum()}")
    
    # 2) 음성 언더샘플링 (요청: FHV(-)을 적절히 랜덤 선택)
    X_bal, y_bal = undersample_negatives(X, y, neg_ratio=args.neg_ratio)
    print(f"[Balanced] total={len(y_bal)} | pos={(y_bal==1).sum()} | neg={(y_bal==0).sum()} (neg_ratio={args.neg_ratio})")
    
    # 3) train/test split then train/val split
    (Xtr_full, ytr_full), (Xte, yte) = train_test_split_stratified(X_bal, y_bal, test_ratio=args.test_ratio, seed=args.seed)
    (Xtr, ytr), (Xval, yval) = train_val_split_stratified(Xtr_full, ytr_full, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[Split] train={len(ytr)} | val={len(yval)} | test={len(yte)} (test_ratio={args.test_ratio}, val_ratio={args.val_ratio})")
    np.save(os.path.join(args.out_dir, "test_data.npy"), Xte)
    
    # 4) Datasets & Dataloaders
    ds_train = SliceDataset(Xtr, ytr)
    ds_val   = SliceDataset(Xval, yval)
    ds_test  = SliceDataset(Xte, yte)
    
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=False,
                          persistent_workers=False if args.num_workers == 0 else True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False,
                          persistent_workers=False if args.num_workers == 0 else True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False,
                          persistent_workers=False if args.num_workers == 0 else True)
