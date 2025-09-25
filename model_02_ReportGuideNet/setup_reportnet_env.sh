#!/bin/bash
# ReportGuidedNet 환경 세팅 스크립트 (Horovod 없음)
# 실행 방법: bash setup_reportnet_env.sh

# ENV_NAME="reportnet"

# echo "[1] Conda 환경 생성 (Python 3.10)"
# conda create -n $ENV_NAME python=3.10 -y
# conda activate $ENV_NAME

echo "[2] PyTorch + CUDA 설치 (CUDA 12.1 wheel, 안정 버전)"
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 \
  --extra-index-url https://download.pytorch.org/whl/cu121

echo "[3] 주요 라이브러리 설치"
pip install transformers==4.37.2 scikit-learn tqdm pandas matplotlib seaborn sentencepiece

echo "[4] NumPy 호환 버전 고정"
pip install numpy==1.26.4

echo "[5] 설치 확인"
python -c "import torch, numpy; \
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available()); \
print('NumPy:', numpy.__version__)"
