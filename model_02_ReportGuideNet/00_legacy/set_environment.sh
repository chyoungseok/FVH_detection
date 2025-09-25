# before start set your authority of this file
# chmod +x horovod_setup.sh

#!/bin/bash
set -e

# ==========================================
# 1. Conda 가상환경 생성 및 활성화
# ==========================================
ENV_NAME=horovod_mpi
PYTHON_VERSION=3.10

echo "[INFO] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# ==========================================
# 2. 기본 개발 도구 설치
# ==========================================
echo "[INFO] Installing build essentials (gcc, g++, gfortran, cmake, git)"
conda install -c conda-forge gxx_linux-64 gcc_linux-64 gfortran_linux-64 cmake==3.28 git -y

echo "[CHECK] gcc version:"
$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc --version | head -n 1

echo "[CHECK] cmake version:"
cmake --version | head -n 1

echo "[CHECK] git version:"
git --version

# ==========================================
# 3. MPI 설치 (MPICH)
# ==========================================
echo "[INFO] Installing MPI (mpich)"
conda install -c conda-forge mpich -y

echo "[CHECK] mpirun version:"
mpirun --version | head -n 1

# ==========================================
# 4. NCCL 설치
# ==========================================
echo "[INFO] Installing NCCL"
conda install -c nvidia nccl -y

echo "[CHECK] NCCL libs in conda prefix:"
ls $CONDA_PREFIX/lib | grep nccl || echo "NCCL not found!"

# ==========================================
# 5. PyTorch + CUDA 설치 (CUDA 12.1)
# ==========================================
echo "[INFO] Installing PyTorch with CUDA 12.1"
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "[CHECK] PyTorch GPU availability:"
python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

# ==========================================
# 6. 환경 변수 세팅 + .bashrc 업데이트
# ==========================================
echo "[INFO] Exporting environment variables"

cat <<EOL >> ~/.bashrc

# ===== Added by Horovod setup =====
export PATH=\$CONDA_PREFIX/bin:\$PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH

# CMake (conda env 전용)
export HOROVOD_CMAKE=\$CONDA_PREFIX/bin/cmake

# CUDA Toolkit
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Horovod build flags
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1
# =================================
EOL

# 즉시 반영
source ~/.bashrc

# ==========================================
# 7. Horovod 설치 (GitHub 최신)
# ==========================================
echo "[INFO] Installing Horovod (GitHub latest)"
HOROVOD_CMAKE=$CONDA_PREFIX/bin/cmake \
HOROVOD_WITH_PYTORCH=1 \
pip install --no-cache-dir --force-reinstall git+https://github.com/horovod/horovod.git

# ==========================================
# 8. 설치 확인
# ==========================================
echo "[CHECK] Horovod build status:"
horovodrun --check-build

# ==========================================
# 9. 간단 테스트 코드 실행
# ==========================================
cat <<'PYCODE' > test_hvd.py
import torch
import horovod.torch as hvd

hvd.init()
torch.cuda.set_device(hvd.local_rank())

print(f"[Rank {hvd.rank()} / {hvd.size()}] "
      f"GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
PYCODE

echo "[INFO] Running Horovod test on 2 processes"
horovodrun -np 4 python test_hvd.py
