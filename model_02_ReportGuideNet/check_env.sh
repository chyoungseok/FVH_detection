#!/bin/bash
echo "===== GPU 정보 ====="
nvidia-smi

echo ""
echo "===== CUDA 설치 경로 ====="
which nvcc || echo "nvcc not found in PATH"
nvcc --version 2>/dev/null || echo "nvcc not available"

echo ""
echo "===== Conda Python 버전 ====="
python -V

echo ""
echo "===== PyTorch 설치 여부 ====="
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)" 2>/dev/null || echo "PyTorch not installed"

echo ""
echo "===== GCC/G++ 버전 ====="
which gcc && gcc --version | head -n 1
which g++ && g++ --version | head -n 1
