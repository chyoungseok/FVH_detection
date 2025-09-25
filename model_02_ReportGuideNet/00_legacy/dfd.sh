conda create -n rgNet python=3.10 -y
conda activate rgNet

conda install -c conda-forge cmake=3.28 -y
conda install -c conda-forge gcc_linux-64 gxx_linux-64 gfortran_linux-64 -y
conda install -c anaconda gcc_linux-64 gxx_linux-64 gfortran_linux-64 -y
conda install -c conda-forge mpich -y

conda install git -y
conda install -c nvidia nccl -y

pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"



export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITHOUT_MPI=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_CMAKE=$CONDA_PREFIX/bin/cmake
export HOROVOD_GPU_OPERATIONS=NCCL

pip install packaging typing-extensions
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


pip install --no-cache-dir --force-reinstall git+https://github.com/horovod/horovod.git
horovodrun -np 4 python test_hvd.py
horovodrun --check-build


/zdisk/users/ext_user_03/miniconda3/envs/rgNet/bin/cmake
/zdisk/users/ext_user_03/miniconda3/envs/horo1/bin/cmake


## 2.
conda install -c conda-forge cmake=3.28 -y
conda install -c nvidia nccl -y
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda install git -y
# export HOROVOD_GPU_OPERATIONS=NCCL
# export HOROVOD_WITH_PYTORCH=1
pip install --no-cache-dir --force-reinstall --verbose git+https://github.com/horovod/horovod.git 2>&1 | tee build_hvd.log


# 3

# PyTorch + CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# nccl
conda install -c nvidia nccl -y

# git
conda install git -y

# cmake
conda install -c conda-forge cmake=3.28 -y

# horovod
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_CMAKE=$CONDA_PREFIX/bin/cmake

pip install --no-cache-dir git+https://github.com/horovod/horovod.git




##

conda create -n reportnet python=3.10 -y
conda activate reportnet


pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.2 scikit-learn tqdm
pip install numpy==1.26.4
python -c "import torch; import numpy as np; print('Torch:', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('NumPy:', np.__version__)"


conda install -c conda-forge openmpi nccl cmake gxx_linux-64 -y
export CC=$(which gcc)
export CXX=$(which g++)
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_GPU_OPERATIONS=NCCL
