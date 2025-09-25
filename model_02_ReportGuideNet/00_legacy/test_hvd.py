import torch
import horovod.torch as hvd

hvd.init()
torch.cuda.set_device(hvd.local_rank())

print(f"[Rank {hvd.rank()} / {hvd.size()}] "
      f"GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")


# horovodrun -np 4 python test_hvd.py
