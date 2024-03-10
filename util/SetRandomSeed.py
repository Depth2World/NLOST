# Set the random see for all possible places to improve the producibility

import os
import numpy as np
import torch
import random

def set_seed(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)  # if you are using multi-GPU.
    np.random.seed(opt.seed)  # Numpy module.
    random.seed(opt.seed)  # Python random module.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # better for cuda>10.2
    os.environ["PYTHONHASHSEED"] = str(opt.seed)

    # if false, improve producibility, else improve the performance
    # torch.set_deterministic(True)
    # torch.backends.cudnn.enabled = False 
    # torch.backends.cudnn.benchmark = False  
    
    print("Random seed for Pytorch, Python, and Numpy of all GPUs are set to : {}".format(opt.seed))
    print("+++++++++++++++++++++++++++++++++++++++++++")

# there are some other methods to set random: https://github.com/pytorch/pytorch/issues/5059
def worker_init(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)