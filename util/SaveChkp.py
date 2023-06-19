import sys
import torch
import torch.distributed as dist
import util.SetDistTrain as utils


def save_checkpoint(n_iter, epoch, model, optimer, file_path):
    """params:
    epcoh: the current epoch
    n_iter: the current iter
    model: the model dict
    optimer: the optimizer dict
    """
        
    state = {}
    state["n_iter"] = n_iter
    state["epoch"] = epoch
    state["lr"] = optimer.param_groups[0]["lr"]
    state["state_dict"] = model.state_dict()
    state["optimizer"] = optimer.state_dict()
    
    if utils.is_main_process():
        torch.save(state, file_path)