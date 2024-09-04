import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_optim(optim_name, model, learning_rate, momentum):
    """
    function to set optimizer for training
    
    Parameters:
        optim_name (str): optimizer name
        model (nn.Module): model for training
        learning_rate (float): learning rate for training
        momentum (float): momentum for training ()
    
    """
    if optim_name == "Adam":
        optim = optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_name == "SGD":
        optim = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise RuntimeError("optimizer is not selected")


def set_device(gpunum=0):
    if torch.cuda.is_available():
        if gpunum == 0:
            device = torch.device("cuda")
        elif gpunum == 1:
            device = torch.device("cuda:1")
        elif gpunum == 2:
            device = torch.device("cuda:2")
        elif gpunum == 3:
            device = torch.device("cuda:3")
        else:
            raise RuntimeError("GPU do not exist")
    else:
        device = torch.device("cpu")
        raise RuntimeError("CUDA is not available. GPU is not available.")
    return device


def set_criteria(name):
    if name == 'CrossEntropyLoss':
        criteria = nn.CrossEntropyLoss()
    elif name == 'MSELoss':
        criteria = nn.MSELoss()
    elif name == 'BCELoss':
        criteria == nn.BCELoss()
    else:
        criteria = nn.CrossEntropyLoss()


# def tensorboad(dataset, optim):
#     from torch.utils.tensorboard import SummaryWriter
#     from datetime import datetime
    
#     # 現在の日時を取得してフォルダ名を生成
#     current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
#     log_dir = f'runs/{dataset+"-"+optim+"-"+current_time}'

#     # TensorBoard writerの初期化
#     writer = SummaryWriter(log_dir)
#     return writer