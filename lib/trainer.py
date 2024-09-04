import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tdqm(i, *args, **kwargs):
        return i


def compare_accuracy(top1_acc, acc):
    if top1_acc < acc:
        newacc = acc
    else: 
        newacc = top1_acc
    return newacc


def train(model, device, optim, loader, criteria):
    """
    誤差逆伝播で学習を行う関数。
    
    Parameters:
        model: 学習したいモデル
        device: GPU
        optim: 定義したオプティマイザー
        loader: 定義したデータローダー
        criteria: 損失関数        
    """
    model.train()

    for img, lbl in tqdm(loader, desc="Train"):
        optim.zero_grad()
        
        img = img.to(device)
        lbl = lbl.to(device)
        out = model.forward(img)
        
        loss = criteria(out, lbl)
        loss.backward()
        optim.step()


def test(model, device, loader):
    """
    学習したモデルの精度を評価する関数
    
    Parameters:
        model: 学習したいモデル
        device: GPU
        loader: 定義したデータローダー     
    """
    model.eval()
    
    total = 0
    correct = 0
        
    for img, lbl in tqdm(loader, desc="Test"):
        img = img.to(device)
        lbl = lbl.to(device)
        with torch.no_grad():
            out = model(img)
            correct += (out.argmax(dim=1) == lbl).sum().item()
            total += out.shape[0]
    
    return correct / total


def trainer(model, device, optim, loader_train, loader_valid, criteria, n_epochs, writer):
    top1_accuracy  = 0
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        train(model, device, optim, loader_train, criteria)
        accuracy = test(model, device, loader_valid)
        top1_accuracy = compare_accuracy(top1_acc=top1_accuracy, acc=accuracy)
        
        print("Epoch %d: %5.2f %%, Top-1: %5.2f %%" % (epoch, accuracy * 100, top1_accuracy * 100))
        writer.add_scalar("Test/Accuracy", accuracy, global_step=epoch)
    
    writer.close()
    return accuracy, top1_accuracy