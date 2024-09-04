import click
import torch
import csv

from lib.trainer import trainer
from lib.utils import set_optim, set_device, set_lossfunction
from lib.model import MobileNet
from lib.datadeal import dataload

from datetime import datetime

# 現在の日時を取得してフォルダ名を生成
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

def tensorboad(dataset, optim):
    from torch.utils.tensorboard import SummaryWriter
    # 現在の日時を取得してフォルダ名を生成
    log_dir = f'runs/{dataset+"-"+optim+"-"+current_time}'

    # TensorBoard writerの初期化
    writer = SummaryWriter(log_dir)
    return writer


@click.command()
@click.option('--dataset', default='CIFAR10', help='Dataset to use for training')
@click.option('--batch-size', default=64, help='Batch size for training')
@click.option('--epochs', default=100, help='Number of epochs to train')
@click.option('--learning-rate', default=0.01, help='Learning rate for training')
@click.option('--optimizer', default='RMSprop', help='optimizer for training')
@click.option('--gpu', default=0, help='GPU device number')
@click.option('--criteria_name', default='CrossEntropyLoss', help='criteria for training')
def main(dataset, batch_size, epochs, learning_rate, optimizer, gpu, criteria_name):
    writer = tensorboad(dataset, optimizer)
    device = set_device(gpu)
    criteria = set_lossfunction(criteria_name)
    model = MobileNet(32, 32, 3, 100).to(device)
    dataloader_train, dataloader_valid = dataload(dataset, batch_size)
    optim = set_optim(optim_name=optimizer, model=model, learning_rate=learning_rate, momentum=0.01)
    
    accuracy, top1accuracy = trainer(model=model, device=device, optim=optim,loader_train=dataloader_train,
                                    loader_valid=dataloader_valid, criteria=criteria, n_epochs=epochs, writer=writer)
    
    data = [current_time, epochs, batch_size, 'Mobile-Net', optimizer, learning_rate, criteria_name, dataset, accuracy, top1accuracy]
    with open('./result.csv', 'a', newline='') as f:
        writer_csv = csv.writer(f, delimiter='\t')
        writer_csv.writerow(data)



if __name__ == '__main__':
    main()