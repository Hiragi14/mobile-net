import click

@click.command()
@click.option('--dataset', default='CIFAR10', help='Dataset to use for training')
@click.option('--batch-size', default=64, help='Batch size for training')
@click.option('--epochs', default=100, help='Number of epochs to train')
@click.option('--learning-rate', default=0.01, help='Learning rate for training')
def train(dataset, batch_size, epochs, learning_rate):
    print(f"Dataset: {dataset}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")

if __name__ == '__main__':
    train()
