import torch
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])

transform_ImageNet = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])


def dataload(dataset, batch_size):
    if dataset == "CIFAR10":
        dataloader_train = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='/DeepLearning/Dataset/torchvision/CIFAR10', train=True, download=False, transform=transform),
            batch_size=batch_size,
            shuffle=True
        )

        dataloader_valid = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='/DeepLearning/Dataset/torchvision/CIFAR10', train=False, download=False, transform=transform),
            batch_size=batch_size,
            shuffle=False
        )
    elif dataset == "CIFAR100":
        dataloader_train = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='/DeepLearning/Dataset/torchvision/CIFAR100', train=True, download=False, transform=transform),
            batch_size=batch_size,
            shuffle=True
        )

        dataloader_valid = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='/DeepLearning/Dataset/torchvision/CIFAR100', train=False, download=False, transform=transform),
            batch_size=batch_size,
            shuffle=False
        )
    elif dataset == "ImageNet":
        dataloader_train = torch.utils.data.DataLoader(
            datasets.ImageNet(root='/DeepLearning/Dataset/torchvision/ImageNet', train=True, download=False, transform=transform_ImageNet),
            batch_size=batch_size,
            shuffle=True
        )

        dataloader_valid = torch.utils.data.DataLoader(
            datasets.ImageNet(root='/DeepLearning/Dataset/torchvision/ImageNet', train=False, download=False, transform=transform_ImageNet),
            batch_size=batch_size,
            shuffle=False
        )
    else :
        raise RuntimeError("dataset is not selected")
    
    return dataloader_train, dataloader_valid