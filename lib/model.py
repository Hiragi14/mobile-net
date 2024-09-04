import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class DSCBlock(nn.Module):
    """Information about DSCBlock
    This class defines the block for depthwise separable convolution.
    Depthwise saparable convolution is made with two conponents; depthwise convolution and pointwise convolution.
    
    Parameters:
        in_channels (int): number of input channel
        out_channels (int): number of output channel
        kernel_size (int): kernel size(for depthwise convolution)
        stride (int): stride(for depthwise convolution)
        padding (int): padding(for depthwise convolution)
        bias : bias
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=None):
        super(DSCBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                        stride=1, padding=0, bias=None)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x



class MobileNet(nn.Module):
    """Information about MobileNet
    This class defines Mobile Net.
    
    Parameters:
        image_width: width of input image
        image_height: height of input image
        image_channels: number of channel of input image
        num_class: the number of classes in your task
    """
    def __init__(self, image_width, image_height, image_channels, num_class):
        super(MobileNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3,
                                        stride=2, padding=1, bias=None)
        
        self.dsconv_layers = nn.Modulelist([
            DSCBlock(32, 64, stride=1),
            DSCBlock(64, 128, stride=2),
            DSCBlock(128, 128, stride=1),
            DSCBlock(128, 256, stride=2),
            DSCBlock(256, 256, stride=1),
            DSCBlock(256, 512, stride=2),
            DSCBlock(512, 512, stride=1),
            DSCBlock(512, 512, stride=1),
            DSCBlock(512, 512, stride=1),
            DSCBlock(512, 512, stride=1),
            DSCBlock(512, 512, stride=1),
            DSCBlock(512, 1024, stride=2),
            DSCBlock(1024, 1024, stride=1),
        ])
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_class, bias=None)
        
    def forward(self, x):
        self.input = x
        
        for layer in self.dsconv_layers:
            x = layer(x)
        
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x