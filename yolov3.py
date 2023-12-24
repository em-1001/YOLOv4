# YOLOv3 model

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            bias=not bn_act, **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)
        

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, use_residual=True, repeats=1):
        super().__init__()
        
        res_layers = []
        
        for _ in range(repeats):
            res_layers += [
                nn.Sequential(
                    CNNBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bn_act=True), # 1x1 
                    CNNBlock(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1, bn_act=True) # 3x3
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.repeats = repeats
        self.use_residual = use_residual
        
    def forward(self, x):
        for layer in self.layers:
            res = x
            x = layer(x)
            if self.use_residual:
                x = x + res  # skip connection
        return x
        

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1, bn_act=True),
            # (4 + 1 + num_classes) * 3 : 4 for [x, y, w, h], 1 for objectness prediction, 3 for anchor boxes per grid ce
            CNNBlock(in_channels*2, 3*(4+1+num_classes), kernel_size=1, stride=1, padding=0, bn_act=False)
        )
        self.num_classes =  num_classes
    
    def forward(self, x):
        output = self.pred(x)
        
        # x = [batch_num, 3*(num_classes + 5), N, N]
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0, 1, 3, 4, 2) # output = [B x 3 x N x N x 5+num_classes]
        return output 
        

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels  
        
        self.layers = nn.ModuleList([
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1, bn_act=True), # padding=1 if kernel_size == 3 else 0
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1, bn_act=True),
            ResidualBlock(64, repeats=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1, bn_act=True),
            ResidualBlock(128, repeats=2),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1, bn_act=True),
            ResidualBlock(256, repeats=8),
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1, bn_act=True),
            ResidualBlock(512, repeats=8),
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1, bn_act=True),
            ResidualBlock(1024, repeats=4), # To this point is Darknet-53
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0, bn_act=True),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1, bn_act=True),
            ResidualBlock(1024, use_residual=False, repeats=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0, bn_act=True),
            ScalePrediction(512, num_classes=num_classes),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, bn_act=True), # for concatenate
            nn.Upsample(scale_factor=2),
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0, bn_act=True),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, bn_act=True),
            ResidualBlock(512, use_residual=False, repeats=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, bn_act=True),
            ScalePrediction(256, num_classes=num_classes),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0, bn_act=True), # for concatenate
            nn.Upsample(scale_factor=2),
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0, bn_act=True),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1, bn_act=True),
            ResidualBlock(256, use_residual=False, repeats=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0, bn_act=True),
            ScalePrediction(128, num_classes=num_classes)
        ])
        
    def forward(self, x):
        outputs = [] # for each scale
        route_connections = [] # for concatenate
        
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue # Since this is the output of each scale, it must skip x = ScalePrediction(x).
            x = layer(x)
            
            if isinstance(layer, ResidualBlock) and layer.repeats == 8:
                route_connections.append(x)
            
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
            
        return outputs
