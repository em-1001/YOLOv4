# YOLOv4 model

import torch
import torch.nn as nn
import math

class DarknetConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=False, bn_act=True, act="mish"):
        super().__init__()

        if downsample:
            kernel_size = 3
            stride = 2
            padding = "valid"

        else:
            stride = 1
            padding = "same"

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=not bn_act
        )

        self.downsample = downsample
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
        self.act = act

    def forward(self, x):
        if self.downsample:
            x = torch.nn.functional.pad(x, (1, 0, 1, 0))
        if self.use_bn_act:
            if self.act == "mish":
                return self.mish(self.bn(self.conv(x)))
            elif self.act == "leaky":
                return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)



class CSPResBlock(nn.Module):
    def __init__(self, in_channels, num_repeats=1):
        super().__init__()

        self.split1x1 = DarknetConv2D(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1)
        self.res1x1 = DarknetConv2D(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=1)
        self.concat1x1 = DarknetConv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.num_repeats = num_repeats

        self.DenseBlock = nn.ModuleList()
        for i in range(num_repeats):
            DenseLayer = nn.ModuleList()
            DenseLayer.append(DarknetConv2D(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=1))
            DenseLayer.append(DarknetConv2D(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3))
            self.DenseBlock.append(DenseLayer)

    def forward(self, x):
        route = self.split1x1(x)
        x = self.split1x1(x)

        for module in self.DenseBlock:
            h = x
            for res in module:
                h = res(h)
            x = x + h

        x = self.res1x1(x)
        x = torch.cat([x, route], dim=1)
        x = self.concat1x1(x)

        return x



class SPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        x = torch.cat([x,
                       self.maxpool5(x),
                       self.maxpool9(x),
                       self.maxpool13(x)], dim=1)

        return x



class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv = DarknetConv2D(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, act="leaky")
        self.ScalePred = DarknetConv2D(in_channels=in_channels*2, out_channels=3*(num_classes+5), kernel_size=1, bn_act=False, act="leaky")
        self.num_classes = num_classes

    def forward(self, x):
        return(
            self.ScalePred(self.conv(x))
            # x = [batch_num, 3*(num_classes + 5), N, N
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
            # output = [B x 3 x N x N x 5+num_classes]
        )



class CSPDarknet53(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.layers = nn.ModuleList([
            DarknetConv2D(in_channels=in_channels, out_channels=32, kernel_size=3),
            DarknetConv2D(in_channels=32, out_channels=64, kernel_size=3, downsample=True),
            CSPResBlock(in_channels=64, num_repeats=1),
            DarknetConv2D(in_channels=64, out_channels=128, kernel_size=3, downsample=True),
            CSPResBlock(in_channels=128, num_repeats=2),
            DarknetConv2D(in_channels=128, out_channels=256, kernel_size=3, downsample=True),
            CSPResBlock(in_channels=256, num_repeats=8), # Route_1
            DarknetConv2D(in_channels=256, out_channels=512, kernel_size=3, downsample=True),
            CSPResBlock(in_channels=512, num_repeats=8), # Route_2
            DarknetConv2D(in_channels=512, out_channels=1024, kernel_size=3, downsample=True),
            CSPResBlock(in_channels=1024, num_repeats=4),
            DarknetConv2D(in_channels=1024, out_channels=512, kernel_size=1, act="leaky"),
            DarknetConv2D(in_channels=512, out_channels=1024, kernel_size=3, act="leaky"),
            DarknetConv2D(in_channels=1024, out_channels=512, kernel_size=1, act="leaky"),
            SPP(),
            DarknetConv2D(in_channels=2048, out_channels=512, kernel_size=1, act="leaky"),
            DarknetConv2D(in_channels=512, out_channels=1024, kernel_size=3, act="leaky"),
            DarknetConv2D(in_channels=1024, out_channels=512, kernel_size=1, act="leaky") # output
        ])

    def forward(self, x):
        route = []

        for layer in self.layers:
            x = layer(x)

            if isinstance(layer, CSPResBlock) and layer.num_repeats == 8:
                route.append(x)

        route.append(x)

        return tuple(route)



class Conv5(nn.Module):
    def __init__(self, in_channels, up=True):
        super().__init__()

        self.in_channels = in_channels
        self.up = up
        self.conv1x1 = DarknetConv2D(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, act="leaky")
        self.conv3x3 = DarknetConv2D(in_channels=in_channels//2, out_channels=in_channels, kernel_size=3, act="leaky")

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)

        return x



class YOLOv4(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.CSPDarknet53 = CSPDarknet53(in_channels)
        self.route2conv = DarknetConv2D(in_channels=512, out_channels=256, kernel_size=1, act="leaky")
        self.route1conv = DarknetConv2D(in_channels=256, out_channels=128, kernel_size=1, act="leaky")

        self.layers = nn.ModuleList([
            DarknetConv2D(in_channels=512, out_channels=256, kernel_size=1, act="leaky"),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv5(in_channels=512, up=True), # after concat
            DarknetConv2D(in_channels=256, out_channels=128, kernel_size=1, act="leaky"),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv5(in_channels=256, up=True), # after concat
            ScalePrediction(in_channels=128, num_classes=num_classes), # sbbox 52x52
            DarknetConv2D(in_channels=128, out_channels=256, kernel_size=3, downsample=True, act="leaky"),
            Conv5(in_channels=512, up=False),
            ScalePrediction(in_channels=256, num_classes=num_classes), # mbbox 26x26
            DarknetConv2D(in_channels=256, out_channels=512, kernel_size=3, downsample=True, act="leaky"),
            Conv5(in_channels=1024, up=False),
            ScalePrediction(in_channels=512, num_classes=num_classes)  # lbbox 13x13
        ])

    def forward(self, x):
        outputs = []
        Route = []
        OutputRoute = []
        route1, route2, CSPout = self.CSPDarknet53(x)

        OutputRoute.append(CSPout)

        route2 = self.route2conv(route2)
        route1 = self.route1conv(route1)
        Route.append(route1)
        Route.append(route2)


        x = CSPout
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue # Since this is the output of each scale, it must skip x = ScalePrediction(x).

            x = layer(x)

            if isinstance(layer, nn.Upsample):
                x = torch.cat([Route[-1], x], dim=1)
                Route.pop()

            if isinstance(layer, Conv5) and layer.in_channels == 512 and layer.up == True:
                OutputRoute.append(x)

            if isinstance(layer, DarknetConv2D) and layer.downsample == True:
                x = torch.cat([x, OutputRoute[-1]], dim=1)
                OutputRoute.pop()

        outputs[0], outputs[1], outputs[2] = outputs[2], outputs[1], outputs[0]
        return outputs
