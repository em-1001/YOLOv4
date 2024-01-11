# YOLOv4 model

import torch
import torch.nn as nn
import math


class DarknetConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act="mish", bn_act=True):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=not bn_act
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()
        self.leaky = nn.LeakyReLU(0.1, inplace=True)
        self.use_bn_act = bn_act
        self.act = act

    def forward(self, x):
        if self.use_bn_act:
            if self.act == "mish":
                return self.mish(self.bn(self.conv(x)))
            elif self.act == "leaky":
                return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)



class CSPResBlock(nn.Module):
    def __init__(self, in_channels, is_first=False, num_repeats=1):
        super().__init__()

        self.route_1 = DarknetConv2D(in_channels, in_channels//2, 1, 1, 'mish')
        self.route_2 = DarknetConv2D(in_channels, in_channels//2, 1, 1, 'mish')
        self.res1x1 = DarknetConv2D(in_channels//2, in_channels//2, 1, 1, 'mish')
        self.concat1x1 = DarknetConv2D(in_channels, in_channels, 1, 1, 'mish')
        self.num_repeats = num_repeats

        self.DenseBlock = nn.ModuleList()
        for i in range(num_repeats):
            DenseLayer = nn.ModuleList()
            DenseLayer.append(DarknetConv2D(in_channels//2, in_channels//2, 1, 1, 'mish'))
            DenseLayer.append(DarknetConv2D(in_channels//2, in_channels//2, 3, 1, 'mish'))
            self.DenseBlock.append(DenseLayer)

        if is_first:
            self.route_1 = DarknetConv2D(in_channels, in_channels, 1, 1, 'mish')
            self.route_2 = DarknetConv2D(in_channels, in_channels, 1, 1, 'mish')
            self.res1x1 = DarknetConv2D(in_channels, in_channels, 1, 1, 'mish')
            self.concat1x1 = DarknetConv2D(in_channels*2, in_channels, 1, 1, 'mish')

            self.DenseBlock = nn.ModuleList()
            for i in range(num_repeats):
                DenseLayer = nn.ModuleList()
                DenseLayer.append(DarknetConv2D(in_channels, in_channels//2, 1, 1, 'mish'))
                DenseLayer.append(DarknetConv2D(in_channels//2, in_channels, 3, 1, 'mish'))
                self.DenseBlock.append(DenseLayer)

    def forward(self, x):
        route = self.route_1(x)
        x = self.route_2(x)

        for module in self.DenseBlock:
            h = x
            for res in module:
                h = res(h)
            x = h + x

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
        x = torch.cat([self.maxpool13(x),
                       self.maxpool9(x),
                       self.maxpool5(x),
                       x], dim=1)

        return x



class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv = DarknetConv2D(in_channels, in_channels*2, 3, 1, "leaky")
        self.ScalePred = DarknetConv2D(in_channels*2, 3*(num_classes+5), 1, 1, "leaky", bn_act=False)
        self.num_classes = num_classes

    def forward(self, x):
        return(
            self.ScalePred(self.conv(x))
            # x = [batch_num, 3*(num_classes + 5), N, N
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
            # output = [B x 3 x N x N x 5+num_classes]
        )



class Conv5(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.conv1x1 = DarknetConv2D(in_channels, in_channels//2, 1, 1, "leaky")
        self.conv3x3 = DarknetConv2D(in_channels//2, in_channels, 3, 1, "leaky")

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)

        return x



class Conv3(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            DarknetConv2D(2048, 512, 1, 1, "leaky"),
            DarknetConv2D(512, 1024, 3, 1, "leaky"),
            DarknetConv2D(1024, 512, 1, 1, "leaky")
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x



class CSPDarknet53(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.layers = nn.ModuleList([
            DarknetConv2D(in_channels, 32, 3, 1, 'mish'),
            DarknetConv2D(32, 64, 3, 2, 'mish'),
            CSPResBlock(in_channels=64, is_first=True, num_repeats=1),
            DarknetConv2D(64, 128, 3, 2, 'mish'),
            CSPResBlock(in_channels=128, num_repeats=2),
            DarknetConv2D(128, 256, 3, 2, 'mish'),
            CSPResBlock(in_channels=256, num_repeats=8), # P3
            DarknetConv2D(256, 512, 3, 2, 'mish'),
            CSPResBlock(in_channels=512, num_repeats=8), # P4
            DarknetConv2D(512, 1024, 3, 2, 'mish'),
            CSPResBlock(in_channels=1024, num_repeats=4) # P5
        ])

    def forward(self, x):
        route = []

        for layer in self.layers:
            x = layer(x)

            if (isinstance(layer, CSPResBlock) and layer.num_repeats == 8) or (isinstance(layer, CSPResBlock) and layer.num_repeats == 4):
                route.append(x)

        P5, P4, P3 = route[2], route[1], route[0]

        return P5, P4, P3



class PANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.UpConv_P4 = DarknetConv2D(512, 256, 1, 1, "leaky")
        self.UpConv_P3 = DarknetConv2D(256, 128, 1, 1, "leaky")

        self.layers = nn.ModuleList([
            DarknetConv2D(1024, 512, 1, 1, "leaky"),
            DarknetConv2D(512, 1024, 3, 1, "leaky"),
            DarknetConv2D(1024, 512, 1, 1, "leaky"),
            SPP(),
            Conv3(), # N5
            DarknetConv2D(512, 256, 1, 1, "leaky"),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv5(in_channels=512), # N4
            DarknetConv2D(256, 128, 1, 1, "leaky"),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv5(in_channels=256) # N3
        ])

    def forward(self, P5, P4, P3):
        P4 = self.UpConv_P4(P4)
        P3 = self.UpConv_P3(P3)

        P = [P3, P4]
        N = []

        x = P5
        for layer in self.layers:
            x = layer(x)

            if isinstance(layer, nn.Upsample):
                x = torch.cat([P[-1], x], dim=1)
                P.pop()

            if isinstance(layer, Conv3):
                N.append(x)

            if isinstance(layer, Conv5):
                N.append(x)

        N5, N4, N3 = N[0], N[1], N[2]

        return N3, N4, N5



class YOLOv4(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()

        self.CSPDarknet53 = CSPDarknet53(in_channels)
        self.PANet = PANet()

        self.layers = nn.ModuleList([
            ScalePrediction(in_channels=128, num_classes=num_classes), # sbbox 52x52
            DarknetConv2D(128, 256, 3, 2, "leaky"),
            Conv5(in_channels=512),
            ScalePrediction(in_channels=256, num_classes=num_classes), # mbbox 26x26
            DarknetConv2D(256, 512, 3, 2, "leaky"),
            Conv5(in_channels=1024),
            ScalePrediction(in_channels=512, num_classes=num_classes)  # lbbox 13x13
        ])

    def forward(self, x):
        P5, P4, P3 = self.CSPDarknet53(x)
        N3, N4, N5 = self.PANet(P5, P4, P3)
        N = [N5, N4]

        outputs = []

        x = N3
        for layer in self.layers:

            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue # Since this is the output of each scale, it must skip x = ScalePrediction(x).

            x = layer(x)

            if isinstance(layer, DarknetConv2D):
                x = torch.cat([x, N[-1]], dim=1)
                N.pop()

        outputs[0], outputs[1], outputs[2] = outputs[2], outputs[1], outputs[0]

        '''
        torch.Size([1, 13, 13, 255])
        torch.Size([1, 26, 26, 255])
        torch.Size([1, 52, 52, 255])
        '''

        return outputs
