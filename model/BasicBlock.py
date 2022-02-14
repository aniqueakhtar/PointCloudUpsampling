import torch.nn as nn
import MinkowskiEngine as ME


class ResNet(nn.Module):
    """
    Basic block: Residual
    """
    
    def __init__(self, channels):
        super(ResNet, self).__init__()
        #path_1
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            has_bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            has_bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


class MyInception_1(nn.Module):
    def __init__(self,
                 channels,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=3):
        super(MyInception_1, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=1, stride=stride, dilation=dilation, has_bias=True, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            channels//4, channels//4, kernel_size=3, stride=stride, dilation=dilation, has_bias=True, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(
            channels//4, channels//2, kernel_size=1, stride=stride, dilation=dilation, has_bias=True, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(channels//2, momentum=bn_momentum)
        
        self.conv4 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=3, stride=stride, dilation=dilation, has_bias=True, dimension=dimension)
        self.norm4 = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.conv5 = ME.MinkowskiConvolution(
            channels//4, channels//2, kernel_size=3, stride=stride, dilation=dilation, has_bias=True, dimension=dimension)
        self.norm5 = ME.MinkowskiBatchNorm(channels//2, momentum=bn_momentum)
        
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)
        
        # 2
        out1 = self.conv4(x)
        out1 = self.norm4(out1)
        out1 = self.relu(out1)
        
        out1 = self.conv5(out1)
        out1 = self.norm5(out1)
        out1 = self.relu(out1)

        # 3
        out2 = ME.cat(out,out1)
        out2 += x

        return out2


class Pyramid_1(nn.Module):
    def __init__(self,
                 channels,
                 bn_momentum=0.1,
                 dimension=3):
        super(Pyramid_1, self).__init__()
        assert dimension > 0
        
        self.aspp1 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=1, stride=1, dilation=1, has_bias=True, dimension=dimension)
        self.aspp2 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=3, stride=1, dilation=6, has_bias=True, dimension=dimension)
        self.aspp3 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=3, stride=1, dilation=12, has_bias=True, dimension=dimension)
        self.aspp4 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=3, stride=1, dilation=18, has_bias=True, dimension=dimension)
        self.aspp5 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=1, stride=1, dilation=1, has_bias=True, dimension=dimension)
        
        self.aspp1_bn = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.aspp2_bn = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.aspp3_bn = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.aspp4_bn = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.aspp5_bn = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        
        self.conv2 = ME.MinkowskiConvolution(
            channels//4 * 5, channels, kernel_size=1, stride=1, dilation=1, has_bias=True, dimension=dimension)
        self.bn2 = ME.MinkowskiBatchNorm(channels, momentum=bn_momentum)
        
        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast = ME.MinkowskiBroadcast()
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        
        x5 = self.pooling(x)
        x5 = self.broadcast(x, x5)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        
        x6 = ME.cat(x1, x2, x3, x4, x5)
        x6 = self.conv2(x6)
        x6 = self.bn2(x6)
        x6 = self.relu(x6)
        
        x7 = x6 + x
        
        return x7

