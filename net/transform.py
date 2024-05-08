import torch
import torch.nn as nn

class CIN(nn.Module):
    def __init__(self, num_features, num_styles):
        super(CIN, self).__init__()
        self.num_features = num_features
        self.num_styles = num_styles
        
        # 初始化風格特定的 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(num_styles, num_features))
        self.beta = nn.Parameter(torch.zeros(num_styles, num_features))
        
    def forward(self, x, style_id):
        """
        x: 輸入特徵，維度為 (N, C, H, W)
        style_id: 表示風格的整數索引，維度為 (N,)
        """
        # 獲取每個樣本的 gamma 和 beta
        gamma = self.gamma[style_id].view(-1, self.num_features, 1, 1)
        beta = self.beta[style_id].view(-1, self.num_features, 1, 1)
        
        # 計算特徵的均值和標準差
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        
        # 應用條件實例正則化
        normalized = (x - mean) / (std + 1e-5)
        out = normalized * gamma + beta
        
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,num_styles):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.cin = CIN(out_channels,num_styles)

    def forward(self, x, style_id):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.cin(out, style_id)
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_styles, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.cin = CIN(out_channels,num_styles)

    def forward(self, x, style_id):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.cin(out,style_id)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels,num_styles):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1,num_styles=num_styles)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1,num_styles=num_styles)

    def forward(self, x, style_id):
        residual = x
        out = self.relu(self.conv1(x, style_id))
        out = self.conv2(out, style_id)
        out = out + residual
        return out 

# Image Transform Network
class TransformNet(nn.Module):
    def __init__(self,num_styles = 8):
        super(TransformNet, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(in_channels=3 , out_channels=32 , kernel_size=9, stride=1,num_styles=num_styles)
        self.conv2 = ConvLayer(in_channels=32, out_channels=64 , kernel_size=3, stride=2,num_styles=num_styles)
        self.conv3 = ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2,num_styles=num_styles)

        self.res1 = ResidualBlock(channels=128,num_styles=num_styles)
        self.res2 = ResidualBlock(channels=128,num_styles=num_styles)
        self.res3 = ResidualBlock(channels=128,num_styles=num_styles)
        self.res4 = ResidualBlock(channels=128,num_styles=num_styles)
        self.res5 = ResidualBlock(channels=128,num_styles=num_styles)

        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1,num_styles=num_styles, upsample=2 )
        self.deconv2 = UpsampleConvLayer(64 , 32, kernel_size=3, stride=1,num_styles=num_styles, upsample=2 )
        self.deconv1 = UpsampleConvLayer(32 , 3 , kernel_size=9, stride=1,num_styles=num_styles)

    def forward(self, x ,style_id):

        y = self.relu(self.conv1(x,style_id))
        y = self.relu(self.conv2(y,style_id))
        y = self.relu(self.conv3(y,style_id))

        y = self.res1(y,style_id)
        y = self.res2(y,style_id)
        y = self.res3(y,style_id)
        y = self.res4(y,style_id)
        y = self.res5(y,style_id)

        y = self.relu(self.deconv3(y,style_id))
        y = self.relu(self.deconv2(y,style_id))
        y = self.deconv1(y,style_id)

        return y