import torch
import torch.nn as nn
from math import sqrt


'''
Pytorch 使用nn.Module定义模型的方法
1.必须继承nn.Module这个类，要让PyTorch知道这个类是一个Module
2.在init(self)中设置好需要的"组件"(如conv,pooling,Linear,BatchNorm等)
3.最后，在forward(self,x)中定义好的“组件”进行组装，就像搭积木，把网络结构搭建出来，这样一个模型就定义好了
'''

# 中间层
class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()  # 找到Net的父类，并将其初始化。这里继承的是nn.Module类

        # 创建中间层 中间层有18个block,每个block为 Conv + Relu
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)

        # 输入的第一个卷积层。 不使用bias,并且先扩大一圈保证输出还是原图大小。  输入为灰度图
        self.input_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        # 输出卷积层
        self.output_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)  # inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出

        for m in self.modules():  # self.modules(), 他会返回该网络中的所有modules.
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # k^2 * c
                m.weight.data.normal_(0, sqrt(2. / n))  # He的方法，权值初始化

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)  # 模块将按照构造函数中传递的顺序添加到模块中

    def forward(self, x):
        ILR = x  # 这里的residual应该是原图
        hidden = self.relu(self.input_conv(x))  # 第一层
        hidden = self.residual_layer(hidden)  # 中间层
        out = self.output_conv(hidden)  # 输出层，无激活函数
        out = torch.add(out, ILR)  # 预测的残差结果和原图相加
        return out