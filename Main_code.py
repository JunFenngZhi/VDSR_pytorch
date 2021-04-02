import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchsummary
from VDSR_model import VDSR
from RDN_model import RDN

'''RDN和VDSR区别：使用L1_loss,不用梯度裁剪，学习率基本不衰减，使用adam优化器,batch_size增大'''




'''训练参数'''
LR = 0.1  # 模型学习率
WEIGHT_DECAY = 1e-4  # 权值衰减参数
MOMENTUM = 0.9  # SGD动量项
GRADIENT_CLIP = 0.4  # 梯度裁剪值
DECAY_INTERVAL = 10   # 学习率衰减间隔
BATCH_SIZE = 64
EPOCHS = 50  # 总训练轮数
INPUT_SIZE = [(1, 41, 41)]
MODEL_PATH = ''


CUDA = True  # 使用GPU进行训练
GPU_ID = 0  # 使用GPU的ID

class ImageDataSet(data.Dataset):
    def __init__(self, x_path, y_path):
        super(ImageDataSet, self).__init__()
        self.data = np.load(file=x_path)  # 加载数据
        self.data = np.reshape(self.data, (self.data.shape[0], 1, self.data.shape[1], self.data.shape[2]))  # 调整格式，与conv层输入格式一致
        self.target = np.load(file=y_path)
        self.target = np.reshape(self.target, (self.target.shape[0], 1, self.target.shape[1], self.target.shape[2]))
        print("数据集大小为：", self.data.shape)

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index, :, :,:]).float()
        y = torch.from_numpy(self.target[index, :, :,:]).float()
        return  x/255., y/255.

    def __len__(self):
        return self.data.shape[0]

def EnvConfiguration():
    if CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    torch.manual_seed(278)  # 产生固定的随机数
    if CUDA:
        torch.cuda.manual_seed(278)  # 产生固定的随机数
    cudnn.benchmark = True  # 使用benchmark模式进行卷积层网络加速，适用于结构固定的网络

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]  # 去除图像边缘，只对中间区域进行计算对比
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt  # 恢复图和真实图作差
    rmse = math.sqrt(np.mean(imdff ** 2))  # residual的均方差再开平方。因为分子的平方提了出去，所以分母要开方
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def Denormalize(image):
    image = image * 255.  # 图片反归一化回0-255
    image[image < 0] = 0.
    image[image > 255.] = 255.
    return image

def adjust_learning_rate(optimizer, epoch):
    '''Sets the learning rate to the initial LR decayed by 10 every 10 epochs'''
    lr = LR * (0.1 ** (epoch //DECAY_INTERVAL))  # lr = lr * 0.1^(epoch//interval)   LR是lr初始值
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr   # 更新优化器中的lr设定
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    #current_lr = adjust_learning_rate(optimizer, epoch-1)  # 调整学习率
    #print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()  # 将module设置为training mode，允许权值更新

    for iteration, batch in enumerate(training_data_loader, 1):  # 这里的1是为了让iteration的值从1开始
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False) # 将tensor封装成variable用于训练

        if CUDA:
            input = input.cuda()  # 输入网络中进行前向传播
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)  # 求解loss(这个batch的loss)
        optimizer.zero_grad()  # 每个batch计算梯度前先将梯度初始化为0
        loss.backward()  # 反向传播，计算梯度
        #nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)  # 梯度裁剪  （并没有对梯度裁剪的的值进行修改，和论文不一样）
        optimizer.step()  # 根据梯度更新权值

        if iteration % 100 == 0:  # 每100个batch,输出当前batch的平均损失
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data.item()/BATCH_SIZE))

def val(val_data_loader, model, epoch):
    global n_val_batch
    val_loss = 0
    val_psnr = 0
    with torch.no_grad():
        for iteration, batch in enumerate(val_data_loader, 1):
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)  # 将tensor封装成variable用于训练
            if CUDA:
                input = input.cuda()  # 输入网络中进行前向传播
                target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            val_loss += loss.item()  # 统计所有batch的loss总和

            output = output.cpu()
            output = output.data.numpy().astype(np.float32)
            target = target.cpu()
            target = target.data.numpy().astype(np.float32)
            for outImg, gtImg in zip(output, target):
                outImg = Denormalize(outImg)
                gtImg = Denormalize(gtImg)
                psnr_predicted = PSNR(gtImg[0, :, :], outImg[0, :, :], shave_border=2)  # 计算恢复的PSNR
                val_psnr += psnr_predicted

    print("Epoch：", epoch, "的val_loss为：", val_loss/ (n_val_batch*BATCH_SIZE))  # 输出验证集数据的平均loss
    print("Epoch：", epoch, "的val_psnr为：", val_psnr / (n_val_batch * BATCH_SIZE), "\n")  # 输出验证集数据的平均psnr



if __name__ == '__main__':
    print("__________开始训练VDSR__________")
    EnvConfiguration()  # 配置训练环境

    print("===> Loading datasets")  # 导入数据，自动划分batch
    train_data = ImageDataSet(x_path="/content/drive/MyDrive/Super_Resolution/ILR_train_data.npy",
                   y_path="/content/drive/MyDrive/Super_Resolution/HR_train_data.npy")
    training_data_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    n_train_batch = len(training_data_loader)  # 训练数据的batch总数

    val_data = ImageDataSet(x_path="/content/drive/MyDrive/Super_Resolution/ILR_val_data.npy",
                  y_path="/content/drive/MyDrive/Super_Resolution/HR_val_data.npy")
    val_data_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    n_val_batch = len(val_data_loader)  # 验证数据的batch总数


    print("===> Building model")
    #model = VDSR()
    model = RDN(D=10, C=3, G=32, G0=64, input_channels=1, output_channels=1)


    #criterion = nn.MSELoss(reduction='sum')  # 计算的是整个batch所有图像的mse_loss的总和  (L2_loss)
    criterion = nn.L1Loss(reduction='sum')  # RDN使用的是L1_loss
    if CUDA:
        model = model.cuda()
        criterion = criterion.cuda()

    torchsummary.summary(model, input_size=INPUT_SIZE)  # 显示模型结构和参数
    print("===> Setting Optimizer")
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("===> Training")  # 模型训练固定个epoch
    for epoch in range(1, EPOCHS+1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        val(val_data_loader, model, epoch)
        #torch.save(model, MODEL_PATH+'_'+str(epoch)+'.pth')

