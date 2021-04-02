import math
import numpy as np
import cv2
import os
import torch
from VDSR_model import VDSR, Conv_ReLU_Block
from RDN_model import RDB_conv, RDN, RDB
from DRRN_B1U25 import drrn

'''目标：1、输入路径和模型，可以自动对set5图片集进行分割和SR处理，并计算整体平均psnr(不同的scale)和bicubic对比
         2、可以显示指定图片的处理结果（原图、Bicubic、VDSR对比）
         3、整张图片的psnr应该是整张图片来计算，不是分块求平均
'''


def PSNR(pred, gt, shave_border=0):
    pred = pred.astype(float)
    gt = gt.astype(float)  # 转换为float型，防止unit8相减出现负数时溢出
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]  # 去除图像边缘，只对中间区域进行计算对比
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt  # 恢复图和真实图作差
    rmse = math.sqrt(np.mean(imdff ** 2))  # residual的均方差再开平方。因为分子的平方提了出去，所以分母要开方
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

class EvalModel():
    def __init__(self, model_path, patch_size, downsize=None, cuda=True):
        if(cuda == False):
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            self.model = torch.load(model_path)
        self.downsize = downsize
        self.patch_size = patch_size
        self.cuda = cuda

    # 计算整个验证集的基准psnr和模型恢复的psnr
    def dataset_test(self, path, scale):
        img_count = 0
        bicubic_psnr = 0
        model_psnr = 0
        for img_name in os.listdir(path):
            img_count += 1
            #print("当前处理图片：",img_count)
            file_path = path + '\\' + img_name
            HR_img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # 读入彩色图片
            if (self.downsize != None):
                HR_img = cv2.resize(HR_img, (HR_img.shape[1] // self.downsize, HR_img.shape[0] // self.downsize))
            HR_img = self.crop_img(img=HR_img)  # 裁剪
            HR_shape = (HR_img.shape[1], HR_img.shape[0])  # dim = (width, height)
            LR_shape = (HR_img.shape[1] // scale, HR_img.shape[0] // scale)  # 确定按比例放缩后的size
            LR_img = cv2.resize(HR_img, LR_shape, interpolation=cv2.INTER_CUBIC)  # 使用双线性三次插值方法
            ILR_img = cv2.resize(LR_img, HR_shape, interpolation=cv2.INTER_CUBIC)  # 插值恢复

            ILR_y_img = cv2.cvtColor(ILR_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]  # 转为Ycbcr格式，只保留Y通道，其余通道信息丢弃
            HR_y_img = cv2.cvtColor(HR_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            bicubic_psnr += PSNR(pred=ILR_y_img.copy(), gt=HR_y_img.copy(), shave_border=0)  # 计算Bicubic恢复出来的psnr(基准值)

            ILR_patches, HR_patches, width_num, height_num = self.spilt_image(HR_img=HR_y_img, ILR_img=ILR_y_img)
            pred_patches = self.model_predict(x_list=ILR_patches)
            pred_y_img = self.rebuild_img(pred_patches, width_num, height_num)  # 将图像块，按顺序重建回一个完整的图像
            model_psnr += PSNR(pred_y_img.copy(), HR_y_img.copy(), shave_border=0)  # 计算模型恢复图片的psnr
        print("Scale = ", scale)
        print("PSNR_bicubic = ", bicubic_psnr/img_count)
        print("PSNR_predicted = ", model_psnr/img_count)

    # 对一张图片进行psnr对比计算，并将复原结果显示出来
    def image_test(self, img_path, scale):
        HR_img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 读入彩色图片
        if (self.downsize != None):
            HR_img = cv2.resize(HR_img, (HR_img.shape[1] // self.downsize, HR_img.shape[0] // self.downsize))
        HR_img = self.crop_img(img=HR_img)  # 裁剪
        HR_shape = (HR_img.shape[1], HR_img.shape[0])  # dim = (width, height)
        LR_shape = (HR_img.shape[1] // scale, HR_img.shape[0] // scale)  # 确定按比例放缩后的size
        LR_img = cv2.resize(HR_img, LR_shape, interpolation=cv2.INTER_CUBIC)  # 使用双线性三次插值方法
        ILR_img = cv2.resize(LR_img, HR_shape, interpolation=cv2.INTER_CUBIC)  # 插值恢复

        ILR_ycrcb_img = cv2.cvtColor(ILR_img, cv2.COLOR_BGR2YCrCb)  # 转换格式
        HR_ycrcb_img = cv2.cvtColor(HR_img, cv2.COLOR_BGR2YCrCb)

        bicubic_psnr = PSNR(pred=ILR_ycrcb_img[:, :, 0].copy(), gt=HR_ycrcb_img[:, :, 0].copy(), shave_border=scale)
        print("PSNR_bicubic = ", bicubic_psnr)

        ILR_patches, HR_patches, width_num, height_num = self.spilt_image(HR_img=HR_ycrcb_img[:, :, 0], ILR_img=ILR_ycrcb_img[:, :, 0])
        pred_patches = self.model_predict(x_list=ILR_patches)  # 对Y通道图片进行恢复
        pred_y_img = self.rebuild_img(pred_patches, width_num, height_num)  # 恢复成完整图片
        model_psnr = PSNR(pred=pred_y_img.copy(), gt=HR_ycrcb_img[:, :, 0].copy(), shave_border=scale)  # 计算模型恢复图片的psnr
        print("PSNR_predicted = ", model_psnr)

        pred_img = self.colorize(y_img=pred_y_img, ycrcb_img=ILR_ycrcb_img)  # 补全其余通道信息，恢复成BGR格式
        print("scale:", scale)
        cv2.imshow("HR_img", HR_img)
        cv2.imshow("ILR_img", ILR_img)
        cv2.imshow("pred_img", pred_img)
        cv2.waitKey(0)

    # 对一张图片进行分块，返回(numpy)列表
    def spilt_image(self, HR_img, ILR_img):
        ILR_list = []
        HR_list = []
        width_num = HR_img.shape[1]//self.patch_size  # 确定横向上可以取多少个块  ( HR\ILR等大)
        height_num = HR_img.shape[0]//self.patch_size  # 确定纵向上可以取多少个块
        #print("width_num:", width_num, "height_num:", height_num)
        for i in range(0, height_num):
            for j in range(0, width_num):
                HR_ImgPatch = HR_img[(0 + i * self.patch_size):(self.patch_size + i * self.patch_size),
                              (0 + j * self.patch_size):(self.patch_size + j * self.patch_size)]
                ILR_ImgPatch = ILR_img[(0 + i * self.patch_size):(self.patch_size + i * self.patch_size),
                               (0 + j * self.patch_size):(self.patch_size + j * self.patch_size)]
                ILR_list.append(ILR_ImgPatch)
                HR_list.append(HR_ImgPatch)
        return np.array(ILR_list), np.array(HR_list), width_num, height_num

    # 获得模型的预测结果
    def model_predict(self, x_list):
        x_list = x_list/255.  # 归一化
        x_list = np.reshape(x_list, (x_list.shape[0], 1, x_list.shape[1], x_list.shape[2]))  # 调整格式
        x_tensor = torch.from_numpy(x_list).float()  # 将numpy转为tensor
        im_input = torch.autograd.Variable(x_tensor)  # view相当于reshape
        if self.cuda:
            model = self.model.cuda()
            im_input = im_input.cuda()
        else:
            model = self.model.cpu()

        out = model(im_input)
        out = out.cpu()
        img_pred_y = out.data.numpy().astype(np.float32)
        img_pred_y = img_pred_y[:, 0, :, :]  # 恢复格式
        img_pred_y = self.denormalize(image=img_pred_y)  # 反归一化
        return img_pred_y

    # 反归一化。 输入可以是一个图，也可以是很多图
    def denormalize(self, image):
        image = image * 255.  # 图片反归一化回0-255
        image[image < 0] = 0
        image[image > 255.] = 255.
        return image

    # 将图像块，按顺序重建回一个完整的图像
    def rebuild_img(self, patch_list, width_num, height_num):
        row_list = []  # 存储每行 行方向上拼接起来的图像
        for i in range(0, height_num):
            row_patch = []  # 存放每行的块
            for j in range(0, width_num):
                row_patch.append(patch_list[i*width_num+j])
            temp = np.concatenate(row_patch, axis=1)  # 将每一行的块拼起来
            row_list.append(temp)
        completed_img = np.concatenate(row_list, axis=0)  # 将各行拼起来，形成完整图像
        return completed_img

    # 根据patch_size,去除图像边缘。使刚好能够被分块
    def crop_img(self, img):
        (crop_w, crop_h) = (img.shape[1]%self.patch_size, img.shape[0]%self.patch_size)  # 计算宽和高需要裁剪的像素
        img = img[0:img.shape[0]-crop_h, 0:img.shape[1]-crop_w]
        return img

    # 将predict后的图片补全其余通道的颜色，并转换为BGR格式
    def colorize(self, y_img, ycrcb_img):
        img = np.zeros((y_img.shape[0], y_img.shape[1], 3), np.uint8)
        img[:, :, 0] = y_img
        img[:, :, 1] = ycrcb_img[:, :, 1]
        img[:, :, 2] = ycrcb_img[:, :, 2]  # 到这里恢复的都是YCrCb下的图片，后面一个函数将其转换到RGB中
        img = cv2.cvtColor(img, code=cv2.COLOR_YCrCb2BGR)
        return img

evaluator = EvalModel(model_path="C:\\Users\\lenovo\PycharmProjects\Super_Rosultion\Model\Image291\VDSR_scale234.pth",
                      patch_size=41, cuda=False,downsize=5)
#evaluator.dataset_test(path="C:\\Users\\lenovo\\PycharmProjects\\Super_Rosultion\\Data\\UAV_data\\test", scale=4)
evaluator.image_test(img_path="C:\\Users\lenovo\PycharmProjects\Super_Rosultion\Data\\UAV_data\\test\DJI_0304.JPG", scale=2)


















