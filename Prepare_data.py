import cv2
import numpy as np
import os
import random



'''bug:数据数量不是batch_zise的整数倍，如何处理
        顺序读取文件夹中的图像块，需要提前打乱    
'''
# 用于keras中自定义图片读取
def Image_generator(ILR_path, HR_path, batch_size):
    ILR_batch = []
    Res_batch = []

    ILR_dir = os.listdir(ILR_path)
    HR_dir = os.listdir(HR_path)
    while(True):
        count = 1
        for ILR_img_name, HR_img_name in zip(ILR_dir, HR_dir):  # 获取本batch中X数据
            ILR_img = cv2.imread(ILR_path + '/' + ILR_img_name, cv2.IMREAD_COLOR)
            HR_img = cv2.imread(HR_path + '/' + HR_img_name, cv2.IMREAD_COLOR)

            Residual_img = HR_img - ILR_img
            ILR_batch.append(ILR_img/255.0)
            Res_batch.append(Residual_img/255.0)

            count += 1
            if(count % batch_size==0):
                yield (np.array(ILR_batch), np.array(Res_batch))  # 返回本批次数据
                ILR_batch.clear()  # 重新清空列表
                Res_batch.clear()

''' (1)划分训练集和验证集（分成两个文件夹另外生成即可）
    (2)生成多尺度图片，则scale传入为列表，然后循环多次调用spilt_image
    (3)数据增强可选，因为验证集，测试集不需要使用数据增强
    (4)可以参考github实现，加入downsize采样+多尺度+旋转+裁剪
'''
# 用于将图片分块后存储在本地
class Image_Preprocess():
    def __init__(self, Source_path, ILR_path, HR_path, patch_size, stride=None):
        self.Source_path = Source_path  # 原始图像所在路径
        self.ILR_path = ILR_path    # 分块后ILR图像所在路径
        self.HR_path = HR_path      # 分块后HR图像所在路径
        self.patch_size = patch_size # 图像分块后块的大小（正方形）
        if (stride == None):
            self.stride = patch_size  # 图像分块时的步长(不指定特定步长则不重叠)
        self.name_list = random.sample(range(1, 1000000), 100000)  # 产生一个包含范围在（a,b）之中的不重复随机数列表
        self.img_patch_num = 0  # 统计生成patch对的总数

    def Generate_Data(self, scale):
        i = 0
        for img_name in os.listdir(self.Source_path):  # os.listdir返回当前文件夹下所有文件
            file_path = self.Source_path + '\\' + img_name
            HR_img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # 读入数据（三通道）
            HR_shape = (HR_img.shape[1], HR_img.shape[0])  # dim = (width, height)
            for s in scale:
                LR_shape = (HR_img.shape[1] // s, HR_img.shape[0] // s)  # 确定按比例放缩后的size
                LR_img = cv2.resize(HR_img, LR_shape, interpolation=cv2.INTER_CUBIC)  # 使用双线性三次插值方法
                ILR_img = cv2.resize(LR_img, HR_shape, interpolation=cv2.INTER_CUBIC)  # 插值恢复
                self.Spilt_Image(HR_img=HR_img, ILR_img=ILR_img)  # 图像分块并存储
            if(i%10 == 0):
                print("当前处理的图片序号为：", i)
            i += 1
        print("生成图像块总数：", self.img_patch_num)

    def Spilt_Image(self, HR_img, ILR_img):
        width_num = (HR_img.shape[1]-self.patch_size)//self.stride  # 确定横向上可以取多少个块  ( HR\ILR等大)
        height_num = (HR_img.shape[0]-self.patch_size)//self.stride  # 确定纵向上可以取多少个块
        for i in range(0, height_num+1):
            for j in range(0, width_num+1):
                HR_img_patch = HR_img[(0 + i * self.patch_size):(self.patch_size + i * self.patch_size),
                                      (0 + j * self.patch_size):(self.patch_size + j * self.patch_size), :]
                ILR_img_patch = ILR_img[(0 + i * self.patch_size):(self.patch_size + i * self.patch_size),
                                        (0 + j * self.patch_size):(self.patch_size + j * self.patch_size), :]
                '''数据增强'''
                cv2.imwrite(self.HR_path+"\\"+str(self.name_list[self.img_patch_num])+"_HR.jpg", HR_img_patch)
                cv2.imwrite(self.ILR_path+"\\"+str(self.name_list[self.img_patch_num])+"_ILR.jpg", ILR_img_patch)
                self.img_patch_num += 1

# 用于将图片分块后存储成一个numpy结构  只保存Ycbcr格式中的Y通道
class ImageToNumpy():
    def __init__(self, sourcePath, ILR_path, HR_path, patchSize, stride=None, dataAugmentation=True, downsize=None):
        self.sourcePath = sourcePath # 原始图像存储路径
        self.ILR_path = ILR_path    # 分块后ILR图像的numpy存储路径
        self.HR_path = HR_path      # 分块后HR图像的numpy存储路径
        self.patchSize = patchSize  # 图像分块后块的大小（正方形）
        self.dataAugmentation = dataAugmentation # 是否进行图像增强
        if (stride == None):
            self.stride = patchSize  # 图像分块时的步长(不指定特定步长则不重叠)
        else:
            self.stride = stride
        self.ImgPatchNum = 0  # 统计生成patch对的总数
        self.ILR_data = []
        self.HR_data = []

    def generate_data(self, scale, downsize):
        image_count = 0
        if len(downsize) == 0:
            print("请输入图片降采样尺度！")
            return 0
        for img_name in os.listdir(self.sourcePath):  # os.listdir返回当前文件夹下所有文件
            file_path = self.sourcePath + '\\' + img_name
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # 读入数据（三通道）
            for d_size in downsize:  # 对图片进行不同尺度的降采样
                if d_size != 1:
                    HR_img = cv2.resize(img, dsize=(0, 0), fx=d_size, fy=d_size)  # x\y方向缩小到原理的d_size分之一
                else:
                    HR_img = img
                HR_shape = (HR_img.shape[1], HR_img.shape[0])  # dim = (width, height) 这个和矩阵的shape反过来
                for s in scale:
                    LR_shape = (HR_img.shape[1] // s, HR_img.shape[0] // s)  # 确定按比例放缩后的size
                    LR_img = cv2.resize(HR_img, LR_shape, interpolation=cv2.INTER_CUBIC)  # 使用双线性三次插值方法
                    ILR_img = cv2.resize(LR_img, HR_shape, interpolation=cv2.INTER_CUBIC)  # 插值恢复
                    if (self.dataAugmentation == True):
                        x_result, y_result = self.image_augmentation(ILR_img, HR_img)
                    else:
                        x_result = [ILR_img]  # 不做增强，则列表只有原图
                        y_result = [HR_img]
                    self.spilt_image(HR_list=y_result, ILR_list=x_result)  # 图像分块并存储
            image_count += 1
            if image_count % 5 == 0:
                print("当前处理图片数量为：", image_count)

        print("所有图片处理完成，总图像块数量：", self.ImgPatchNum)
        ILR_data = np.array(self.ILR_data)
        HR_data = np.array(self.HR_data)
        print("ILR_data.shape: ", ILR_data.shape)
        print("HR_data.shape: ", HR_data.shape)
        #np.save(file=self.ILR_path, arr=ILR_data)
        #np.save(file=self.HR_path, arr=HR_data)
        print("处理完毕")

    def image_augmentation(self, ILR_img, HR_img):
        x_result = []
        y_result = []
        rotateCode = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

        # 依次为垂直翻转、水平翻转
        for i in range(0, 2):
            # 依次顺时针旋转0，90，180，270
            for j in range(0, 4):
                xImg = cv2.flip(ILR_img, i, dst=None)
                yImg = cv2.flip(HR_img, i, dst=None)
                if j>0:
                    xImg = cv2.rotate(xImg,rotateCode=rotateCode[j-1])
                    yImg = cv2.rotate(yImg, rotateCode=rotateCode[j-1])
                x_result.append(xImg)
                y_result.append(yImg)
        return x_result, y_result  # 返回增强后的结果

    def spilt_image(self, HR_list, ILR_list):
        for HR_img, ILR_img in zip(HR_list, ILR_list):
            width_num = (HR_img.shape[1] - self.patchSize) // self.stride  # 确定横向上可以取多少个块  ( HR\ILR等大)
            height_num = (HR_img.shape[0] - self.patchSize) // self.stride  # 确定纵向上可以取多少个块  去掉边缘
            for i in range(0, height_num + 1):
                for j in range(0, width_num + 1):
                    HR_ImgPatch = HR_img[(0 + i * self.patchSize):(self.patchSize + i * self.patchSize),
                                  (0 + j * self.patchSize):(self.patchSize + j * self.patchSize), :]
                    ILR_ImgPatch = ILR_img[(0 + i * self.patchSize):(self.patchSize + i * self.patchSize),
                                   (0 + j * self.patchSize):(self.patchSize + j * self.patchSize), :]
                    HR_ImgPatch = cv2.cvtColor(HR_ImgPatch,cv2.COLOR_BGR2YCrCb)  # 颜色通道转换
                    ILR_ImgPatch = cv2.cvtColor(ILR_ImgPatch, cv2.COLOR_BGR2YCrCb)
                    self.ILR_data.append(ILR_ImgPatch[:, :, 0])
                    self.HR_data.append(HR_ImgPatch[:, :, 0])  # 只存储Y通道下的图片 且不做归一化
                    self.ImgPatchNum += 1


train_data = ImageToNumpy(sourcePath="C:\\Users\\lenovo\\PycharmProjects\\Super_Rosultion\\Data\\Image_291",
                          ILR_path="",
                          HR_path="",
                          patchSize=32, dataAugmentation=True)
train_data.generate_data(scale=[2, 3, 4],downsize=[1, 0.7, 0.5])


























