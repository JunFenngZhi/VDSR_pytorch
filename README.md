# VDSR_pytorch
 使用pytorch实现VDSR算法。在BSD200和T91数据集上进行训练，在Set5和Set14数据集上进行测试。

## 模块功能说明
Prepare_data.py: 处理数据，将图片进行预处理和数据增强。将处理结果存储在numpy中，便于后续训练时统一读取。  
VDSR_model.py: 实现VDSR网络。  
Main_code.py:  &nbsp;&nbsp;&nbsp;读取数据，训练网络。  
Eval_model.py: &nbsp;&nbsp;对训练得到的网络进行评估。获取超分结果以及计算PSNR。  

## 超分结果展示
Scale = 2  
![Image text](https://github.com/JunFenngZhi/VDSR_pytorch/blob/master/Results/butterfly_scale2.JPG)  
![Image text](https://github.com/JunFenngZhi/VDSR_pytorch/blob/master/Results/coastguard_scale2.JPG)  
  
  
Scale = 3  
![Image text](https://github.com/JunFenngZhi/VDSR_pytorch/blob/master/Results/butterfly_scale3.JPG)  
![Image text](https://github.com/JunFenngZhi/VDSR_pytorch/blob/master/Results/coastguard_scale3.JPG)  
![Image text](https://github.com/JunFenngZhi/VDSR_pytorch/blob/master/Results/comic_scale3.JPG)  
  
     
Scale = 4  
![Image text](https://github.com/JunFenngZhi/VDSR_pytorch/blob/master/Results/butterfly_scale4.JPG)  
![Image text](https://github.com/JunFenngZhi/VDSR_pytorch/blob/master/Results/comic_scale4.JPG)  

## 存在的问题
网络在测试集上的PSNR较低，和论文结果相差较大。但图片恢复效果较好。
