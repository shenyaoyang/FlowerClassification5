import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

#冻结模型的层数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad =False

#将模型输出层改成自己需要的目标
def initialize_model(model_name,num_classes, feature_extract, use_pretrained= True):

    model_ft = models.resnet50(pretrained = use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)#定义成自己的分类任务类别

    input_size = 224#输入数据大小

    return model_ft, input_size

#训练模型
def train_model(model, dataloaders, loss_function, optimizer, num_epochs, filename):
    since = time.time()#记录时间
    best_acc = 0#记录最好的一次
    model.to(device)
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    lr = [optimizer.param_groups[0]['lr']]
    best_model_weights = copy.deepcopy(model.state_dict())
    #遍历epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            #遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #梯度清零
                optimizer.zero_grad()
                #训练时计算、更新梯度
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                _, preds = torch.max(outputs, 1)
                #训练阶段更新权重
                if phase == 'train':
                    loss.backward()#反向传播
                    optimizer.step()#参数更新

                #计算损失
                running_loss += loss.item() * inputs.size(0) # loss 乘以 batch
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time()-since#一次epoch耗时
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
            print('{} Loss :{:.4f}   Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            #最好的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weight = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)

        print('Optimizer learning rate: {:.7f}'.format(optimizer.param_groups[0]['lr']))
        lr.append(optimizer.param_groups[0]['lr'])
        print()
        scheduler.step()#学习率衰减

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best Val Acc:{:.4f}'.format(best_acc))

    #加载训练效果最佳模型，作为后续预测基础
    model.load_state_dict(best_model_weight)

    return model, val_acc, train_acc, val_losses, train_losses, lr

#绘制训练曲线
def plot_training_history(train_losses, val_losses, train_acc, val_acc):
    plt.figure(figsize=(12, 6))  # Adjust figure size if needed

    # 获取数据列表的长度（即 epoch 数量）
    epochs = range(1, len(train_losses) + 1)

    # Plot losses (left y-axis)
    plt.plot(epochs, train_losses, label='Training Loss', linestyle='-', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', linestyle='--', color='blue')
    plt.ylabel('Loss', color='blue')  # Set y-axis label for loss
    plt.legend(loc='upper left')  # Place legend in the upper left corner

    # Create a second y-axis for accuracy
    ax2 = plt.twinx()  # Create a twin axis
    ax2.plot(epochs, train_acc, label='Training Accuracy', linestyle='-', color='#FFC300')
    ax2.plot(epochs, val_acc, label='Validation Accuracy', linestyle='--',color='#FFC300')
    ax2.set_ylabel('Accuracy', color='#FFC300')  # Set y-axis label for accuracy
    ax2.legend(loc='upper right')  # Place legend in the upper right corner

    plt.xlabel('Epochs')  # Set x-axis label
    plt.title('Training and Validation History')
    plt.show()


# 数据集读取
data_dir = './dataset/'
train_dir = data_dir+'/train'
val_dir = data_dir+'valid'

# 预处理
data_transform = {
    'train':
        transforms.Compose([
        transforms.RandomResizedCrop([224,224]),#随机调整尺寸、裁剪至224x224
        transforms.RandomHorizontalFlip(p=0.5),#水平翻转，概率为0.5
        transforms.RandomVerticalFlip(p=0.5),#垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#色域调整，参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换为灰度
        transforms.ToTensor(),#转换为灰度张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#归一化，参数1为均值，参数2为标准差
    ]),
    'valid':
        transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

#batch定义以及数据集加载、处理
batch_size = 32
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['train','valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle= True) for x in ['train','valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

#读取对应类别标签
with open ('flower_to_name.json', 'r') as f:
    flower_to_name = json.load(f)

# GPU训练判断
train_to_gpu = torch.cuda.is_available()
if not train_to_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#模型初始化
model_name = 'resnet'
feature_extract = True
model_ft, input_size = initialize_model(model_name,5, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

#模型训练权重存储路径
filename = r'F:\pytorch项目\图像分类\best_model_weight.pt'

#是否训练所有层
params_to_update = model_ft.parameters()
# print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

####初次训练####
# #优化器设置
# optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
# #学习率衰减
# scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma= 0.2)#每10个epoch衰减为原来的1/10
# loss_function = nn.CrossEntropyLoss()
# #开始进行第一次训练(只训练自定义的fc层)
# model_ft, val_acc, train_acc, val_loss, train_loss, lr =train_model(model_ft, dataloaders, loss_function, optimizer_ft, 20 , filename)
# #绘制训练曲线
# plot_training_history(train_loss,val_loss, train_acc, val_acc)

#再次训练所有层（有fc层训练权重的前提下）
for param in model_ft.parameters():
    param.requires_grad = True

#二次设置超参数，进行微调
optimizer_ft = optim.Adam(params_to_update, lr=1e-3)
#学习率衰减
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma= 0.2)#每10个epoch衰减为原来的1/10
loss_function = nn.CrossEntropyLoss()
#进行第二次训练(只训练自定义的fc层)
model_ft, val_acc, train_acc, val_loss, train_loss, lr =train_model(model_ft, dataloaders, loss_function, optimizer_ft, 20 , filename)
#二次绘制曲线
plot_training_history(train_loss,val_loss, train_acc, val_acc)







