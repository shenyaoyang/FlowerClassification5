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
#初始化模型，适配任务
def initialize_model(model_name,num_classes, feature_extract, use_pretrained= True):

    model_ft = models.resnet50(pretrained = use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)#定义成自己的分类任务类别

    input_size = 224#输入数据大小

    return model_ft, input_size




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # create model
    model_name = 'resnet'
    model, input_size = initialize_model(model_name, 5, feature_extract=True, use_pretrained=True)
    model = model.to(device)
    # load model weights
    filename = r'F:\pytorch项目\图像分类\best_model_weight.pt'
    checkpoint = torch.load(filename, map_location=device)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    assert os.path.exists(filename), "file: '{}' dose not exist.".format(filename)
    model.eval()

    # 读取类别字典
    json_path = 'flower_to_name.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 测试文件夹路径
    test_dir = "./dataset/test"

    # 选择10张随机图片
    image_paths = random.sample(os.listdir(test_dir), 10)

    # 创建图片展示
    fig = plt.figure(figsize=(25, 20))
    column = 5
    rows = 2

    for idx, image_path in enumerate(image_paths):
        # 加载图片
        img_path = os.path.join(test_dir, image_path)
        img_0 = Image.open(img_path)
        img = data_transform(img_0)
        img = torch.unsqueeze(img, dim=0).to(device)

        # 预测类别
        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # 显示图片和预测类别
        ax = fig.add_subplot(rows, column, idx + 1)
        plt.imshow(img_0)
        ax.set_title(f"{class_indict[str(predict_cla + 1)]}", color="green", fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])

    # 保存图片
    plt.savefig("prediction_result.png")
    plt.show()



if __name__ == '__main__':
    main()