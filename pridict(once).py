import os
import json
from torchvision import transforms, datasets, models
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn

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
        [transforms.Resize([224,224]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "./dataset/valid/5/3550491463_3eb092054c_m.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img_0 = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img_0)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'flower_to_name.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

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

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3f}".format(class_indict[str(predict_cla + 1)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    # 创建图像和柱状图
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原始图像
    ax[0].imshow(img_0)
    ax[0].set_title(print_res)
    ax[0].axis('off')  # 隐藏坐标轴

    # 创建柱状图
    ax[1].bar(range(len(predict)), predict.numpy())
    ax[1].set_xticks(range(len(predict)))
    ax[1].set_xticklabels([class_indict[str(i + 1)] for i in range(len(predict))],
                          rotation=45, ha='right')
    ax[1].set_ylabel('Probability')
    ax[1].set_title('Class Probabilities')

    plt.tight_layout()
    plt.savefig("prediction_once.png")
    plt.show()
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3f}".format(class_indict[str(i+1)],
                                                  predict[i].numpy()))
    # plt.imshow(img_0) # Display the image after transforms
    # plt.savefig("prediction_once.png")
    # plt.show()


if __name__ == '__main__':
    main()