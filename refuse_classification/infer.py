from dataset import Garbage_Loader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def st(x):
    exp_x = np.exp(x)
    st_x = exp_x / np.sum(exp_x, 0)
    return st_x


with open('dir_label.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    labels = list(map(lambda x: x.strip().split('\t'), labels))

if __name__ == "__main__":
    test_list = 'test.txt'
    test_data = Garbage_Loader(test_list, train_flag=False)
    test_loader = DataLoader(dataset=test_data, num_workers=1, pin_memory=True, batch_size=1)
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 214)
    model = model.to(device)
    # 加载训练好的模型
    checkpoint = torch.load('model_best_checkpoint_resnet50.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for i, (image, label) in enumerate(test_loader):
        src = image.numpy()
        src = src.reshape(3, 224, 224)
        src = np.transpose(src, (1, 2, 0))
        image = image.to(device)
        label = label.to(device)
        pred = model(image)
        pred = pred.data.cpu().numpy()[0]
        score = st(pred)
        pred_id = np.argmax(score)
        plt.imshow(src)
        print('预测结果：', labels[pred_id][0])
        plt.show()
