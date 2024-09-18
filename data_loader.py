import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import torch.nn.functional as F
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cuda')
# def normalize_tensor(tensor):
#     mean = tensor.mean(dim=0, keepdim=True)
#     std = tensor.std(dim=0, keepdim=True)
#     return (tensor - mean) / std
# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='Script to set batch size.')

# 添加一个命令行参数
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loaders')
parser.add_argument('--epoch', type=int, default=50, help='epoch')
parser.add_argument('--lr', type=float, default=0.00001, help='epoch')
# 解析命令行参数
args = parser.parse_args()

# 将命令行参数赋值给batch_size变量
batch_size = args.batch_size
epoch=args.epoch
learning_rate=args.lr

data = pd.read_csv('train.csv').values.astype(np.float32)
data = torch.tensor(data)

# 假设data是一个PyTorch张量
inputs = torch.stack([data[:, i] for i in [1,2,3,4,5,6,7]], dim=1)

inputs.requires_grad_()


targets = torch.stack([data[:, i] for i in [8]], dim=1)

targets.requires_grad_()
inputs_min = torch.min(inputs, dim=0).values
inputs_max = torch.max(inputs, dim=0).values
targets_min = torch.min(targets, dim=0).values
targets_max = torch.max(targets, dim=0).values

inputs_normalized = (inputs - inputs_min) / (inputs_max - inputs_min)
targets_normalized = (targets - targets_min) / (targets_max - targets_min)
X_train, X_remaining, y_train, y_remaining = train_test_split(inputs_normalized, targets_normalized, test_size=0.1, random_state=42)
# 划分验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.1, random_state=42)

# 将数据封装成TensorDataset
X_train=X_train.to(device)
X_test=X_test.to(device)
y_train=y_train.to(device)
y_test=y_test.to(device)
X_val=X_val.to(device)
y_val=y_val.to(device)
# 将数据封装成TensorDataset
train_dataset = TensorDataset(X_train, y_train)

val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
# # 定义数据加载器
train_loader_normal = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader_normal = DataLoader(val_dataset, batch_size=batch_size)
test_loader_normal = DataLoader(test_dataset, batch_size=batch_size)
inputs_min=inputs_min.to(device)
inputs_max=inputs_max.to(device)
targets_min=targets_min.to(device)
targets_max=targets_max.to(device)