import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='Script to set batch size.')

# 添加一个命令行参数
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for data loaders')
parser.add_argument('--epoch', type=int, default=1000, help='epoch')
parser.add_argument('--lr', type=float, default=0.01, help='epoch')
# 解析命令行参数
args = parser.parse_args()

# 将命令行参数赋值给batch_size变量
batch_size = args.batch_size
epoch=args.epoch
learning_rate=args.lr



data = pd.read_csv('output.csv').values.astype(np.float32)
data = torch.tensor(data)
# data = data_loader.csv_data
input1 = data[:,1]
input2 = data[:,2]
inputs=np.column_stack((input1,input2))
inputs=torch.tensor(inputs,requires_grad=True)
# inputs=inputs.float()
n2 = data[:,3]
k2 = data[:,4]
n3 = data[:,5]
k3 = data[:,6]
d  = data[:,7]
# lamda=data[:,0]
targets=np.column_stack((n2,k2,n3,k3,d))
targets=torch.tensor(targets,requires_grad=True)
# targets=targets.float()
# 划分训练集和剩余数据
X_train, X_remaining, y_train, y_remaining = train_test_split(inputs, targets, test_size=0.3, random_state=42)

# 划分验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.2, random_state=42)
# 将数据封装成TensorDataset
# 将数据封装成TensorDataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
