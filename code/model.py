import torch
import torch.nn as nn
input_size=2
hidden_size=100
output_size=5
class Net(nn.Module):
    #初始化网络结构
    def __init__(self, input_size, hidden_size, outputsize):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #输入层，线性（liner）关系
        self.relu = nn.ReLU()#隐藏层，使用ReLU函数
        self.fc2 = nn.Linear(hidden_size, hidden_size)  #输出层，线性（liner）关系
        self.relu = nn.ReLU()#隐藏层，使用ReLU函数
        self.fc3 = nn.Linear(hidden_size, hidden_size)  #输出层，线性（liner）关系
        self.relu = nn.ReLU()#隐藏层，使用ReLU函数
        self.fc4 = nn.Linear(hidden_size, outputsize)  #输出层，线性（liner）关系
    #forword 参数传递函数，网络中数据的流动
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out