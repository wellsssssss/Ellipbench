import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data_loader
from sklearn.model_selection import train_test_split
from math import sin,pi,cos,exp
from numpy import arcsin
import cmath
import csv
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import data_loader
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def N(n,k):
    n=float(n)
    k=float(k)
    N=complex(n,k)
    return N
def rp_1_2(N1,N2):#x1
    theta1=(64.5/180)*pi
    theta2=arcsin(sin(theta1)/N2)
    rp_1_2=(N2*cos(theta1)-N1*cos(theta2))/(N2*cos(theta1)+N1*cos(theta2))
    return rp_1_2
def rp_2_3(N2,N3):#x2
    theta1=(64.5/180)*pi
    theta2=arcsin(sin(theta1)/N2)
    theta3=arcsin(sin(theta1)/N3)
    rp_2_3=(N3 * cos(theta2) - N2 * cos(theta3)) / (N3 * cos(theta2) + N2 * cos(theta3))
    return rp_2_3
def rs_1_2(N1,N2):#y1
    theta1=(64.5/180)*pi
    theta2=arcsin(sin(theta1)/N2)
    rs_1_2=(N1*cos(theta1)-N2*cos(theta2))/(N1*cos(theta1)+N2*cos(theta2))
    return rs_1_2
def rs_2_3(N2,N3):#y2
    theta1=(64.5/180)*pi
    theta2=arcsin(sin(theta1)/N2)
    theta3=arcsin(sin(theta1)/N3)
    rs_2_3=(N2*cos(theta2)-N3*cos(theta3))/(N2*cos(theta2)+N3*cos(theta3))
    return rs_2_3
def t(N2,lamda,d):
    lamda=float(lamda)
    d=float(d)
    theta1=(64.5/180)*pi
    theta2=arcsin(sin(theta1)/N2)
    beta = 2 * pi * (d / lamda) * N2 * cos(theta2)
    t=cmath.exp(complex(0,-2*beta))
    return t
def result(a,b):
    result=cmath.tan(a)*cmath.exp(complex(0,b))
    return result
# def loss(n2,k2,n3,k3,d,lamda,a,b):
with open('output.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = []
    for row in csv_reader:
        data.append(row)
    data = np.array(data)
    data = data.astype(float)
# data = data_loader.csv_data
input1 = data[:,1]
input2 = data[:,2]
inputs=np.column_stack((input1,input2))
inputs=torch.tensor(inputs)
inputs=inputs.float()
n2 = data[:,3]
k2 = data[:,4]
n3 = data[:,5]
k3 = data[:,6]
d  = data[:,7]
# lamda=data[:,0]
targets=np.column_stack((n2,k2,n3,k3,d))
targets=torch.tensor(targets)
targets=targets.float()
X_train, X_test, y_train, y_test = train_test_split(inputs,targets, test_size=0.2, random_state=42)
# 将数据封装成TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
# 定义数据加载器
batch_size = 32  # 设置批大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
def loss(y_predict,y_true,csv_data):
    csv_data=np.array(csv_data)
    n2, k2, n3, k3, d, lamda=y_predict
    n2_1, k2_1, n3_1, k3_1, d_1, lamda_1=y_true
    mask = (csv_data[:, 3] == n2_1) & (csv_data[:, 4] == k2_1) & (csv_data[:, 5] == n3_1) & (csv_data[:, 6] == k3_1) & (
                csv_data[:, 7] == d_1)
    lamda = csv_data[mask, 0][0]
    N2=N(n2,k2)
    N3=N(n3,k3)
    rp12=rp_1_2(1,N2)
    rp23=rp_2_3(N2,N3)
    rs12=rs_1_2(1,N2)
    rs23=rs_2_3(N2,N3)
    t_value=t(N2,lamda,d)

    N2_1=N(n2,k2)
    N3_1=N(n3,k3)
    rp12_1=rp_1_2(1,N2_1)
    rp23_1=rp_2_3(N2_1,N3_1)
    rs12_1=rs_1_2(1,N2_1)
    rs23_1=rs_2_3(N2_1,N3_1)
    t_value_1=t(N2_1,lamda,d_1)

    result_cal=((rp12+rp23*t_value)/(1+rp12*rp23*t_value))/((rs12+rs23*t_value)/(1+rs12*rs23*t_value))
    result_true=((rp12_1+rp23_1*t_value_1)/(1+rp12_1*rp23_1*t_value_1))/((rs12_1+rs23_1*t_value_1)/(1+rs12_1*rs23_1*t_value_1))
    loss = np.square(result_cal.real - result_true.real) + np.square(result_cal.imag - result_cal.imag)
    return loss
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(2, 50)  # 输入层到隐藏层
        self.fc2 = nn.Linear(50, 100)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(100, 150)
        self.fc4 = nn.Linear(150, 5)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
def train(model, train_input, train_target, num_epochs, learning_rate, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_input = train_input.to(device)
    train_target = train_target.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_samples = train_input.size(0)

    train_dataset = torch.utils.data.TensorDataset(train_input, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_input, batch_target in train_loader:
            optimizer.zero_grad()
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            output = model(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
        print('训练次数：', epoch)
        print('损失：', loss.item())
def test(model, test_input, test_target):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    output = model(test_input)
    mse = nn.MSELoss()
    loss = mse(output, test_target)
    return loss.item()

# 创建模型实例
model = RegressionNet()

"""train(model, X_train, y_train, num_epochs=500, learning_rate=0.01,batch_size=10000)
# 测试模型
accuracy = test(model, X_test, y_test)
print("测试集准确率：", accuracy)
filename = 'D:/regression_model2.pth'
# 保存模型参数
torch.save(model.state_dict(), filename)"""

# 创建模型实例
model = RegressionNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('训练次数：', epoch, '损失：', loss)

# 设置模型为评估模式
model.eval()

# 运行模型进行预测
with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        test_outputs = model(inputs)
        test_loss += criterion(test_outputs, labels).item()

    mse = test_loss / len(test_loader)
    print("Mean Squared Error:", mse)
with open('mlp_model2.pkl', 'wb') as file:
    pickle.dump(model, file)






























"""criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 训练模型
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
# 设置模型为评估模式
model.eval()

# 运行模型进行预测
with torch.no_grad():
    test_outputs = model(X_test)

# 计算均方误差
mse = criterion(test_outputs, y_test)
print("Mean Squared Error:", mse.item())
with open('D:/mlp_model2.pkl', 'wb') as file:
    pickle.dump(model, file)"""



