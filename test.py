import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 256*2)
        self.fc2 = nn.Linear(256*2, 256*4)
        self.fc3 = nn.Linear(256*4, 256*4)
        self.fc4 = nn.Linear(256*4, 256*4)
        self.fc5 = nn.Linear(256*4, 256*4)
        self.fc6 = nn.Linear(256*4, 256*4)
        self.fc7 = nn.Linear(256*4, 256*4)
        self.fc8 = nn.Linear(256*4, 256*4)
        self.fc9 = nn.Linear(256*4, 256)
        self.fc10 = nn.Linear(256, 6)  # 将输出维度改为6

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = self.fc10(x)  # 移除sigmoid激活函数
        return x
# 加载测试数据
test_data = pd.read_csv('test.csv')
test_ids = test_data.iloc[:, 0]  # 第一列是id
test_features = test_data.iloc[:, 1:]  # 剩下的列是特征

# 将测试特征转换为PyTorch张量
test_features = torch.tensor(test_features.values, dtype=torch.float32)

# 加载训练好的模型参数
model = Net()  # 这里假设你的模型类名为Net
model.load_state_dict(torch.load('model'))
model.eval()

# 对测试数据进行预测
with torch.no_grad():
    outputs = model(test_features)
    predicted = (outputs > 0.5).float()

# 将预测结果转换为列表
predicted_list = predicted.squeeze().tolist()

# 创建一个DataFrame,包含id和预测结果
result_df = pd.DataFrame({'id': test_ids, 'predicted': predicted_list})

# 将结果保存到新的csv文件
result_df.to_csv('prediction_result.csv', index=False)