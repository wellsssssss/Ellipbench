import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.elu = nn.ELU()

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Linear(in_channels, out_channels)

        self.residual_mapping = nn.Linear(in_channels, out_channels)  # 残差映射层

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # 计算残差映射
        if self.downsample is not None:
            residual = self.downsample(residual)
        residual = self.residual_mapping(residual)

        x = x + residual
        x = self.elu(x)

        return x
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(5, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.layer1 = ResidualBlock(512, 512)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.layer2 = ResidualBlock(512,512)

        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.layer3 = ResidualBlock(1024,1024)

        self.fc4 = nn.Linear(1024,1024)
        self.bn4 = nn.BatchNorm1d(1024)

        self.layer4 = ResidualBlock(1024,1024)

        self.fc5 = nn.Linear(1024,2048)
        self.bn5 = nn.BatchNorm1d(2048)

        self.layer5 = ResidualBlock(2048,2048)

        self.fc6 = nn.Linear(2048,4096)
        self.bn6 = nn.BatchNorm1d(4096)

        self.layer6 = ResidualBlock(4096,4096)

        self.fc7 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        
        self.layer7 = ResidualBlock(4096, 4096)

        self.fc8 = nn.Linear(4096, 4096*2)
        self.bn8 = nn.BatchNorm1d(4096 * 2)
        
        self.layer8 = ResidualBlock(4096 * 2, 4096 * 2)

        self.fc9 = nn.Linear(4096 * 2, 4096 * 2)
        self.bn9 = nn.BatchNorm1d(4096 * 2)
        
        self.layer9 = ResidualBlock(4096 * 2, 4096 * 2)
        self.fc10 = nn.Linear(4096 * 2, 4096 * 2)
        self.bn10 = nn.BatchNorm1d(4096 * 2)
        
        self.layer10 = ResidualBlock(4096 * 2, 4096 * 2)
        self.fc11 = nn.Linear(4096 * 2, 3)
#         self.fc1 = nn.Linear(5, 512)
#         self.bn1 = nn.BatchNorm1d(512)

#         self.layer1 = ResidualBlock(512, 512)

#         self.fc2 = nn.Linear(512, 1024)
#         self.bn2 = nn.BatchNorm1d(1024)

#         self.layer2 = ResidualBlock(1024, 1024)

#         self.fc3 = nn.Linear(1024, 1024)
#         self.bn3 = nn.BatchNorm1d(1024)

#         self.layer3 = ResidualBlock(1024, 1024)

#         self.fc4 = nn.Linear(1024, 1024)
#         self.bn4 = nn.BatchNorm1d(1024)

#         self.layer4 = ResidualBlock(1024, 1024)

#         self.fc5 = nn.Linear(1024, 1024 * 2)
#         self.bn5 = nn.BatchNorm1d(1024 * 2)

#         self.layer5 = ResidualBlock(1024 * 2, 1024 * 2)

#         self.fc6 = nn.Linear(1024 * 2, 1024 * 2)
#         self.bn6 = nn.BatchNorm1d(1024 * 2)

#         self.layer6 = ResidualBlock(1024 * 2, 1024 * 2)

#         self.fc7 = nn.Linear(1024 * 2, 1024 * 2)
#         self.bn7 = nn.BatchNorm1d(1024 * 2)

#         self.fc8 = nn.Linear(1024 * 2, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.layer1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.layer2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.elu(x)

        x = self.layer3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.elu(x)

        x = self.layer4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.elu(x)

        x = self.layer5(x)

        x = self.fc6(x)
        x = self.bn6(x)
        x = self.elu(x)

        x = self.layer6(x)

        x = self.fc7(x)
        x = self.bn7(x)
        x = self.elu(x)
        
        x = self.layer7(x)
        
        x = self.fc8(x)
        x = self.bn8(x)
        x = self.elu(x)
        
        x = self.layer8(x)
        
        x = self.fc9(x)
        x = self.bn9(x)
        x = self.elu(x)
        
        x = self.layer9(x)
        
        x = self.fc10(x)
        x = self.bn10(x)
        x = self.elu(x)
        
        x = self.layer10(x)
        
        x = self.fc11(x)
        x = nn.Sigmoid()(x)
        return x
