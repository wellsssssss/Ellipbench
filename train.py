import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data_loader import train_loader_normal,test_loader_normal,val_loader_normal,data,train_dataset,test_dataset,val_dataset,batch_size,learning_rate,epoch,y_train,X_train,X_test,y_test,targets_min,targets_max
from ResNet_nn import Net  # 确保这个是你的网络模型

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

# 使用适合二分类问题的损失函数
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = epoch
model_save_dir = "model_checkpoints"
os.makedirs(model_save_dir, exist_ok=True)

with open('training_log.txt', 'w') as log_file:
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_loader_normal), desc='Training Progress')
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader_normal:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1).float())  # 确保targets是正确的形状和类型
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.update(1)
        progress_bar.close()

        # 验证模式
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader_normal:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1).float())
                val_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))  # 使用sigmoid激活函数并四舍五入得到预测类别
                total += targets.size(0)
                correct += (predicted == targets.unsqueeze(1)).sum().item()

        val_accuracy = 100 * correct / total

        log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")
        log_file.flush()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth'))

model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader_normal:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1).float())
        test_loss += loss.item()
        predicted = torch.round(torch.sigmoid(outputs))
        total += targets.size(0)
        correct += (predicted == targets.unsqueeze(1)).sum().item()

test_accuracy = 100 * correct / total
torch.save(model.state_dict(), 'model_parameters_final.pth')
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")