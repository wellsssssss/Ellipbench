import torch
from model import Net,output_size,input_size,hidden_size
import torch.optim as optim
import torch.nn as nn
from data_loader import train_loader,test_loader,val_loader,data,train_dataset,test_dataset,val_dataset,batch_size,learning_rate,epoch
from loss_function import loss_1,acc
import matplotlib.pyplot as plt
model = Net(input_size,hidden_size,output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 定义优化器
criterion = nn.MSELoss()
# 定义损失函数和优化器
train_losses = []
val_accuracies = []
num_epochs = epoch  # 设置训练的轮数
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        # 前向传播
        inputs=inputs.to(device)
        targets=targets.to(device)
        outputs = model(inputs)
        outputs=outputs.to(device)
        loss = criterion(outputs,targets)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.requires_grad_(True)
        #         loss_value.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    # 验证模式
    model.eval()
    val_loss = 0.0
    correct = 0
    acc_val=0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss=criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            acc_val+=acc(outputs,targets)
    # 打印训练和验证的损失和准确率
    train_loss = train_loss / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    val_accuracy = acc_val / len(val_dataset)
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
model.eval()
test_loss = 0.0
correct = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs=inputs.to(device)
        targets=targets.to(device)
        outputs = model(inputs)
        loss = loss_1(outputs,targets,inputs,data)

        test_loss += loss.item() * inputs.size(0)
        correct += acc(outputs,targets)
# 计算测试集上的损失和准确率
test_loss = test_loss / len(test_dataset)
test_accuracy = correct / len(test_dataset)
torch.save(model.state_dict(), 'model_parameters.pth')
# 绘制训练集损失下降曲线
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
# 绘制验证集准确率变化曲线
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
