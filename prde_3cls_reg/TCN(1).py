import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchmetrics.regression import R2Score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
# TCN的TemporalBlock定义
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        # print("Residual shape:", residual.shape)
        # print("Out shape:", out.shape)

        if self.downsample is not None:
            residual = self.downsample(residual)

            # 确保residual的时间维度与out的时间维度长度相同
        if out.size(2) < residual.size(2):
            # 如果residual时间维度更长，裁剪residual
            residual = residual[:, :, :out.size(2)]
        elif out.size(2) > residual.size(2):
            # 如果out时间维度更长，填充residual
            padding = torch.zeros(residual.size(0), residual.size(1), out.size(2) - residual.size(2)).to(
                residual.device)
            residual = torch.cat([residual, padding], dim=2)

        return self.relu(out + residual)
class TCNFeatureExtractor(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNFeatureExtractor, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 创建一个共享的特征提取层类
class SharedFeatureExtractor(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super(SharedFeatureExtractor, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.shared_layer(x)

# class load_data(Dataset):
#     def __init__(self, filepath, length=1140, type='train'):
#         self.data = pd.read_excel(filepath, header=None)
#         self.data = self.data.drop(columns=[11, 12, 13, 14])
#         if type == 'train':
#             self.data = self.data.iloc[:length, :]
#         elif type == 'test':
#             self.data = self.data.iloc[length:, :]
#         self.X, self.Y1, self.Y2 = self.data.iloc[:, :-3], self.data.iloc[:, -3], self.data.iloc[:, -2:]
#         self.X = torch.FloatTensor(self.X.values)
#         self.Y1 = torch.LongTensor(self.Y1.values - 1)
#         self.Y2 = torch.FloatTensor(self.Y2.values)
#
#     def __len__(self):
#         return self.X.shape[0]
#
#     def __getitem__(self, index):
#         return self.X[index], self.Y1[index], self.Y2[index]
class load_data(Dataset):
    def __init__(self, filepath, length=1140, type='train'):
        self.data = pd.read_excel(filepath, header=None)
        self.data = self.data.drop(columns=[11, 12, 13, 14])
        # if type == 'train':
        #     self.data = self.data.iloc[:length, :]
        # elif type == 'test':
        #     self.data = self.data.iloc[length:, :]
        self.data = self.data.iloc[length,:]
        self.X, self.Y1, self.Y2 = self.data.iloc[:, :-3], self.data.iloc[:, -3], self.data.iloc[:, -2:]
        self.X = torch.FloatTensor(self.X.values)
        self.Y1 = torch.LongTensor(self.Y1.values - 1)
        self.Y2 = torch.FloatTensor(self.Y2.values)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y1[index], self.Y2[index]
# 深度神经网络模型
class dnn_mlt(nn.Module):
    def __init__(self, input_features, timestep, hidden_dim, num_layers, l1, n3, l2, n4, shared_dim):
        super(dnn_mlt, self).__init__()

        # 创建 TCN 特征提取块
        self.tcn_feature_extractor = TCNFeatureExtractor(input_features, [hidden_dim] * num_layers)

        # 降维层：将 TCN 输出从 56 维降到 8 维
        self.dimension_reduction = nn.Linear(56, 8)

        # 创建共享特征提取层
        self.shared_feature_extractor = SharedFeatureExtractor(8, shared_dim)

        self.dropout = nn.Dropout(l1)

        # 分类网络
        self.subnet1 = nn.Sequential(
            nn.Linear(shared_dim, n3),
            nn.ReLU(),
            nn.Dropout(l2),
            nn.Linear(n3, n4),
            nn.ReLU(),
            nn.Linear(n4, 3),
        )
        # 回归网络
        self.subnet2 = nn.Sequential(
            nn.Linear(shared_dim, n3),
            nn.ReLU(),
            nn.Dropout(l2),
            nn.Linear(n3, n4),
            nn.ReLU(),
            nn.Linear(n4, 2),
        )

    def forward(self, X):
        X = X.permute(0, 2, 1)  # 调整维度以适应 TCN
        tcn_features = self.tcn_feature_extractor(X)
        out = self.dropout(tcn_features)

        # 展平操作
        out = out.view(out.size(0), -1)

        # 应用降维层
        out = self.dimension_reduction(out)

        # 使用共享特征提取层
        shared_features = self.shared_feature_extractor(out)

        # 应用两个子网络
        out_cls = self.subnet1(shared_features)
        out_reg = self.subnet2(shared_features)

        return out_cls, out_reg

if __name__ == '__main__':
    features = 8
    timestep = 1
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 1000
    n1 = 8 # 你的值
    # n2 = 30 # 你的值
    n3 = 20  # 你的值
    n4 = 40  # 你的值
    l1 = 0.2  # 你的值
    l2 = 0.1  # 你的值
    alpha = 0.5  # 你的值
    beta = 0.0001 # 你的值
    shared_dim = 32 # 你的
    hidden_dim = 8  # 设置LSTM的隐藏单元数
    num_layers = 2  # 设置LSTM的层数
    # train, test = load_data(filepath='./data_1.xlsx', length=1140, type='train'), load_data(filepath='./data_1.xlsx', length=1140,type='test')
    choice = np.random.choice(range(1680), size=(1140,), replace=False)
    train_idx = np.zeros(1680, dtype=bool)
    train_idx[choice] = True
    test_idx = ~train_idx
    train, test = load_data(filepath='./data_1.xlsx', length=train_idx, type='train'), load_data(
        filepath='./data_1.xlsx', length=test_idx, type='test')
    # 计算均值和标准差以进行数据归一化
    mean = train.X.mean(dim=0)
    std = train.X.std(dim=0)

    # 归一化训练数据
    train.X = (train.X - mean) / std

    # 归一化测试数据，使用训练数据的均值和标准差
    test.X = (test.X - mean) / std
    train_loader, test_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size,shuffle=False), \
                                torch.utils.data.DataLoader(dataset=test, batch_size=test.X.shape[0],shuffle=True)

    model = dnn_mlt(features, timestep, n1, num_layers, l1, n3, l2, n4, shared_dim).to(device)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=beta)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=beta)
    train_losses, train_accuracies, test_accuracies = [], [], []

    # 添加保存最佳模型权重的变量
    best_test_accuracy = 0.0
    best_test_r2 = -float('inf')
    best_model_state = None
    r2score = R2Score(num_outputs=2, multioutput='raw_values').to(device)
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
        # for _, data in enumerate(train_loader):
        #     X, Y1, Y2 = data
        #     X = X.view(X.shape[0], timestep, features)
        running_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(train_loader):
            X, Y1, Y2 = data
            X = X.view(X.shape[0], timestep, features)

            pred, reg = model(X.to(device))  # 前向传播，模型生成预测

            optimizer.zero_grad()  # 梯度清零，以避免梯度累积

            loss = criterion1(pred, Y1.to(device)) + (1 - alpha) * criterion2(reg, Y2.to(device))  # 计算损失函数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用优化器更新模型参数

            _, predicted = torch.max(pred, 1)
            total += Y1.size(0)
            correct += (predicted == Y1.to(device)).sum().item()

            running_loss += loss.item()

        train_loss = running_loss / total
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_running_loss = 0.0  # 添加这一行来计算测试集损失
            all_predictions = []  # 用于保存所有预测结果
            all_targets = []  # 用于保存所有真实标签
            for i, data in enumerate(test_loader):
                X, Y1, Y2 = data
                X = X.view(X.shape[0], timestep, features)
                pred, reg = model(X.to(device))

                _, predicted = torch.max(pred, 1)
                total += Y1.size(0)
                correct += (predicted == Y1.to(device)).sum().item()
                # 计算测试集损失
                loss = criterion1(pred, Y1.to(device)) + (1 - alpha) * criterion2(reg, Y2.to(device))
                test_running_loss += loss.item()

                #计算MAE和RMSE
                mae = mean_absolute_error(Y2.numpy(), reg.cpu().numpy())
                mse = mean_squared_error(Y2.numpy(), reg.cpu().numpy())
                rmse = np.sqrt(mse)

                # 保存预测结果和真实标签
                all_predictions.append(reg.cpu().numpy())
                all_targets.append(Y2.cpu().numpy())

            test_loss = test_running_loss / total  # 计算平均测试集损失
            test_losses.append(test_loss)  # 将测试集损失保存到列表中
            test_accuracy = correct / total
            test_accuracies.append(test_accuracy)

            # 保存模型的权重，如果测试准确率提高了
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_test_r2 = r2score(reg, Y2.to(device))
                best_model_state = model.state_dict()

        r2score = R2Score(num_outputs=2, multioutput='raw_values').to(device)
        epoch_info = f"Epoch {epoch + 1}/{num_epochs}, "
        spaces = ' ' * len(epoch_info)
        print(f"{spaces} Test Loss: {test_loss},Test Accuracy: {test_accuracy}")
        print( f"{spaces} r2score(reg, Y2): {r2score(reg, Y2.to(device))}")
        # 计算 MAE 和 RMSE 并打印出来
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = mean_squared_error(all_targets, all_predictions, squared=False)
        print(f"{spaces} Test MAE: {mae}, Test RMSE: {rmse}")

    # 打印最佳准确率和最高R方值
    print(f"Best Test Accuracy: {best_test_accuracy}")
    print(f"Best Test R^2: {best_test_r2}")
    # 保存最佳模型权重
    torch.save(best_model_state, 'best_model_weights.pth')

    # Plot the loss and accuracy curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# 在下次运行时，加载最佳权重并在需要时使用它们
load_weights = False  # 是否加载最佳权重
if load_weights:
    model.load_state_dict(torch.load('best_model_weights.pth'))
    model.eval()


