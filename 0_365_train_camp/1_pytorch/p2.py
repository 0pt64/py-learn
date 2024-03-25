import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch.nn.functional as F
from torchinfo import summary

# 1 设置gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2 获取训练数据集和测试数据集
train_ds = torchvision.datasets.CIFAR10("data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_ds = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 3 数据分批
batch_size = 32
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

imgs, labels = next(iter(train_dl))
print(imgs.shape)

# 4 数据可视化
plt.figure(figsize=(20, 5))
for i, img in enumerate(imgs[:20]):
    # 维度缩减
    npimg = img.numpy().transpose((1, 2, 0))
    # 把figure分成2行10列，绘制第i+1个子图
    plt.subplot(2, 10, i + 1)
    plt.imshow(npimg, cmap=plt.cm.binary)
    plt.axis('off')

plt.show()

# 5 构建cnn

num_classes = 10  # 图片的类别数


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取网络
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # 分类网络
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    # 前向传播
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 将模型转移到GPU中
model = Model().to(device)

summary(model=model, input_size=(32, 3, 32, 32))

# 设置超参数
loss_fn = nn.CrossEntropyLoss()  # 创建损失函数
learn_rate = 1e-2  # 学习率
opt = torch.optim.SGD(model.parameters(), lr=learn_rate)


# 训练循环
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练集大小，60000张图片
    num_batches = len(dataloader)  # 批次数目，60000/32

    train_loss, train_acc = 0, 0

    for X, y in dataloader:  # 获取图片与标签
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)  # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出和真实值之间的差距，targets为真实值，两者差即为损失

        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()  # 反向传播
        optimizer.step()  # 每一步自动更新

        # 记录acc与loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

    train_acc /= size
    train_loss /= num_batches

    return train_acc, train_loss


# 测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_acc = 0, 0

    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)

            # 计算loss
            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)

            test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()
            test_loss += loss.item()

    test_acc /= size
    test_loss /= num_batches

    return test_acc, test_loss


# 训练
epochs = 10
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    model.train()
    epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_fn, opt)

    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%，Test_loss:{:.3f}')
    print(template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss))

print("111")
