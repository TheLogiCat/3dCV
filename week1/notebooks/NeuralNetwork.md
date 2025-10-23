### 简单神经网络结构
#### 下载数据集
```
from torchvision import datasets
train_data = datasets.FasionMNIST(root, train, download, transforms)
```
- 将指定数据集下载到root所示路径下
- 若train=True，则将其作为训练集
- transforms则对数据集进行指定变换，如ToTensor()

#### 加载数据集
```
from torch.utils.data import DataLoader
train_dataLoader = DataLoader(training_data, batch_size=64, shuffle=True...)
```
DataLoader将数据包装成可迭代的数据流，方便模型训练时批量捕获数据。DataLoader生成一个迭代器，每个迭代返回一个批次的数据。

DataLoader的核心功能如下：
- 批量处理：将数据按设定的batch_size分成多个批次
- 数据混洗：设置shuffle参数打乱数据顺序

```
# 常用方式：
for batch_idx, (images, labels) in enumerate(train_dataloader):
```
#### 构建神经网络结构
```
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # flatten()将二维图片展开为一维数列，便于作为input输入网络
        self.flatten = nn.Flatten()
        # 利用Sequential()快速构建网络结构，使用Linear()实现全连接层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        # logits表示输出层的输出结果，可以看作模型对于输入样本属于哪个类别的预测结果
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```
继承nn.Module，记得定义__init__()函数初始化类
#### 定义损失函数和优化器
使用**交叉熵**损失函数：
```
loss_fn = nn.CrossEntropyLoss()
```
使用**SGD**作为优化器：
```
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
- model.parameters()返回模型所有的可学习参数，包括权重矩阵和偏置项等
- learning_rate控制每次学习的步长

SGD更新公式：
```
param = param - lr * param.grad
```
#### 网络训练过程
```
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 将模型设置为训练模式，激活Dropout和梯度计算等
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        # 通过前向传播得到一个batch_size内的模型预测结果pred
        pred = model(X)
        # 计算pred与y之间的损失函数值
        loss = loss_fn(pred, y)

        # 防止梯度累加，在每次循环开始时清空梯度
        optimizer.zero_grad()
        # 计算损失函数关于各个参数的梯度值
        loss.backward()
        # 根据计算得到的梯度更新参数
        optimizer.step()
        
        # 每经过100个batch_size，即6400次循环过后，输出当前的loss值和训练轮数
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
```
current:>5d的效果：
- 将变量 current格式化为​​十进制整数
- 确保输出至少占用5个字符
- 若数字位数不足5位，则右对齐，左边用空格补足

#### 模型测试过程
test_loop()函数的作用是对训练后的模型进行​​验证/测试评估​​，通过与真实标签对比计算模型的精度和损失。
```
def test_loop(dataloader, model, loss_fn):
    # model.eval()关闭 Dropout/BatchNorm 等训练专用层，确保推理一致性。
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # torch.no_grad()禁用梯度计算，减少内存消耗并加速计算
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # 取预测概率最大的类别判断与真实标签是否相等，将其累加，得到所有预测正确的数量
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy:{(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n')
```
#### 神经网络训练过程
模型在经过train_loop()的训练后，将训练后的模型与真实标签对比，得到一个epoch的精度和Loss。

一个epoch表示迭代一轮所有的batch。
```
for t in range(10):
    print(f"Epoch {t + 1}\n---------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```
