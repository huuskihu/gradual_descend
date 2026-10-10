"""
step2_torch_mlp.py —— 第 2 步：用 PyTorch 重写"一模一样"的网络
=====================================================================
【和第 1 步的关系】
同样的 784-30-10 全连接网络、同样的 MNIST 数据。
区别：反向传播不再手写 —— PyTorch 的【自动微分 autograd】帮你算。
我们要看清：框架到底帮你省了哪些事。

【PyTorch 三大件】
  1. Tensor + autograd：记录了所有运算，一句 loss.backward() 自动算出全部梯度
  2. nn.Module：把"参数"和"怎么算输出(forward)"打包成一个对象
  3. Optimizer 优化器：封装了 参数 -= 学习率 × 梯度（SGD），还有更聪明的走法（Adam）

【标准训练循环五连 —— 背下来，所有 PyTorch 项目都长这样】
    optimizer.zero_grad()      # 1 清空上一步梯度（不清会累加！经典 bug）
    pred = model(x)            # 2 前向传播
    loss = criterion(pred, y)  # 3 算损失
    loss.backward()            # 4 反向传播：自动算梯度
    optimizer.step()           # 5 用梯度更新参数

【和第 1 步最大的不同：GPU】
torch.device("cuda") 把张量和模型搬进显卡。MNIST 太小提速不明显，
但网络变大、图片变大时，GPU 是几十倍的区别。

运行：python src/step2_torch_mlp.py
"""

from __future__ import annotations

import argparse                 # 标准库：解析命令行参数
import sys                      # 标准库：这里用来加模块搜索路径
import time                     # 标准库：计时
from pathlib import Path        # 标准库：路径处理

import torch                    # PyTorch 主库（张量、GPU）
import torch.nn as nn           # nn = neural network，放各种网络层和损失函数
import torch.nn.functional as F # F 里是"函数版"的激活函数（relu / sigmoid ...）

# 把 src 目录加入模块搜索路径，才能 import 同目录的 mnist_data / step3
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_data import PROJECT_ROOT, make_loaders


class MLP(nn.Module):
    """全连接网络（Multi-Layer Perceptron）。

    【Python 语法】class MLP(nn.Module) 表示"继承"nn.Module：
    白拿它的一大堆功能（参数管理、.to(device)、.eval()、.parameters() ...），我们只写自己的部分。

    nn.Linear(in, out) 就是第 1 步手写的 "X @ W + b" 的封装，
    里面的 W 和 b 是它自己创建并管理的参数，不用我们手动初始化。"""

    def __init__(self, hidden: int = 30, activation: str = "relu"):
        super().__init__()                  # 必须先调用父类的初始化（注册参数），漏了会报错
        self.fc1 = nn.Linear(784, hidden)   # 第 1 层：784 -> hidden（fc = fully connected 全连接）
        self.fc2 = nn.Linear(hidden, 10)    # 第 2 层：hidden -> 10（10 个数字的得分）
        # 把激活函数存成属性，第 4 步实验要对比 relu 和 sigmoid
        self.act = F.relu if activation == "relu" else torch.sigmoid

    def forward(self, x):
        """定义"数据怎么流过网络"。【注意：没写反向传播】—— autograd 会根据它自动推出反向。"""
        x = self.act(self.fc1(x))           # 先过第 1 层线性变换，再激活
        return self.fc2(x)                  # 输出【原始得分 logits】，故意不接 softmax！
        # 原因：下面的 CrossEntropyLoss 内部会自己做 softmax，重复做会出错。


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    """跑一轮（训练 或 评估）。两种模式共用一个函数，区别只在 train_mode 这个开关。

    model.train() / model.eval() 的区别：影响 Dropout、BatchNorm 这类
    "训练和推理行为不同"的层。忘记切换是新手最常见的隐蔽 bug。"""
    model.train() if train_mode else model.eval()   # 三元表达式：if 真走前面，假走后面

    total_loss, correct, total = 0.0, 0, 0          # 一行给三个变量赋初值

    # torch.set_grad_enabled(train_mode)：评估时不需要梯度，关掉能省显存、提速
    with torch.set_grad_enabled(train_mode):
        for x, y in loader:                          # DataLoader 每次吐出一批 (数据, 标签)
            x, y = x.to(device), y.to(device)        # .to(device) 把数据搬到 GPU（或留在 CPU）

            if train_mode:
                optimizer.zero_grad()                # ① 清空上一步的梯度（不清会累加！）
            pred = model(x)                          # ② 前向传播（自动调用 forward）
            loss = criterion(pred, y)                # ③ 算损失
            if train_mode:
                loss.backward()                      # ④ 反向传播：autograd 自动算所有参数的梯度
                optimizer.step()                     # ⑤ 用梯度更新参数

            # .item() 把只有一个元素的张量变成 Python 普通数字，方便累加
            total_loss += loss.item() * len(x)       # 乘 len(x) 是为了后面能算"平均到每张图"
            # pred.argmax(1)：在 10 个类别里取最大值的下标，即预测的数字
            correct += (pred.argmax(1) == y).sum().item()   # 统计这批对了几个
            total += len(x)

    return total_loss / total, correct / total        # 返回 (平均损失, 正确率)


def main():
    p = argparse.ArgumentParser(description="PyTorch 版 MLP 训练 MNIST")
    p.add_argument("--hidden", type=int, default=128)          # type=int 表示命令行传来的字符串要转成整数
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="不填则用该优化器的默认推荐值")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")   # choices 限定只能选这两个
    p.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    args = p.parse_args()                                       # 解析命令行，结果放进 args

    # torch.cuda.is_available()：显卡能不能用。能用就用 cuda，否则退回 cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    loaders, (mean, std) = make_loaders(batch_size=args.batch)   # 拿到三个 DataLoader
    print(f"网络结构: 784-{args.hidden}-10   激活: {args.activation}   优化器: {args.optimizer}")

    model = MLP(args.hidden, args.activation).to(device)   # .to(device) 把模型参数也搬进 GPU
    # model.parameters() 返回所有可训练参数；t.numel() 数一数这个张量有多少个数
    n_params = sum(t.numel() for t in model.parameters())
    print(f"可训练参数个数: {n_params:,}")                  # :, 是千分位格式

    # CrossEntropyLoss = softmax + 交叉熵一步到位（所以 forward 里不能自己加 softmax）
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        # 动量 momentum：更新时带一点"惯性"，能冲过小坑、收敛更快
        # args.lr or 0.1 的写法：如果 lr 是 None（没传）就用 0.1
        opt = torch.optim.SGD(model.parameters(), lr=args.lr or 0.1, momentum=0.9)
    else:
        # Adam：给每个参数自适应学习率（梯度小的走大步、梯度大的走小步），现代默认起点
        opt = torch.optim.Adam(model.parameters(), lr=args.lr or 1e-3)

    best_val = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, opt, device, True)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, opt, device, False)
        best_val = max(best_val, va_acc)               # max 记住历史最好成绩
        print(f"  epoch {epoch:3d}/{args.epochs}  train loss={tr_loss:.4f} acc={tr_acc * 100:.2f}%  "
              f"val acc={va_acc * 100:.2f}%")

    _, te_acc = run_epoch(model, loaders["test"], criterion, opt, device, False)
    print("-" * 60)
    print(f"最终测试集准确率: {te_acc * 100:.2f}%   最佳验证: {best_val * 100:.2f}%")
    print(f"耗时 {time.time() - t0:.1f}s   （第 1 步 NumPy 版同样结构约 95.7%）")

    # ---- 保存模型 ----
    # state_dict 是"参数名 -> 张量"的字典，官方推荐只存它（不要存整个模型对象）
    ckpt = PROJECT_ROOT / "models" / "step2_mlp.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)     # 确保 models 目录存在
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "standardize": (mean, std), "val_acc": best_val}, ckpt)
    # vars(args) 把 argparse 的命名空间转成普通字典，方便以后查看当时用的什么超参
    print(f"模型已保存 -> {ckpt}")


if __name__ == "__main__":
    main()
