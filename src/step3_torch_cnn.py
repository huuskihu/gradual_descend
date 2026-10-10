"""
step3_torch_cnn.py —— 第 3 步：卷积神经网络（CNN），冲击 99%+
=====================================================================
【为什么全连接网络会撞天花板】
第 2 步的 MLP 把 28x28 拉平成 784 个数，有两个致命缺点：
  1. 丢掉空间结构：像素"左移一格"对它来说是完全不同的输入，
     它不知道笔画是连着的、也不知道数字长什么样。
  2. 参数爆炸：换成 224x224 的照片，第一层就是 150528 x 1000 ≈ 1.5 亿个参数。

【卷积的两大思想】
  * 局部连接 + 权重共享：一个 3x3 的"卷积核"在整张图上滑动，
    同一套权重扫遍全图，它学到的是"局部图案探测器"（比如"这里有一小段竖线"）。
    参数量从 784xH 降到 3x3x通道数，少几个数量级。
  * 平移不变性：数字写在左上角还是右下角，同一个探测器都能发现它。

【本文件的结构（形状变化是理解 CNN 的关键）】
    输入 (N, 784)
    view          -> (N, 1, 28, 28)   1 是通道数（灰度=1，彩色=3）
    Conv(1->32)   -> (N, 32, 28, 28)  32 个"局部图案探测器"
    MaxPool(2)    -> (N, 32, 14, 14)  每 2x2 只留最大值：缩图 + 增强鲁棒性
    Conv(32->64)  -> (N, 64, 14, 14)  更高级的图案（弧线、闭环…）
    MaxPool(2)    -> (N, 64, 7, 7)
    Flatten       -> (N, 3136)
    Linear(3136->128) -> ReLU -> Dropout(0.5) -> Linear(128->10)

【Dropout】训练时随机让一半神经元"请假"，逼网络不依赖任何单个神经元，
像"故意缺席队员来训练整支球队"。推理时全员回归。这是第 2 步过拟合的解药。

运行：python src/step3_torch_cnn.py     # GPU 上约 13 秒，目标 99%+
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn          # nn 里放着各种现成的网络层

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_data import PROJECT_ROOT, make_loaders


class SimpleCNN(nn.Module):
    """小型 LeNet 风格 CNN。60 行以内，MNIST 上就能打到 99%+。"""

    def __init__(self):
        super().__init__()                 # 必须先调父类初始化
        # nn.Sequential：把若干层按顺序串起来，数据依次流过，省去手写 forward 里的每一句
        self.features = nn.Sequential(
            # nn.Conv2d(输入通道, 输出通道, 卷积核边长, padding=1)
            # padding=1 表示四周补一圈 0，好让 28x28 进去还是 28x28 出来
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),                     # ReLU：负数变 0，正数不变（现代默认激活函数）
            nn.MaxPool2d(2),               # 2x2 窗口取最大值：28x28 -> 14x14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),               # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 把 (N,64,7,7) 拉平成 (N, 64*7*7=3136)
            nn.Linear(64 * 7 * 7, 128),    # 全连接层：3136 -> 128
            nn.ReLU(),
            nn.Dropout(0.5),               # 训练时随机丢掉 50% 的神经元，防过拟合
            nn.Linear(128, 10),            # 128 -> 10 个类别得分
        )

    def forward(self, x):
        # x 进来是 (N, 784) 的拉平图，view 把它还原成 (N, 1, 28, 28)
        # -1 是"这一维你自己算"，是常用的懒人写法
        x = x.view(-1, 1, 28, 28)
        return self.classifier(self.features(x))   # 先过卷积部分，再过分类部分


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    """和 step2 完全同款的训练/评估循环。

    【重点】从 MLP 换成 CNN，训练代码一个字都不用改 ——
    这就是"模型"和"训练流程"解耦的好处。"""
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):        # 评估时关掉梯度计算，省显存提速
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train_mode:
                optimizer.zero_grad()               # ① 清梯度
            pred = model(x)                         # ② 前向
            loss = criterion(pred, y)               # ③ 损失
            if train_mode:
                loss.backward()                     # ④ 反向
                optimizer.step()                    # ⑤ 更新
            total_loss += loss.item() * len(x)
            correct += (pred.argmax(1) == y).sum().item()
            total += len(x)
    return total_loss / total, correct / total


def main():
    p = argparse.ArgumentParser(description="PyTorch CNN 训练 MNIST")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)     # 1e-3 = 0.001，Adam 的经典学习率
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    loaders, (mean, std) = make_loaders(batch_size=args.batch)
    model = SimpleCNN().to(device)
    print(f"可训练参数个数: {sum(t.numel() for t in model.parameters()):,}"
          f"   （卷积层只占一小部分，大头在最后的全连接层）")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = PROJECT_ROOT / "models" / "mnist_cnn.pt"
    best_val = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        if va_acc > best_val:                   # 只保存"验证集上最好"的那一版（早停思想）
            best_val = va_acc
            torch.save({"state_dict": model.state_dict(),
                        "standardize": (mean, std), "val_acc": best_val}, ckpt_path)
            star = "  <- 已保存"
        else:
            star = ""
        print(f"  epoch {epoch:3d}/{args.epochs}  train loss={tr_loss:.4f} acc={tr_acc * 100:.2f}%  "
              f"val acc={va_acc * 100:.2f}%{star}")

    # 把刚才存的最佳权重读回来，再考一次试 —— 这才是最终交付的模型
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])        # 把参数填回网络
    _, te_acc = run_epoch(model, loaders["test"], criterion, optimizer, device, False)
    print("-" * 60)
    print(f"最终测试集准确率: {te_acc * 100:.2f}%   (训练全程 {time.time() - t0:.1f}s)")
    print("参考历史: 1998 LeNet ~99% | 2012 AlexNet (更大图) | 本文件 99%+")
    print(f"最佳模型已保存 -> {ckpt_path}")


if __name__ == "__main__":
    main()
