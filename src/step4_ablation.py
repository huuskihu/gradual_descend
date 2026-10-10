"""
step4_ablation.py —— 第 4 步：像科学家一样做对比实验（消融实验 / ablation）
=====================================================================
【为什么要做这个】
"学习率设多少？用 Adam 还是 SGD？ReLU 还是 sigmoid？"
新手常靠感觉猜，专业做法是：控制变量、一次只改一个因素、跑对比。

【本文件回答的问题】（每组只跑 5 轮，约 30 秒出全部结果）
  激活函数 sigmoid（2012 年的默认） vs ReLU（现在的默认），差多少？
  优化器 SGD vs Adam，同样时间谁学得多？

【什么是消融实验】
把系统里某个部件"拿掉"或"换掉"，看性能掉多少，就知道它值多少分。
这是深度学习论文里最常见（也最常被吐槽）的实验表格。

运行：python src/step4_ablation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# 把 src 加入模块搜索路径，才能 import 下面的 mnist_data 和 step2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_data import PROJECT_ROOT, make_loaders
from step2_torch_mlp import MLP, run_epoch      # 直接复用第 2 步写好的模型与训练循环


def train_one(tag: str, hidden: int, activation: str, optimizer_name: str,
              epochs: int, device) -> dict:
    """训练一组配置并返回结果。

    【控制变量的精髓】四组实验共用这一份代码，只换传入的配置参数，
    其他条件（数据、轮数、网络大小、batch）完全相同 —— 这样差异才能归因到被改的那个因素。"""
    loaders, _ = make_loaders(batch_size=64)          # 每次都重新取数据（内存里其实一样）
    model = MLP(hidden, activation).to(device)        # 造模型并搬到指定设备
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    hist_val = []                                     # 记录每轮的验证准确率
    t0 = time.time()
    for _ in range(epochs):
        run_epoch(model, loaders["train"], criterion, opt, device, True)    # 训练一轮
        _, va = run_epoch(model, loaders["val"], criterion, opt, device, False)  # 验证一轮
        hist_val.append(va)                           # append 追加到列表末尾
    _, te = run_epoch(model, loaders["test"], criterion, opt, device, False)
    return {"tag": tag, "val": hist_val, "test": te, "time": time.time() - t0}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    epochs = 5
    results = []

    # 列表里放元组，for 循环一次解包三个值：(显示名, 激活函数, 优化器)
    for tag, act, optname in [
        ("sigmoid + SGD ", "sigmoid", "sgd"),   # 2012 年视频时代的经典配方
        ("sigmoid + Adam", "sigmoid", "adam"),
        ("ReLU    + SGD ", "relu", "sgd"),
        ("ReLU    + Adam", "relu", "adam"),     # 现代默认配方
    ]:
        print(f">>> 训练 {tag.strip()}  ({epochs} 轮)...")   # .strip() 去掉首尾空格
        r = train_one(tag, 128, act, optname, epochs, device)
        results.append(r)
        # 下面这个 '  '.join(...) 把每轮的准确率拼成一行字符串
        print(f"    验证集每轮: {'  '.join(f'{a * 100:.2f}%' for a in r['val'])}")
        print(f"    测试集: {r['test'] * 100:.2f}%   用时 {r['time']:.1f}s\n")

    # ---- 打印对比表 ----
    print("=" * 66)
    print(f"{'配方':16s}{'第1轮验证':>10s}{'第5轮验证':>10s}{'测试集':>10s}")
    # 格式说明：{变量:16s} 表示占 16 个字符宽、左对齐；{:>10s} 表示占 10 宽、右对齐
    print("-" * 66)
    for r in results:
        print(f"{r['tag']:16s}{r['val'][0] * 100:9.2f}%{r['val'][-1] * 100:9.2f}%{r['test'] * 100:9.2f}%")
    print("=" * 66)
    print("""解读指南（重要：按你自己跑出来的数据说话，而不是背结论）：
* 本实验的配方差距其实不大 —— MNIST 太简单，怎么配都能 95%+。
  真正拉开差距的是更深网络、更大数据、更难任务。
* ReLU+SGD 反而垫底？这通常说明学习率(0.1)对 ReLU 偏大，训练在震荡 ——
  优化器和学习率必须配套调，单看"优化器名字"没有意义。
* 流行经验（ReLU > sigmoid、Adam 省心）在更深/更难的任务上会更明显，
  但经验法则只是起点，最终一切以实验为准 —— 这就是机器学习的基本工作方式。
* 动手试试：把 epochs 调到 30、hidden 调到 32 或 256，看结论怎么变。""")


if __name__ == "__main__":
    main()
