"""
step1_numpy_nn.py —— 第 1 步：不借助任何框架，纯 NumPy 手写一个神经网络
=====================================================================
【这个文件干什么】
用最原始的矩阵运算，从零实现"训练一个能认手写数字的网络"。
**下面每一行代码都有注释**，包括 Python 语法本身（import、def、self 是什么都讲）。
零基础建议从上往下一行一行读。

【它对应的数学】（看不懂没关系，代码后面会一句句讲）
    前向：  Z1 = X·W1 + b1  ->  A1 = sigmoid(Z1)  ->  Z2 = A1·W2 + b2  ->  A2 = softmax(Z2)
    损失：  L = -平均( log( A2 在正确类别上的概率 ) )
    反向：  从 L 出发，用链式法则一层层往回算每个权重的"责任"（梯度）
    更新：  W = W - 学习率 × 梯度

【运行】
    python src/step1_numpy_nn.py                    # 默认 30 轮，约 8 秒
    python src/step1_numpy_nn.py --epochs 5 --lr 0.5
"""

# ----------------------------------------------------------------------
# import 区域：借用别人写好的代码
# ----------------------------------------------------------------------
import argparse                  # 标准库：支持命令行参数（--epochs 这种）
import sys                       # 标准库：跟解释器打交道，这里用来加模块搜索路径
import time                      # 标准库：计时
from pathlib import Path         # 标准库：处理文件路径

import numpy as np               # 第三方库：数值计算。as np 是起小名，大家都这么写

# sys.path 是"Python 去哪些文件夹里找模块"的列表。
# 把本文件所在目录(src)加进去，下一行才能 import 同目录的 mnist_data.py
# （不加这句，直接 python src/step1_numpy_nn.py 会报 ModuleNotFoundError）
sys.path.insert(0, str(Path(__file__).resolve().parent))

# from xxx import yyy：只借用某个模块里指定的工具
from mnist_data import load_arrays, make_one_hot, PROJECT_ROOT


# ======================================================================
# 1. 激活函数
# ======================================================================
def sigmoid(z):
    """sigmoid(z) = 1 / (1 + e^-z)，把任意实数压到 (0,1) 区间。

    直觉：像"神经元的开关"，输入越大越接近 1（激活），越小越接近 0（抑制）。

    【Python 语法】def 名字(参数): 定义函数；缩进里的代码属于它；return 把结果交回调用者。
    下面三引号里的内容叫 docstring（说明文档），是给人看的。

    【数值稳定性】z 很负时 e^-z 会溢出成 inf。np.clip 先把 z 截断到 [-500, 500]，
    sigmoid(±500) 已经和 0/1 没区别，不影响结果 —— 写数值代码要养成这个习惯。"""
    z = np.clip(z, -500.0, 500.0)      # clip = 把超出范围的值夹住
    return 1.0 / (1.0 + np.exp(-z))    # np.exp = e 的次方


def softmax(z):
    """把 10 个"得分"变成 10 个"概率"（都非负、加起来等于 1）。

    减最大值是数值稳定技巧，防止 exp(很大的数) 溢出，数学上结果不变。"""
    z = z - z.max(axis=1, keepdims=True)   # axis=1 = "在每一行内部"求最大值
    e = np.exp(z)                          # keepdims=True 保持维度，方便相减时广播
    return e / e.sum(axis=1, keepdims=True)  # 每行各自除以该行的总和


# ======================================================================
# 2. 网络本体
# 【Python 语法】class 是"图纸"，net = Network(...) 才是照图纸造出的"房子"（对象）
# ======================================================================
class Network:
    """一个可训练的多层感知机（MLP）。

    W1/b1/W2/b2 是"参数"，训练的全部目的就是找到这四个数组的最佳取值。"""

    def __init__(self, sizes, seed=42):
        """造网络时自动执行。sizes 例如 [784, 30, 10]。
        【Python 语法】self 代表"这个对象自己"；self.xxx = ... 就是把数据挂在自己身上。"""
        self.sizes = sizes                            # 记下来，以后 self.sizes 能取用
        rng = np.random.default_rng(seed)             # 造随机数发生器；seed 固定 => 结果可复现

        # Xavier 初始化：让初始权重的标准差刚好合适。
        # 全初始化成 0 会让所有神经元学得一模一样（对称性问题）；随机数太大又会梯度饱和。
        # rng.normal(均值, 标准差, size=形状) 生成正态分布的随机数
        self.W1 = rng.normal(0, np.sqrt(1.0 / sizes[0]), size=(sizes[0], sizes[1])).astype(np.float32)
        self.b1 = np.zeros(sizes[1], dtype=np.float32)   # np.zeros = 全 0 数组；偏置一般初始化为 0
        self.W2 = rng.normal(0, np.sqrt(1.0 / sizes[1]), size=(sizes[1], sizes[2])).astype(np.float32)
        self.b2 = np.zeros(sizes[2], dtype=np.float32)

        # 字典里放两个空列表，训练时每轮往里追加数字（append），最后用来画曲线
        self.history = {"loss": [], "val_acc": []}

    # ------------------------------------------------------------------
    def forward(self, X):
        """前向传播：输入 -> 预测。X 形状是 (批量, 784)。

        返回两个东西：最后的概率 A2，以及反向传播要用的中间结果 (X, A1)。
        【Python 语法】return a, b 返回多个值，调用时用 p, (x, a) = func() 接住（元组解包）。"""
        Z1 = X @ self.W1 + self.b1     # @ 是矩阵乘法；(N,784)@(784,30) -> (N,30)
                                       # + self.b1 是"广播"：b1 只有 (30,)，自动复制 N 份再相加
        A1 = sigmoid(Z1)               # 激活：把数值压到 (0,1)
        Z2 = A1 @ self.W2 + self.b2    # (N,30)@(30,10) -> (N,10)
        A2 = softmax(Z2)               # 每一行变成 10 个概率，加起来 = 1
        return A2, (X, A1)             # 把中间结果存下来，反向传播要用

    # ------------------------------------------------------------------
    def predict(self, X):
        """预测类别：取概率最大的那个下标。argmax = "最大值在第几个位置"。"""
        A2, _ = self.forward(X)        # _ 是惯用写法，表示"这个返回值我不用"
        return A2.argmax(axis=1)       # axis=1 = 在每一行内部找最大值的位置

    # ------------------------------------------------------------------
    def train_step(self, X, Y, lr):
        """【整个文件的心脏】看一批图 -> 算损失 -> 反推梯度 -> 更新参数。

        参数：X = 一小批图片 (N,784)；Y = 对应的正确答案 one-hot (N,10)；lr = 学习率
        返回：这一批的平均损失（一个数字）"""
        m = len(X)                               # len() 是内置函数，求这批有几张图

        # ---- 前向传播 ----
        A2, (X_in, A1) = self.forward(X)         # 拿到预测概率和中间结果

        # ---- 算损失（交叉熵）----
        eps = 1e-12                              # 1e-12 = 0.000000000001，防止 log(0) 变负无穷
        loss = -np.sum(Y * np.log(A2 + eps)) / m
        # 解释：Y 是 one-hot，只有正确答案那一列是 1，所以 Y * log(A2) 把其他列清零，
        #       只留下"网络给正确答案的概率"的对数；取负号；除以 m 取平均。

        # ---- 反向传播：从损失出发，用链式法则逐层往回算梯度 ----
        dZ2 = (A2 - Y) / m                       # softmax + 交叉熵求导后恰好是「预测 - 答案」
        dW2 = A1.T @ dZ2                         # A1.T 是转置；(30,N)@(N,10) -> (30,10)，和 W2 同形状
        db2 = dZ2.sum(axis=0)                    # 偏置的梯度 = 把这一批的误差加起来（跨样本压扁）

        dA1 = dZ2 @ self.W2.T                    # 把"责任"传回上一层，所以用 W2 的转置
        dZ1 = dA1 * A1 * (1.0 - A1)              # * 是逐元素相乘（不是矩阵乘！）；乘上 sigmoid 的导数 a(1-a)
        dW1 = X_in.T @ dZ1                       # (784,N)@(N,30) -> (784,30)，和 W1 同形状
        db1 = dZ1.sum(axis=0)

        # ---- 梯度下降：参数 = 参数 - 学习率 × 梯度 ----
        # a -= b 是 a = a - b 的简写。梯度指向"损失上升最快"的方向，所以我们往反方向走。
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return float(loss)                       # float() 把 NumPy 的单数字转成 Python 普通小数

    # ------------------------------------------------------------------
    def evaluate(self, X, y):
        """在没见过的数据上算正确率：预测对的个数 ÷ 总数。"""
        return float((self.predict(X) == y).mean())   # == 得到 True/False 数组；.mean() 即正确率


# ======================================================================
# 3. 训练主循环
# ======================================================================
def train(args):
    """args 是命令行参数打包好的对象，用 args.epochs、args.lr 这样取用。"""
    arrays = load_arrays()                                   # 读数据，返回字典
    (Xtr, ytr), (Xva, yva), (Xte, yte) = arrays["train"], arrays["val"], arrays["test"]
    Ytr = make_one_hot(ytr)                                  # 训练需要 one-hot 形式的答案

    print(f"网络结构: 784-{args.hidden}-10   训练集 {len(Xtr)} 张   "
          f"学习率 {args.lr}   batch {args.batch}   轮数 {args.epochs}")

    net = Network([784, args.hidden, 10], seed=args.seed)    # 造网络
    rng = np.random.default_rng(args.seed)
    t0 = time.time()                                         # 记下开始时刻，最后算耗时

    n = len(Xtr)
    for epoch in range(1, args.epochs + 1):                  # range(1,31) 产生 1~30，所以要 +1
        # ---- 一个 epoch = 把全部训练数据完整过一遍 ----
        perm = rng.permutation(n)                            # 把 0~n-1 随机打乱顺序
        epoch_loss, n_batches = 0.0, 0                       # 一行给两个变量赋值

        for start in range(0, n, args.batch):                # range(0,n,10) = 0,10,20,... 每次跳 batch 格
            idx = perm[start:start + args.batch]             # 这一批用哪些样本的下标
            batch_loss = net.train_step(Xtr[idx], Ytr[idx], args.lr)   # Xtr[idx] 花式索引：一次取多行
            epoch_loss += batch_loss                         # += 累加
            n_batches += 1

        val_acc = net.evaluate(Xva, yva)                     # 每轮结束，在验证集上考一次
        net.history["loss"].append(epoch_loss / n_batches)   # append 往列表末尾追加
        net.history["val_acc"].append(val_acc)
        print(f"  epoch {epoch:3d}/{args.epochs}  平均损失={epoch_loss / n_batches:.4f}  "
              f"验证集准确率={val_acc * 100:.2f}%")

    test_acc = net.evaluate(Xte, yte)                        # 全部训练完，才在测试集上考一次
    print("-" * 60)
    print(f"最终测试集准确率: {test_acc * 100:.2f}%   (10000 张图里认对 {int(test_acc * 10000)} 张)")
    print(f"耗时 {time.time() - t0:.1f}s")

    np.save(PROJECT_ROOT / "outputs" / "step1_history.npy", net.history, allow_pickle=True)
    return net


def plot_history(net):
    """把训练过程画成两条曲线：损失要下降、准确率要上升。"""
    import matplotlib                     # 函数内部 import：只有画图时才加载，省启动时间
    matplotlib.use("Agg")                 # Agg = 不弹窗口，直接存成文件
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))    # 1 行 2 列的子图
    axes[0].plot(net.history["loss"], color="#2563eb")
    axes[0].set_title("Training loss (should go down)")
    axes[0].set_xlabel("epoch")
    axes[1].plot(net.history["val_acc"], color="#16a34a")
    axes[1].set_title("Validation accuracy (should go up)")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)
    out = PROJECT_ROOT / "outputs" / "step1_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)          # 目录不存在就创建
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"训练曲线已保存 -> {out}")


def main():
    """程序入口：解析命令行参数 -> 训练 -> 画图。"""
    p = argparse.ArgumentParser(description="纯 NumPy 手写神经网络训练 MNIST")
    p.add_argument("--hidden", type=int, default=30, help="隐藏层神经元个数")    # default = 不传时的默认值
    p.add_argument("--epochs", type=int, default=30, help="训练多少轮")
    p.add_argument("--batch", type=int, default=10, help="小批量大小")
    p.add_argument("--lr", type=float, default=1.0, help="学习率")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--no-plot", action="store_true", help="不画图")   # store_true：命令行写了就是 True
    args = p.parse_args()                                    # 把命令行参数收集进 args

    net = train(args)
    if not args.no_plot:
        plot_history(net)


# 【Python 语法】这句读作："如果本文件是被直接运行的（而不是被别人 import 的），就执行 main()"
# 这样既能 python src/step1_numpy_nn.py 直接跑，也能被别的脚本 import 而不触发训练。
if __name__ == "__main__":
    main()
