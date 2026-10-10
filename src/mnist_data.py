"""
mnist_data.py —— 数据加载模块（整个项目共用）
=====================================================================
【这个文件干什么】
把那份 2012 年的 mnist.pkl.gz 读出来，整理成现代深度学习爱用的格式：
  * NumPy 数组          -> 给第 1 步"纯手写网络"用
  * PyTorch DataLoader  -> 给后面 PyTorch 版本用

【为什么要单独一个文件】
三个训练脚本都要读数据，写三遍既累又容易错。
专业做法：数据准备(Data) 和 模型训练(Train) 分开，叫"关注点分离"。

【MNIST 是什么】
70000 张 28x28 灰度手写数字图，深度学习界的 Hello World。
本仓库划分：50000 训练 + 10000 验证 + 10000 测试。
每张图被"拉平"成 784(=28*28) 个数，0.0~1.0 表示这个像素有多黑。

【为什么分 train / val / test】
  train 训练集：喂给模型"长脑子"（更新参数）
  val   验证集：训练中定期考试，用来调超参（学习率、网络大小…）
  test  测试集：最后只考一次，汇报真实水平
  拿 test 调参 = 看着答案做题，分数是假的，这叫【数据泄露】。
"""

# from __future__ import annotations：让类型标注写法更宽松（Python 3.10+ 其实可以不要）
from __future__ import annotations

import gzip                      # 标准库：读写 .gz 压缩文件
import pickle                    # 标准库：Python 的"对象存档/读档"格式
from pathlib import Path         # 标准库：处理文件路径

import numpy as np               # 第三方库：数值计算
import torch                     # 第三方库：深度学习框架
from torch.utils.data import DataLoader, TensorDataset   # 从子模块里借两个工具

# Path(__file__)      = 当前这个 .py 文件的路径
# .resolve()          = 转成绝对路径
# .parents[0]         = 所在目录(src)，.parents[1] = 上一级目录(项目根)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 用 / 拼接路径（Windows/Linux 通用），不用手写反斜杠
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "mnist.pkl.gz"


def load_raw(data_file: str | Path = DEFAULT_DATA_FILE):
    """读取 gz 压缩包里的 pickle 数据，返回 (训练, 验证, 测试) 三份原始数据。

    【Python 语法】def 函数名(参数: 类型 = 默认值) -> 返回值类型:
    参数有默认值，调用时不传就用它。"""
    data_file = Path(data_file)      # 传入的可能是字符串，统一转成 Path 对象

    if not data_file.exists():       # .exists() 判断文件是否存在，返回 True/False
        # raise 主动抛出异常，程序会停在这里并打印这句话（比莫名其妙报错友好）
        raise FileNotFoundError(
            f"找不到数据文件：{data_file}\n"
            f"请确认 data/mnist.pkl.gz 存在（它从原仓库 data/ 目录复制而来）"
        )

    # with ... as f: 是"用完自动关闭文件"的标准写法，不用手动 f.close()
    # "rb" = read binary（以二进制只读方式打开），因为 pickle 是二进制格式
    with gzip.open(data_file, "rb") as f:
        unpickler = pickle._Unpickler(f)   # 造一个"读档器"
        unpickler.encoding = "latin1"      # 老存档是 Python 2 生成的，编码不同，必须改成 latin1
                                           # 不加这行会报 UnicodeDecodeError（老代码在新环境的经典坑）
        training_data, validation_data, test_data = unpickler.load()   # 一次读出三个对象（元组解包）
    return training_data, validation_data, test_data


def load_arrays(data_file: str | Path = DEFAULT_DATA_FILE) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """整理成 NumPy 数组，返回字典：
        {"train": (X, y), "val": (X, y), "test": (X, y)}
    X 形状 (样本数, 784)，float32，0.0~1.0
    y 形状 (样本数,)，int64，取值 0~9（这就是人工标注的正确答案）

    【为什么转 float32】GPU 和神经网络默认用 32 位浮点；float64 精度过剩、
    还慢一倍、占两倍显存。"""
    tr, va, te = load_raw(data_file)     # 读原始数据，三份一次接住

    def unpack(pair):
        """函数里再定义函数（内部函数），避免重复写两遍转换代码。"""
        x, y = pair                      # pair 是 (图片, 标签) 二元组，拆开
        # np.asarray(数据, dtype=类型)：转成 NumPy 数组并指定类型
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)

    # 返回字典：用名字取数据，比按位置取（tr[0]）清晰得多
    return {
        "train": unpack(tr),
        "val": unpack(va),
        "test": unpack(te),
    }


def make_one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """把标签 3 变成 [0,0,0,1,0,0,0,0,0,0]，叫【独热编码 one-hot】。

    为什么需要？网络最后输出 10 个数，答案也得写成 10 个数的形式才能比较。"""
    out = np.zeros((len(y), num_classes), dtype=np.float32)   # 先造一个全 0 的 (N,10) 矩阵
    # np.arange(len(y)) = [0,1,2,...,N-1]，和 y 一起做"花式索引"
    # 意思：第 i 行的第 y[i] 列 设为 1
    out[np.arange(len(y)), y] = 1.0
    return out


def make_loaders(batch_size: int = 64,
                 standardize: bool = True,
                 data_file: str | Path = DEFAULT_DATA_FILE,
                 num_workers: int = 0):
    """生成 PyTorch 的 DataLoader（自动发牌员：打乱、分批、喂给模型）。

    【什么是 standardize 标准化】把像素从 0~1 变成"均值 0、标准差 1"左右：x' = (x-mean)/std
    好处：输入尺度统一，梯度不会忽大忽小，训练更快更稳。
    注意！mean/std 只能用【训练集】算，再原样套到验证/测试集 —— 用测试集算也是一种数据泄露。

    【batch_size】一次给模型看多少张图。太小→梯度抖；太大→显存爆。64 是安全起点。"""
    arrays = load_arrays(data_file)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = arrays["train"], arrays["val"], arrays["test"]

    if standardize:
        mean = float(Xtr.mean())          # 只用训练集算！Xtr.mean() 是全部元素的平均值
        std = float(Xtr.std())            # 标准差
        Xtr = (Xtr - mean) / std          # 广播：整个数组一起减、一起除
        Xva = (Xva - mean) / std
        Xte = (Xte - mean) / std
    else:
        mean, std = 0.0, 1.0

    def to_dataset(X, y):
        # torch.from_numpy：NumPy 数组 -> PyTorch 张量（PyTorch 只认自己的 Tensor）
        # TensorDataset：把"数据"和"标签"打包成一个数据集对象
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    # DataLoader 的参数：
    #   shuffle=True   训练集要打乱，否则模型会记住顺序
    #   shuffle=False  验证/测试不用打乱，保证结果可复现
    #   num_workers    用几个子进程加载数据；Windows 上设 0 最稳
    loaders = {
        "train": DataLoader(to_dataset(Xtr, ytr), batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, drop_last=False),
        "val": DataLoader(to_dataset(Xva, yva), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers),
        "test": DataLoader(to_dataset(Xte, yte), batch_size=batch_size,
                           shuffle=False, num_workers=num_workers),
    }
    return loaders, (mean, std)           # 返回两个东西：三个 loader，以及标准化参数（推理时要用到）


def describe():
    """打印数据体检报告 —— 所有机器学习项目的第一步都是：先"看"数据，别急着建模。"""
    arrays = load_arrays()
    print("=" * 62)
    print("MNIST 数据体检报告")
    print("=" * 62)
    # .items() 遍历字典，每次给 (键, 值)；这里值本身又是 (X, y) 元组
    for name, (X, y) in arrays.items():
        print(f"[{name:5s}] 图片张数={len(X):6d}  形状={str(X.shape):12s} "
              f"标签形状={str(y.shape):10s} 像素范围=[{X.min():.3f}, {X.max():.3f}]")
    Xtr, ytr = arrays["train"]
    print("-" * 62)
    print(f"前 10 张训练图的正确答案（人工标注）: {ytr[:10]}")     # 切片取前 10 个
    print(f"每个数字出现的次数: {np.bincount(ytr, minlength=10)}")  # bincount 统计每个整数出现几次
    print(f"单张图占用内存: {Xtr[0].nbytes} 字节 = 28*28 个 float32")
    print("=" * 62)


def show_samples(n: int = 10, save_to: str | Path | None = None):
    """把前 n 张图画出来看看。"""
    import matplotlib
    matplotlib.use("Agg")                      # Agg = 不弹窗口，直接存文件
    import matplotlib.pyplot as plt

    (Xtr, ytr), _, _ = load_arrays()["train"], load_arrays()["val"], load_arrays()["test"]
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.0))   # 1 行 n 列的子图
    for i, ax in enumerate(axes):              # enumerate 同时给出下标 i 和元素 ax
        ax.imshow(Xtr[i].reshape(28, 28), cmap="gray")        # 784 -> 28x28 才能显示；gray = 灰度
        ax.set_title(f"label={ytr[i]}")
        ax.axis("off")                          # 不显示坐标轴
    plt.tight_layout()
    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)     # 目录不存在就创建（exist_ok 避免已存在时报错）
        plt.savefig(save_to, dpi=120)
        print(f"已保存图片 -> {save_to}")
    else:
        plt.show()


# 【Python 语法】下面这句读作："如果本文件被直接运行（而不是被 import），就执行下面两行"
if __name__ == "__main__":
    describe()
    show_samples(10, save_to=PROJECT_ROOT / "outputs" / "mnist_samples.png")
