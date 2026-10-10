# Python 基础 + 代码逐行讲解（零基础版）

> 目标：让你不仅能**运行**这个项目，还能**读懂每一行**。
> 读法：先看第一部分（语法），第二部分（别人写的库），第三部分（NumPy），
> 最后第四部分是**本项目真实代码的逐行拆解**。

---

## 第 0 部分 · 怎么读代码：三条基本规则

**规则 1：代码从上往下、一行一行执行。**

**规则 2：缩进决定"归谁管"。** 冒号 `:` 后面的**缩进块**属于上一行：
```python
for i in range(3):      # ← 冒号
    print(i)            # ← 缩进 4 格，属于 for，会执行 3 次
print("结束")            # ← 没缩进，不属于 for，只执行 1 次
```

**规则 3：`#` 后面是注释**，给人看的，计算机直接跳过。本项目代码里大量中文注释就是给你看的。

---

## 第 1 部分 · 本项目用到的 Python 语法

### 1.1 变量与常见类型

| 类型 | 例子 | 说明 |
|---|---|---|
| `int` 整数 | `784` | |
| `float` 小数 | `0.1304` | |
| `str` 字符串 | `"data/mnist.pkl.gz"` | 引号包起来 |
| `bool` 真假 | `True` / `False` | **首字母大写** |
| `list` 列表 | `[784, 30, 10]` | 方括号，可以改 |
| `tuple` 元组 | `(50000, 784)` | 圆括号，**不能改** |
| `dict` 字典 | `{"train": X, "val": Y}` | 键 → 值 |
| `ndarray` 数组 | NumPy 的多维表格 | 见第三部分 |

**列表/数组的"切片"**（取一部分）：
```python
a = [10, 20, 30, 40, 50]
a[0]      # 10   ← 从 0 开始数！这是新手最常犯的错
a[1:3]    # [20, 30]  ← 取第 1 个到第 3 个【之前】，不含第 3 个
a[-1]     # 50   ← 负数表示从末尾数
```

### 1.2 f-string：往字符串里塞变量

字符串前面加 `f`，里面用 `{}` 包变量：
```python
acc = 95.66
print(f"准确率是 {acc}%")              # 准确率是 95.66%
print(f"准确率是 {acc:.2f}%")          # 准确率是 95.66%   ← .2f = 保留 2 位小数
print(f"{'对' if acc > 90 else '错'}")  # 对
```

### 1.3 函数（function）：把一段代码打包复用

```python
def sigmoid(z):          # def 定义函数，z 是"参数"
    return 1 / (1 + np.exp(-z))   # return 把结果交出去

y = sigmoid(0.5)         # 使用（也叫"调用"）
```
- **参数** = 你传给函数的数据
- **返回值** = 函数算完交回来的结果
- 有默认值的参数可以不传：`def load_arrays(data_file=DEFAULT_DATA_FILE)` → 不写就用默认值

### 1.4 方法（method）：属于某个东西的函数

```python
len(X)          # 函数：独立存在，把 X 交给它
X.sum()         # 方法：属于 X 自己，X 在前，用点号调用
X.reshape(28,28)  # 方法
```
**口诀：东西在前用点号的是方法，函数名在前括号包东西的是函数。** 两者本质一样。
不知道某个东西有哪些方法？`print(dir(X))` 能全列出来。

### 1.5 类（class）：图纸与房子

```python
class Network:              # class 是"图纸"
    def __init__(self, sizes):   # 造房子时自动执行（构造函数）
        self.sizes = sizes       # self. 开头 = 挂在这个对象身上，以后能用
        self.W1 = ...

    def forward(self, X):        # 类里的函数叫"方法"，第一个参数必须是 self
        return X @ self.W1       # 用 self.W1 访问自己的数据

net = Network([784, 30, 10])     # 照图纸造一个"房子"（实例/对象）
net.forward(X)                   # 让这个房子干活
```
**`self` 就是"我自己"**：`self.W1` 读起来就是"我自己的 W1"。调用时不用管 self，Python 自动传。

### 1.6 循环与简写

```python
# 普通写法
result = []
for i in range(10):          # range(10) = 0,1,2,...,9
    result.append(i * 2)

# 列表推导（同样的事，一行）
result = [i * 2 for i in range(10)]

# 带下标的循环
for i, name in enumerate(["a", "b"]):
    print(i, name)           # 0 a / 1 b
```

### 1.7 `import`：借用别人写好的代码

```python
import numpy as np                  # 借整个库，起个小名 np
from pathlib import Path            # 只借其中一个工具
from mnist_data import load_arrays  # 借【自己项目里】的模块
import torch.nn as nn
```
库（别人写好的代码包）分三类：
- **标准库**：Python 自带，如 `os` `sys` `time` `pickle` `gzip` `argparse` `pathlib`
- **第三方库**：要 `pip install`，如 `numpy` `torch` `matplotlib` `PIL`
- **自己的模块**：项目里的 `src/mnist_data.py` 等

### 1.8 `if __name__ == "__main__":` 是什么

```python
def main():
    ...

if __name__ == "__main__":
    main()
```
读成：**"如果这个文件是被直接运行的（而不是被别人 import 的），就执行 main()"**。
这样既能 `python step1.py` 直接跑，又能 `import step1` 借用它的函数而不触发训练。

---

## 第 2 部分 · 本项目用到的标准库（逐个讲）

### `pathlib` —— 处理文件路径（推荐写法）

```python
from pathlib import Path
p = Path(__file__).resolve()     # __file__ = 当前这个 .py 文件的路径
                                 # .resolve() = 转成绝对路径
root = p.parents[1]              # .parents[0]=所在目录, [1]=上一级目录
data = root / "data" / "mnist.pkl.gz"   # 用 / 拼路径（不用管 Windows 的反斜杠）
data.exists()                    # 这个文件存在吗？ True/False
data.stat().st_size              # 文件大小（字节）
```
**为什么不用字符串拼路径？** 因为 Windows 用 `\`、Linux 用 `/`，用 `Path` 两边都能跑。

### `sys` —— 跟 Python 解释器本身打交道

```python
import sys
sys.executable        # 当前用的是哪个 python.exe（你昨天踩坑时打印的就是它）
sys.path              # Python 去哪些目录里找模块（一个列表）
sys.path.insert(0, "src")   # 把 src 目录加进去，这样 import mnist_data 才找得到
sys.argv              # 命令行参数，如 ["step1.py", "--epochs", "30"]
```
**`sys.path.insert` 是新手最常遇到的"找不到模块"的解药**：模块不在搜索目录里，就加进去。

### `os` —— 操作系统相关

```python
import os
os.chdir(ROOT)        # 切换"当前工作目录"（相对路径都基于它）
os.getcwd()           # 看看当前在哪个目录
os.environ["PATH"]    # 读环境变量
```

### `gzip` + `pickle` —— 读那个 17MB 的老数据文件

```python
import gzip, pickle
with gzip.open("data/mnist.pkl.gz", "rb") as f:   # "rb" = 以二进制只读方式打开
    u = pickle._Unpickler(f)      # pickle = Python 的"对象存档/读档"格式
    u.encoding = "latin1"         # 这个存档是 Python 2 时代做的，编码不同，必须改
    data = u.load()               # 读档
```
**`with ... as f:`** 是"用完自动关闭文件"的标准写法，不用手动 `f.close()`。
**为什么用 `pickle._Unpickler`？** 因为老存档是 Python 2 生成的，直接用 `pickle.load` 会报编码错误。

### `argparse` —— 让脚本支持命令行参数

```python
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30, help="训练多少轮")
args = p.parse_args()
print(args.epochs)      # 命令行不写就是 30
```
所以 `python src/step1_numpy_nn.py --epochs 10 --lr 0.5` 能改参数而不用改代码。

### `time` —— 计时

```python
import time
t0 = time.time()          # 记下当前时刻（秒）
...                        # 干活
print(f"耗时 {time.time() - t0:.1f}s")
```

### `random` / NumPy 随机与 `seed`

```python
rng = np.random.default_rng(42)   # 造一个"随机数发生器"，42 是种子
rng.normal(0, 1, size=(3, 4))     # 生成 3x4 的正态分布随机数
rng.permutation(50000)            # 把 0~49999 随机打乱顺序
```
**种子（seed）的意义**：同样的种子 = 同样的随机结果 = **别人能复现你的实验**。
深度学习里到处要固定种子，就是为了让结果可复现。

---

## 第 3 部分 · NumPy 速查（本项目的数学工具）

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape        # (2, 3)  两行三列
a.dtype        # 数据类型，如 float32 / int64
a.sum()        # 全部求和 = 21
a.sum(axis=0)  # 沿"行"的方向压扁 -> [5, 7, 9]  （上下相加）
a.sum(axis=1)  # 沿"列"的方向压扁 -> [6, 15]    （左右相加）
```
**`axis` 记忆法**：axis=0 是"跨行"（从上往下），axis=1 是"跨列"（从左往右）。

| 写法 | 含义 |
|---|---|
| `A @ B` | 矩阵乘法 |
| `A * B` | 对应位置各自相乘（**形状必须完全一样**） |
| `A.T` | 转置（横竖翻转） |
| `A.reshape(28,28)` | 换形状，元素个数不变 |
| `np.zeros((2,3))` | 全 0 数组 |
| `np.eye(10)` | 10×10 单位矩阵（用来造 one-hot） |
| `np.exp(x)` / `np.log(x)` | e^x / ln x |
| `np.clip(x, -500, 500)` | 把超范围的值截断（防溢出） |
| `a.argmax()` | 最大值**在哪个位置** |
| `np.allclose(a, b)` | 两个数组是否几乎相等（比较浮点数用） |

**广播（broadcasting）**——为什么 `(10, 30)` 能和 `(30,)` 相加？
```python
Z1 = X @ W1 + b1     # X@W1 是 (10,30)，b1 是 (30,)
```
NumPy 会自动把 `b1` **复制 10 份**变成 (10,30) 再相加。这就是广播，省去手写循环。

---

## 第 4 部分 · 本项目真实代码逐行拆解

### 4.1 `src/mnist_data.py` 里的 `load_arrays()`

```python
def load_arrays(data_file=DEFAULT_DATA_FILE):        # ①
    tr, va, te = load_raw(data_file)                 # ②

    def unpack(pair):                                # ③
        x, y = pair                                  # ④
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)   # ⑤

    return {                                         # ⑥
        "train": unpack(tr),
        "val":   unpack(va),
        "test":  unpack(te),
    }
```

| 行 | 逐行解释 |
|---|---|
| ① | 定义函数。`data_file=DEFAULT_DATA_FILE` 表示"不传就用默认路径" |
| ② | 调用另一个函数读原始文件，返回 3 份数据，用 `a, b, c = ...` 一次接住（叫**元组解包**） |
| ③ | 在**函数内部再定义一个小函数**，只在内部用（避免重复写代码） |
| ④ | `pair` 是 `(图片, 标签)` 这样的二元组，`x, y = pair` 拆开 |
| ⑤ | `np.asarray(..., dtype=...)` 转成 NumPy 数组并**指定类型**：图片用 float32（省显存），标签用 int64（整数下标） |
| ⑥ | 返回一个**字典**，用名字取数据：`arrays["train"]` |

### 4.2 `src/step1_numpy_nn.py` 的 `__init__`（造网络）

```python
def __init__(self, sizes, seed=42):                                       # ①
    self.sizes = sizes                                                    # ②
    rng = np.random.default_rng(seed)                                     # ③
    self.W1 = rng.normal(0, np.sqrt(1.0/sizes[0]), size=(sizes[0], sizes[1])).astype(np.float32)  # ④
    self.b1 = np.zeros(sizes[1], dtype=np.float32)                        # ⑤
    self.W2 = rng.normal(0, np.sqrt(1.0/sizes[1]), size=(sizes[1], sizes[2])).astype(np.float32)
    self.b2 = np.zeros(sizes[2], dtype=np.float32)
    self.history = {"loss": [], "val_acc": []}                            # ⑥
```

| 行 | 逐行解释 |
|---|---|
| ① | 造网络时自动调用。`sizes = [784, 30, 10]` |
| ② | 把 sizes 记住（挂到自己身上，以后 `self.sizes` 能用） |
| ③ | 造随机数发生器，种子固定 → 每次初始化结果一样 |
| ④ | 生成权重矩阵：均值 0、标准差 `sqrt(1/上一层的大小)`、形状 `(784, 30)`。这叫 **Xavier 初始化**——标准差选得刚好，太大太小都学不动。`.astype(np.float32)` 转成 32 位浮点省内存 |
| ⑤ | 偏置初始化成全 0（标准做法） |
| ⑥ | 建一个字典，值是**空列表** `[]`，后面每轮往里 `append` 追加数字，用来画曲线 |

### 4.3 心脏：`train_step()`（前向 → 损失 → 反向 → 更新）

```python
def train_step(self, X, Y, lr):
    m = len(X)                                   # ①
    A2, (X_in, A1) = self.forward(X)             # ②
    eps = 1e-12                                  # ③
    loss = -np.sum(Y * np.log(A2 + eps)) / m     # ④

    dZ2 = (A2 - Y) / m                           # ⑤
    dW2 = A1.T @ dZ2                             # ⑥
    db2 = dZ2.sum(axis=0)                        # ⑦
    dA1 = dZ2 @ self.W2.T                        # ⑧
    dZ1 = dA1 * A1 * (1.0 - A1)                  # ⑨
    dW1 = X_in.T @ dZ1                           # ⑩
    db1 = dZ1.sum(axis=0)

    self.W1 -= lr * dW1                          # ⑪
    self.b1 -= lr * db1
    self.W2 -= lr * dW2
    self.b2 -= lr * db2
    return float(loss)                           # ⑫
```

| 行 | 逐行解释 |
|---|---|
| ① | `len(X)` = 这一批有几张图。后面除以它是为了**取平均**，这样学习率不用随批量大小改 |
| ② | 前向传播，返回预测概率 A2，还有"中间结果" X 和 A1（反向传播要用，所以存下来）。`(X_in, A1)` 是解包 |
| ③ | `1e-12` 是 0.000000000001。加它是为了**防止 `log(0)` 变成负无穷**——数值计算的标准防护动作 |
| ④ | 交叉熵损失。`Y * np.log(A2+eps)` 只保留**正确答案那一列**的对数（其余乘 0 消掉），`np.sum` 加起来，取负号，再除以批量数取平均 |
| ⑤ | 反向第一步。`A2 - Y` 就是"预测减答案"（softmax+交叉熵求导后恰好这么简洁），除以 m 是取平均 |
| ⑥ | 第 2 层权重的梯度。`A1.T` 是 (30,10 批次)，`@ dZ2` 得到 (30,10)——**形状必须和 W2 完全一样**，因为待会儿要用它去改 W2 |
| ⑦ | 偏置的梯度 = 把这个批次的误差**加起来**。`axis=0` 表示跨样本压扁 |
| ⑧ | 把误差**传回上一层**。注意这里用 `self.W2.T`（转置）—— 前向是 `A1 @ W2`，反着走就要用它的转置 |
| ⑨ | 乘上 sigmoid 的导数 `a(1-a)`。三个同样形状的数组**逐元素相乘**（`*` 不是矩阵乘法！） |
| ⑩ | 第 1 层权重的梯度。`X_in.T` 是 (784, 批次)，得到 (784,30)，和 W1 同形状 |
| ⑪ | **梯度下降**：`W -= lr × dW`。`a -= b` 是 `a = a - b` 的简写。往梯度的**反方向**走一小步，损失就下降 |
| ⑫ | `float(...)` 把 NumPy 的单个数字转成 Python 普通小数，方便打印和存储 |

### 4.4 训练循环（`train()`）

```python
for epoch in range(1, args.epochs + 1):              # ①
    perm = rng.permutation(n)                        # ②
    epoch_loss, n_batches = 0.0, 0
    for start in range(0, n, args.batch):            # ③
        idx = perm[start:start + args.batch]         # ④
        batch_loss = net.train_step(Xtr[idx], Ytr[idx], args.lr)   # ⑤
        epoch_loss += batch_loss
        n_batches += 1
    val_acc = net.evaluate(Xva, yva)                 # ⑥
```

| 行 | 解释 |
|---|---|
| ① | `range(1, 31)` 产生 1~30，所以写 `+1`（因为 range 不含末尾） |
| ② | 把这轮的样本顺序**随机打乱**（每轮不一样），防止模型记住顺序 |
| ③ | `range(0, n, batch)` = 从 0 到 n，**每次跳 batch 格**：0, 10, 20, ... 用来切小批 |
| ④ | 取出这一批用哪些样本的下标 |
| ⑤ | `Xtr[idx]` 叫**花式索引**：用下标数组一次取出多行 |
| ⑥ | 一个 epoch 结束后，在验证集上考一次，记录准确率 |

### 4.5 `recognize_draw.py` 的 `preprocess_to_28()`

```python
arr = np.asarray(img, dtype=np.float32) / 255.0     # ①
mask = arr > 0.05                                   # ②
if not mask.any():                                  # ③
    return None
rows = np.where(mask.any(axis=1))[0]                # ④
crop = arr[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]  # ⑤
scale = 20.0 / max(h, w)                            # ⑥
im = Image.fromarray((crop*255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)  # ⑦
ys, xs = np.nonzero(a)                              # ⑧
top = int(round(13.5 - ys.mean()))                  # ⑨
top = min(max(top, 0), 28 - a.shape[0])             # ⑩
canvas[top:top+a.shape[0], left:left+a.shape[1]] = a  # ⑪
```

| 行 | 解释 |
|---|---|
| ① | PIL 图片是 0~255 的整数，除以 255 变成 0~1 的小数（和训练数据一致） |
| ② | `arr > 0.05` 对**每个像素**做比较，得到一个 True/False 的"掩码"表 |
| ③ | `mask.any()` = "有没有任何一个 True"，即画板是不是空的 |
| ④ | `mask.any(axis=1)` = 每一行有没有字；`np.where(...)[0]` = 取出是 True 的行号 |
| ⑤ | 用切片把"有字的那块"裁出来（**注意末尾要 +1**，切片不含末尾） |
| ⑥ | MNIST 的标准做法：数字最长边归一到 20 像素 |
| ⑦ | `astype(np.uint8)` 转回 0~255 整数（PIL 要求），`resize` 缩放，`LANCZOS` 是高质量缩放算法 |
| ⑧ | `np.nonzero` 找出所有非 0 像素的坐标（y 数组和 x 数组） |
| ⑨ | 按**质心**（平均位置）居中，13.5 是 28×28 画布的中心 |
| ⑩ | `min(max(...))` 是**夹逼**写法：保证 top 不小于 0、也不超出边界 |
| ⑪ | 把小图**贴到** 28×28 黑画布的对应位置（左侧是切片赋值） |

---

## 第 5 部分 · 遇到问题怎么自己查

| 你想知道 | 怎么做 |
|---|---|
| 这个变量是什么类型 | `print(type(x))` |
| 这个数组多大 | `print(x.shape)` |
| 这个函数怎么用 | `print(help(np.sum))` 或在代码里写 `np.sum?`（Jupyter） |
| 这个对象有哪些方法 | `print(dir(x))` |
| 这行报什么错 | 看最后一行 `XXXError: ...`，复制去搜索 |

**最实用的一招**：在报错那行的前一行加 `print(xxx.shape)` 或 `print(type(xxx))`，
90% 的 bug 都是"形状和你想的不一样"。

---

## 下一步

打开 `notebooks/00_Python基础速成.ipynb`，把里面的例子**自己敲一遍、改一改**——
看懂和会写是两回事，动手才是真学会。
