# gradual_descend
GD algorithm

----------------

# 手写数字识别器 —— 从 2012 年的老代码到 2026 年的完整实战

> 灵感来源：3b1b深度学习系列视频[【官方双语】深度学习之神经网络的结构 Part 1 ver 2.0-哔哩哔哩】] (https://b23.tv/5TdNQtn “video”) + Michael Nielsen《Neural Networks and Deep Learning》
> （原仓库代码依赖 Theano，已于 2017 年停止维护，本项目用 NumPy + PyTorch 全部重写，
> **但那份人工标注的 70000 张手写数字图片（MNIST）是经典数据集，完全可继续使用。**）

## 项目成果一览

| 步骤 | 模型 | 测试集准确率 | 训练耗时 | 用了什么工具 |
|---|---|---|---|---|
| Step 1 | 784-30-10 全连接（视频原版结构） | **95.66%** | 8.4 s | 纯 NumPy，手写反向传播 |
| Step 2 | 784-128-10 MLP + ReLU + Adam | **97.64%** | 18.4 s | PyTorch，GPU |
| Step 3 | CNN（Conv+Pool+Dropout） | **99.32%** | 13.4 s | PyTorch，GPU |
| 最终 | `recognize_draw.py` 画板识别器 | 实时推理 <10 ms | — | tkinter + 上面的 CNN |

视频时代的原版效果约 95%，本项目最终做到了 **99.3%**，并做成��可以在屏幕上写字的交互识别器。

## 快速开始

> ⚠️ **先确认你使用的是正确的 Python 环境。** 如果电脑里安装了多个 Python，
> 命令行中的 `python` 可能不是安装了项目依赖的那个解释器。推荐使用本目录的 `run.cmd`，
> 或在 PyCharm / VS Code 中选择正确的项目解释器。详见下方[用哪个 Python](#用哪个-python常见坑)。

```bash
cd <项目目录>

run 00_env_check.py                  # 0. 环境自检（推荐，run.cmd 自动选择项目解释器）
run src\step0_python_basics.py       # 1. Python 语法速成（零基础先跑这个）
run src\mnist_data.py                # 2. 认识数据（生成样例图）
run src\step1_numpy_nn.py            # 2. 纯手写神经网络（教学核心）
run src\step2_torch_mlp.py           # 3. PyTorch 版，体会框架的便利
run src\step3_torch_cnn.py           # 4. CNN，冲击 99%
run src\step4_ablation.py            # 5. 对比实验：激活函数/优化器
run recognize_draw.py                # 6. 手写识别器（画板写数字，AI 实时认）
```

若不使用 `run.cmd`，请使用当前 Python 环境运行：

```bash
python src/step1_numpy_nn.py
```

## 目录结构

```
ml/
├── 00_env_check.py            环境自检
├── recognize_draw.py          ★ 最终成品：画板手写识别器
├── requirements.txt           依赖清单
├── notebooks/
│   └── 01_互动教学_从零手写神经网络.ipynb   ★零基础推荐入口：逐格运行、当场看结果
├── data/mnist.pkl.gz          MNIST 数据（70000 张人工标注的手写数字）
├── src/
│   ├── step0_python_basics.py  ★Python 基础速成（可运行，每行都有注释）
│   ├── mnist_data.py           数据加载与预处理（共用模块）
│   ├── step1_numpy_nn.py       ★纯 NumPy 手写网络：前向/反向/梯度下降，逐行注释
│   ├── step2_torch_mlp.py     PyTorch 版 MLP：autograd/优化器/GPU
│   ├── step3_torch_cnn.py     CNN：卷积/池化/Dropout
│   └── step4_ablation.py      消融实验：sigmoid vs ReLU，SGD vs Adam
├── docs/
│   ├── 术语手册_零基础版.md  ★零基础先看这个：@/.T/argmax 等符号 + 全部行话的人话解释
│   └── 教学讲义.md           配套讲义：原理图解 + 常见问题 + 现代进展
├── models/                    训练好的模型权重 (.pt)
└── outputs/                   训练曲线、样例图、调试输出
```

## 零基础推荐入口：Jupyter Notebook

`notebooks/01_互动教学_从零手写神经网络.ipynb` —— **26 格（12 个代码格），已全部执行验证通过**，
代码里带逐行注释。每一格都能单独运行、当场看到结果，比读代码更适合入门。

```bash
python -m jupyter lab          # 或直接在 PyCharm / VS Code 里打开 .ipynb
```

打开后：选中格子按 `Shift + Enter` 逐格运行（PyCharm 点格子左侧绿色三角）。
内容：数据长什么样 → `@` 是什么意思 → 激活函数 → 前向传播 → 损失 →
反向传播 → 真的训练 → PyTorch 对比 → 加载 99.3% 的 CNN 看它认字。

> 另有一个 `notebooks/00_Python基础速成.ipynb`，内容与 `src/step0_python_basics.py` 对应，
> **二选一即可** —— 推荐用 `.py`（讲解内嵌在代码里，PyCharm 里直接跑）。

## 学习路线（零基础推荐顺序）

> **notebook 和 .py 的关系**：notebook 是"带解说的导览"（9 关，边跑边看输出），
> .py 是"能打包带走的工程成品"（能存模型、带命令行参数）。
> **先用 notebook 走一遍全流程，再回头读 .py**，理解会快很多。
> （notebook 第 7 关其实就是在调用 `step1_numpy_nn.py` 里的 `Network` 类。）

| 顺序 | 做什么 | 花多久 | 目的 |
|---|---|---|---|
| ① | `python src/step0_python_basics.py` | 20~30 分钟 | 补 Python 语法（每行都有注释） |
| ② | **打开 `notebooks/01_互动教学_从零手写神经网络.ipynb`** 逐格跑 | 40~60 分钟 | **主线**：9 关走完整个流程，每步都能看到真实输出 |
| ③ | 读 `src/step1_numpy_nn.py`，重点 `train_step()` | 30 分钟 | 第二遍：看完整工程代码，反向传播那 20 行 |
| ④ | 读 `src/step2_torch_mlp.py` | 15 分钟 | 找出"框架帮你省掉的是哪几行" |
| ⑤ | 读 `src/step3_torch_cnn.py` 并跑一次 | 15 分钟 | 理解卷积"局部连接 + 权重共享"为什么强 |
| ⑥ | `python src/step4_ablation.py` | 5 分钟 | 控制变量，用数据说话 |
| ⑦ | `python recognize_draw.py` | 随时玩 | 训练好的模型变成真正能用的程序 |

**时间紧的话**：① → ② → ⑦ 三步就够（约 1.5 小时走完全流程）。
**卡住了就看** `docs/术语手册_零基础版.md`（查行话）和 `docs/Python基础与逐行讲解.md`（查语法）。

## PyCharm 里一堆红色"未解析的引用"？那是误报

**先说结论：这些警告不影响运行。** notebook 实测执行 0 报错。它们是 PyCharm 的静态检查
在不了解项目结构时的误报，常见几类：

| 警告 | 真相 |
|---|---|
| 未解析的引用 'mnist_data' / 'Network' / 'MLP' ... | PyCharm 不知道 `src` 目录是源码位置（第 0 关是动态加的 sys.path，静态检查看不见） |
| 在 '' 中找不到引用 '__version__' / 'device' | 没给项目配解释器，PyCharm 不知道 torch 是谁 |
| 拼写错误 'preds' / 'numel' | 拼写检查器不认识这些词，纯属骚扰 |
| 隐藏了来自外部作用域的名称 'x' | 不同格子里复用了变量名，在 notebook 里很正常 |
| float 没有 __getitem__ | 类型推断的误报 |

**治本三步**（配好之后 90% 的红字消失）：

1. **打开项目文件夹而不是单个文件**：在 PyCharm 中选择项目根目录作为项目打开。
2. **配置解释器**：`Ctrl+Alt+S > Project > Python Interpreter`，选择已安装项目依赖的 Python 解释器。
3. **标记源码根目录**：在左侧项目树右键 `src` 文件夹 > `Mark Directory as > Sources Root`。
   （标记后 `src` 变蓝色），`import mnist_data` 等引用就能解析了。

剩下的拼写警告可以关闭：`Ctrl+Alt+S > Editor > Inspections > Proofreading > Typo` 取消勾选。
判定真伪的终极标准：**跑一遍 `00_env_check.py` 全绿 = 环境没问题**；
notebook 里格子能出结果 = 代码没问题。IDE 的红字只是"猜测"。

## 现代进展：2012 → 2026，这片天地已经变了

视频和原书都是 2012 年前后的事。十几年间发生了什么（时间线）：

**2012 AlexNet** —— CNN 在 ImageNet 竞赛一战成名，错误率从 26% 降到 15%，
GPU 训练 + ReLU + Dropout 的组合成为标配，"深度学习十年浪潮"从此开始。

**2015 ResNet（残差网络）** —— 残差连接让上千层的网络也能稳定训练，
ImageNet top-5 错误率降到 3.6%（超过人类）。ResNet 至今仍是图像任务的骨干网络。

**2017 Transformer（《Attention Is All You Need》）** —— 本为机器翻译设计，
凭"自注意力机制"横扫 NLP。**这是本世纪最重要的论文之一**：没有它就没有 ChatGPT。

**2020 Vision Transformer (ViT)** —— 证明纯注意力结构也能做视觉，
不需要卷积。此后"Transformer 一统江湖"：文本、图像、语音、视频用同一套架构。

**2022 至今 大模型时代** —— ChatGPT（GPT 系列）、多模态大模型（看图写字）、
Stable Diffusion / Midjourney（扩散模型文生图）、Sora（视频生成）。
今天的"手写识别"早已不是独立课题：拿一张写满公式的草稿纸给多模态大模型，
它能逐字识别、翻译并解释含义。

**对手写数字/文字识别（OCR）这个具体任务而言，今天的工业级方案：**
- PaddleOCR（百度开源）、Tesseract：几行代码识别整页文档，支持 80+ 语言
- 云服务 OCR API：身份证、票据、手写表单直接结构化输出
- 多模态大模型：复杂版面、数学手稿、潦草笔迹都能处理
- MNIST 本身也有了扩展版：EMNIST（含字母）、Kuzushiji-MNIST（日文古书法）

**但为什么今天仍要学 MNIST + 手写网络？**
因为麻雀虽小五脏俱全：前向传播、损失函数、反向传播、优化器、过拟合、
训练/验证/测试划分、数据泄露……这些概念是所有现代模型的地基，
而 MNIST 让你在 10 秒内跑完一次完整训练，是理解这一切的最佳沙盘。

## 用哪个 Python（常见坑）

如果电脑上安装了多个 Python，命令行中的 `python` 可能指向没有安装项目依赖的解释器。
先运行下面的命令检查当前使用的是哪个 Python：

```bash
python --version
python -c "import sys; print(sys.executable)"
```

如果输出的解释器不是你安装项目依赖的环境，请使用以下任一方式：

1. **使用项目里的 `run.cmd`**（Windows）：`run src\step1_numpy_nn.py`
2. **在 PyCharm / VS Code 中选择正确的 Python 解释器**，并确认已安装 `requirements.txt` 中的依赖。
3. **使用虚拟环境**，避免不同项目之间的依赖互相影响：

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Windows cmd
   .venv\Scripts\activate.bat
   # macOS / Linux
   source .venv/bin/activate
   ```

如果遇到 `ModuleNotFoundError: No module named 'numpy'`，通常是当前解释器没有安装项目依赖，
或命令行使用了与 IDE 不同的 Python。激活正确环境后安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 常见问题（详见 docs/教学讲义.md）

- **为什么用训练集 99.5% 而验证集只有 97.8%？** —— 过拟合，Step 2 里有详细分析。
- **学习率怎么选？** —— 先用默认值跑通，再看损失曲线：震荡→调小；不下降→调大。
- **为什么我的结果和你的不一样？** —— 随机初始化不同。固定 `--seed` 可复现。

## 环境说明

项目依赖和安装方式请参见 `requirements.txt`。实际训练速度和 GPU 加速效果会因操作系统、
Python 版本、PyTorch 版本以及硬件配置而异。
