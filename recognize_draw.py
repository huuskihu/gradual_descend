"""
recognize_draw.py —— 最终交付：手写数字识别器（画板上写字，模型实时认）
=====================================================================
【用法】
    python recognize_draw.py
    弹出窗口：左边用鼠标写数字，右边实时显示 AI 的判断和置信度。Clear 清空重写。

【这一步的意义】
第 1 步手写了反向传播，第 3 步训练出 99.3% 的 CNN，
这一步把"训练好的模型"变成真正能用的程序 —— 这叫【推理 inference】或【部署 deployment】。
训练只做一次，推理可以服务千万次。

【本文件最重要的知识点：预处理必须和训练时一致】
模型只在"标准 MNIST 图片"上见过世界，你的输入必须长得像它：
  1. 白字黑底（MNIST 是黑底白字，别弄反！）
  2. 把笔画裁剪出来，缩放到最长边不超过 20 像素（MNIST 官方就是这么归一化的）
  3. 按【质心】贴回 28x28 画布中央（用质心比用外接框中心更均匀）
  4. 用训练时同样的 mean/std 标准化
做错任何一步，准确率都会从 99% 掉到 50% 以下。
"训练和推理的数据分布必须一致"是所有 AI 项目的铁律。

【为什么用 tkinter】Python 自带、零依赖，画板够用（更漂亮可以学 PyQt / 网页前端）。
"""

from __future__ import annotations

import sys                 # 标准库：加模块搜索路径
import tkinter as tk       # 标准库：GUI 图形界面（tk 是它的小名）
from pathlib import Path

import numpy as np         # 数值计算
import torch               # 深度学习
from PIL import Image, ImageDraw   # 第三方库 Pillow：图像读写与绘制

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # 让下面能 import src 里的模块
from mnist_data import load_arrays              # noqa: E402  （这行注释是告诉检查工具：import 位置靠后是有意的）
from step3_torch_cnn import SimpleCNN           # 复用第 3 步定义的网络结构

CANVAS_SIZE = 280          # 界面画板边长（把 28 放大 10 倍，写起来舒服）
BRUSH_WIDTH = 22           # 笔画粗细。MNIST 笔画较粗，太细的笔画模型没见过


def preprocess_to_28(img: Image.Image) -> np.ndarray | None:
    """把画板图片 -> 一张"MNIST 风格"的 28x28 浮点图（0~1）。
    返回 None 表示画板是空的。每一步对应上面注释里的 4 条铁律。"""
    arr = np.asarray(img, dtype=np.float32) / 255.0    # PIL 图片是 0~255 整数，除以 255 变成 0~1

    mask = arr > 0.05                                  # 对每个像素比较，得到 True/False 的"掩码"表
    if not mask.any():                                 # .any() = "有没有任何一个 True"，即画板是否为空
        return None

    # ---- 第 2 步：裁出笔画的外接框 ----
    rows = np.where(mask.any(axis=1))[0]   # mask.any(axis=1) = 每一行有没有字；np.where 取出是 True 的行号
    cols = np.where(mask.any(axis=0))[0]   # axis=0 = 每一列有没有字
    crop = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]   # 切片裁出来（末尾 +1，因为切片不含终点）

    # ---- 第 3 步：缩放到最长边 20 像素（保持长宽比）----
    h, w = crop.shape                      # 形状是 (高, 宽)
    scale = 20.0 / max(h, w)               # 缩放比例
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))   # round 四舍五入；max(1,...) 保证至少 1 像素
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    #    ↑ 先转回 0~255 的 uint8（PIL 要求），再缩放；LANCZOS 是高质量缩放算法
    a = np.asarray(im, dtype=np.float32) / 255.0       # 再变回 0~1

    # ---- 第 4 步：按质心贴回 28x28 画布中心 ----
    ys, xs = np.nonzero(a)                 # 找出所有非 0 像素的 y 坐标数组和 x 坐标数组
    canvas = np.zeros((28, 28), dtype=np.float32)      # 造一张全黑的 28x28 画布
    top = int(round(13.5 - ys.mean()))     # 13.5 是 28x28 的中心；按质心(平均位置)对齐
    left = int(round(13.5 - xs.mean()))
    top = min(max(top, 0), 28 - a.shape[0])     # 夹逼写法：保证不小于 0
    left = min(max(left, 0), 28 - a.shape[1])   # 也不超出右/下边界
    canvas[top:top + a.shape[0], left:left + a.shape[1]] = a   # 切片赋值：把小图贴上去
    return canvas


class App:
    """整个 GUI 程序。class 把"数据(画板、模型)"和"行为(写字、识别)"打包在一起。"""

    def __init__(self, root: tk.Tk, model: SimpleCNN, mean: float, std: float):
        self.model, self.mean, self.std = model, mean, std
        self.model.eval()                                  # 推理模式：关闭 Dropout！忘了这行结果会变差

        root.title("手写数字识别器  |  训练准确率 99.3%（本项目第 3 步训练）")
        root.resizable(False, False)                       # 禁止拖动改变窗口大小

        # ---- 左边：画板 ----
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="pencil")   # Canvas = 一块可以画东西的画布
        self.canvas.grid(row=0, column=0, padx=12, pady=12)    # grid = 按网格摆放（行, 列, 内边距）

        # PIL 图片和 tkinter 画板"平行地画"：PIL 存真实数据（要给模型），tkinter 只负责显示
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)   # "L" = 灰度模式，0 = 全黑
        self.draw = ImageDraw.Draw(self.img)                   # 拿到一支"画笔"
        self.old = None                                        # 记住上一个鼠标位置，用来连线

        # bind = 把"事件"和"处理函数"绑定起来；<ButtonPress-1> = 按下鼠标左键
        self.canvas.bind("<ButtonPress-1>", self.on_press)       # 注意：这里传函数名，不加括号
        self.canvas.bind("<B1-Motion>", self.on_drag)             # 按住左键移动 = 写字
        self.canvas.bind("<ButtonRelease-1>", self.on_release)    # 松开左键

        tk.Button(root, text="Clear (清空)", command=self.clear,
                  font=("Microsoft YaHei", 11)).grid(row=1, column=0, pady=(0, 12))

        # ---- 右边：识别结果 ----
        panel = tk.Frame(root)                                 # Frame = 一个容器，用来分组摆放
        panel.grid(row=0, column=1, rowspan=2, padx=(0, 16), sticky="nsew")
        tk.Label(panel, text="我的判断", font=("Microsoft YaHei", 12), fg="#666").pack(pady=(18, 0))
        self.lbl_pred = tk.Label(panel, text="—", font=("Microsoft YaHei", 72, "bold"), fg="#2563eb")
        self.lbl_pred.pack()                                   # pack = 从上往下依次摆放
        self.lbl_conf = tk.Label(panel, text="在左边写个数字试试", font=("Microsoft YaHei", 10), fg="#666")
        self.lbl_conf.pack(pady=(0, 10))
        tk.Label(panel, text="十个数字的概率", font=("Microsoft YaHei", 10), fg="#666").pack()
        self.lbl_bars = tk.Label(panel, text="", font=("Consolas", 11), justify="left")
        self.lbl_bars.pack(pady=4)

    # ---------------- 画板交互 ----------------
    def on_press(self, event):
        """按下鼠标：记下起点。event 是 tkinter 传来的事件对象，event.x / event.y 是坐标。"""
        self.old = (event.x, event.y)

    def on_drag(self, event):
        """拖动鼠标：画一笔。"""
        x, y = event.x, event.y
        r = BRUSH_WIDTH // 2                       # // 是整除
        # tkinter 显示：画一个白色圆点（连续拖动时圆点连成线）
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        # PIL 真实数据：画线 + 画圆点（圆点让转折处不断线）
        self.draw.line([self.old, (x, y)], fill=255, width=BRUSH_WIDTH)   # 255 = 白色
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.old = (x, y)                          # 更新起点，下一段线从这里开始

    def on_release(self, event):
        """松开鼠标：识别一次（每写完一笔就认）。"""
        self.predict()

    def clear(self):
        """清空画板：tkinter 和 PIL 两边都要清。"""
        self.canvas.delete("all")                                        # 删掉画布上所有图形
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)   # 重新造一张全黑图
        self.draw = ImageDraw.Draw(self.img)                             # 画笔也要重新拿
        self.lbl_pred.config(text="—")                                   # .config 修改控件属性
        self.lbl_conf.config(text="画板已清空")

    # ---------------- 推理 ----------------
    def predict(self):
        """把画板内容送进模型，把结果显示到界面上。"""
        small = preprocess_to_28(self.img)
        if small is None:                          # 画板是空的，什么都不做
            return

        # 调试彩蛋：把喂给模型的 28x28 小图放大存下来，亲眼看看模型"看到"了什么
        Image.fromarray((small * 255).astype(np.uint8)).resize((112, 112), Image.NEAREST)\
             .save(PROJECT_ROOT / "outputs" / "last_input_28x28.png")

        # 第 5 步：用训练时同样的 mean/std 标准化，再 reshape 成 (1, 784) 给模型
        x = torch.from_numpy(((small - self.mean) / self.std).reshape(1, 784))
        with torch.no_grad():                                  # 推理不需要梯度，省显存又提速
            probs = torch.softmax(self.model(x), dim=1)[0]     # 得分 -> 概率；[0] 取出这一张的结果

        conf, pred = probs.max(0)                              # 最大概率值 和 它的下标
        self.lbl_pred.config(text=str(int(pred)))
        self.lbl_conf.config(text=f"置信度 {conf.item() * 100:.1f}%")

        # 用字符画一个简单的概率条形图（零依赖方案）
        lines = []
        for d in range(10):
            p = probs[d].item()                    # .item() 把单元素张量变成 Python 小数
            bar = "█" * int(round(p * 24))          # 字符串乘数字 = 重复
            mark = " <-- " if d == int(pred) else ""
            lines.append(f"{d} │{bar:<24s}{p * 100:5.1f}%{mark}")   # :<24s 左对齐占 24 宽
        self.lbl_bars.config(text="\n".join(lines))            # join 把列表拼成一个字符串


def main():
    ckpt_path = PROJECT_ROOT / "models" / "mnist_cnn.pt"
    if not ckpt_path.exists():
        # sys.exit(字符串) 会打印这句话并结束程序
        sys.exit("找不到 models/mnist_cnn.pt，请先运行: python src/step3_torch_cnn.py")

    device = torch.device("cpu")      # 单张 28x28 图 CPU 推理不到 1 毫秒，没必要占 GPU
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SimpleCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])   # 把训练好的参数填进网络
    mean, std = ckpt["standardize"]             # 取出训练时用的标准化参数

    # 【好习惯】部署前的"冒烟测试"：先用 5 张测试图确认模型加载正确
    arrays = load_arrays()
    Xte, yte = arrays["test"]
    with torch.no_grad():
        sample = torch.from_numpy(((Xte[:5] - mean) / std))
        preds = model(sample).argmax(1).numpy()
    print(f"冒烟测试 - 测试集前5张 预测: {preds}  真实: {yte[:5]}  "
          f"{'一致' if (preds == yte[:5]).all() else '不一致'}")

    root = tk.Tk()                 # 创建窗口
    App(root, model, mean, std)    # 把我们的界面装进去
    root.mainloop()                # 进入事件循环：持续监听鼠标/键盘，直到窗口关闭


if __name__ == "__main__":
    main()
