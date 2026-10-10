"""
00_env_check.py —— 环境自检（换任何一台新机器，先跑这个）
=====================================================================
【为什么需要】深度学习初学者 80% 的时间浪费在环境问题上。
先跑这个脚本，30 秒确认：Python 版本、库齐不齐、GPU 能不能用、数据在不在。
"""

from __future__ import annotations

import importlib            # 标准库：动态导入模块（用来"试探"某个库装没装）
import platform             # 标准库：查看操作系统和 Python 版本
import sys                  # 标准库：sys.executable 能告诉我们当前用的是哪个 python.exe
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # 本文件所在目录 = 项目根目录

print("=" * 62)
print("环境自检报告")
print("=" * 62)
# platform.python_version() 返回版本字符串；sys.executable 是当前解释器的完整路径
print(f"Python     : {platform.python_version()}   ({sys.executable})")

# ----------------------------------------------------------------------
# 1. 依赖库逐个检查
# ----------------------------------------------------------------------
# 字典：库名 -> 说明文字
required = {"numpy": "NumPy（矩阵运算）", "torch": "PyTorch（深度学习框架）",
            "matplotlib": "matplotlib（画图）", "PIL": "Pillow（图像读写）",
            "tkinter": "tkinter（GUI 画板）"}
ok = True
for mod, desc in required.items():          # .items() 遍历字典的 (键, 值)
    try:
        m = importlib.import_module(mod)    # 尝试导入；失败会抛 ImportError
        # getattr(对象, 属性名, 默认值)：取不到就返回默认值，避免某些库没有 __version__ 时报错
        ver = getattr(m, "__version__", "built-in")
        print(f"[OK]  {mod:<11s} {ver:<12s} {desc}")     # :<11s 左对齐占 11 个字符宽
    except ImportError:
        ok = False
        print(f"[缺失] {mod:<11s} {'-':<12s} {desc}  ->  pip install {mod}")

# ----------------------------------------------------------------------
# 2. GPU 检查（torch 缺失时不能 import，否则脚本会崩 —— 先判断再 import）
# ----------------------------------------------------------------------
try:
    import torch
except ImportError:
    torch = None
    ok = False
    # 【最常见的坑】装了库的 Python 和当前运行的 Python 不是同一个。
    # 系统里有好几个 Python 时，cmd 里敲 python 未必是你想要的那个。
    print("\n" + "!" * 62)
    print("当前这个 Python 里没有 torch，你很可能【用错了解释器】。")
    print(f"现在用的是 : {sys.executable}")
    print("解决方法（任选其一）：")
    print("  1) 用完整路径运行（最稳）:")
    print(r'     D:\Program Files\python\python.exe 00_env_check.py')
    print(r"  2) 或用本目录的 run.cmd：run 00_env_check.py")
    print(r"  3) PyCharm: Settings > Python Interpreter > 选 D:\Program Files\python\python.exe")
    print(r"  4) 一劳永逸：把 D:\Program Files\python 移到系统 PATH 的最前面")
    print("!" * 62 + "\n")
else:
    if torch.cuda.is_available():           # 显卡能不能用 PyTorch
        print(f"[OK]  GPU        {torch.cuda.get_device_name(0)}  "
              f"(CUDA {torch.version.cuda}, 显存 {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        x = torch.randn(1000, 1000, device="cuda")   # 在 GPU 上造一个 1000x1000 的随机矩阵
        _ = (x @ x).sum().item()                     # 真跑一次矩阵乘法，确认 GPU 计算正常
        print("[OK]  GPU 计算正常（1000x1000 矩阵乘法已测试）")
    else:
        print("[警告] GPU 不可用，将使用 CPU 训练（本项目也能跑，只是慢）")

# ----------------------------------------------------------------------
# 3. 数据 / 模型文件检查
# ----------------------------------------------------------------------
data = ROOT / "data" / "mnist.pkl.gz"
if data.exists():
    print(f"[OK]  数据       {data}  ({data.stat().st_size / 1e6:.1f} MB)")   # .stat().st_size 是字节数
else:
    ok = False
    print(f"[缺失] 数据       请把原仓库 data/mnist.pkl.gz 复制到 {data}")

model = ROOT / "models" / "mnist_cnn.pt"
# 三元表达式写在一行里：条件为真取前者，为假取后者
print(f"[{'OK' if model.exists() else '提示'}]  模型文件  "
      + (str(model) if model.exists() else "还没训练，先运行 src/step3_torch_cnn.py"))

print("=" * 62)
print("结论: 一切就绪，可以开始学习！" if ok else "结论: 请先解决上面的 [缺失] 项。")
