# FFT-Denoising-Demo-频域降噪演示工具 —— PyQt5 + PyTorch
一个轻量级桌面应用，用于直观展示快速傅里叶变换（FFT）在信号降噪中的应用。  程序生成含高斯噪声的 5 Hz正弦波，通过频域幅度阈值滤波（保留 30% 最大幅度分量）实现实时降噪，并动态更新波形与频谱图。
#功能特点
生成干净5 Hz正弦波+可调节高斯噪声
一键执行 FFT 频域降噪
实时显示三个子图：原始干净信号、含噪信号（红色）与降噪后信号（绿色虚线） 
降噪前后的幅度谱对比：支持“重新生成噪声”以多次测试算法鲁棒性
完全基于PyQt5构建GUI，Matplotlib动态绘图，PyTorch完成张量与 FFT 计算
#运行环境
Python 3.8、PyQt5、 PyTorch、Matplotlib、 NumPy
#快速开始
```bash
# 克隆仓库
git clone https://github.com/Zhanshuqin/FFT-Denoising-Demo.gitcd FFT-Denoising-Demo
# 安装依赖
pip install pyqt5 torch matplotlib numpy
# 运行程序
python fft_denoise_gui.py
