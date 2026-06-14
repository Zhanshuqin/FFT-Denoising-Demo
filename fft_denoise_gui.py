import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import sys
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DenoiseDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 + PyTorch: FFT 频域降噪演示")
        self.setGeometry(100, 100, 1000, 700)

        # 信号参数
        self.fs = 1000          # 采样率 1000 Hz
        self.duration = 1.0     # 1 秒
        self.t = torch.linspace(0, self.duration, int(self.fs * self.duration))
        self.clean_signal = torch.sin(2 * np.pi * 5 * self.t)   # 5Hz 正弦波
        self.noise = None
        self.noisy_signal = None
        self.filtered_signal = None

        # 生成初始含噪信号
        self.update_noise()

        # UI 布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 按钮区域
        btn_layout = QHBoxLayout()
        self.denoise_btn = QPushButton("执行 FFT 降噪")
        self.denoise_btn.clicked.connect(self.fft_denoise)
        self.new_noise_btn = QPushButton("重新生成噪声")
        self.new_noise_btn.clicked.connect(self.update_noise_and_plot)
        btn_layout.addWidget(self.denoise_btn)
        btn_layout.addWidget(self.new_noise_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 添加状态标签
        self.status_label = QLabel("就绪 | 噪声强度: 0.3")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Matplotlib 画布
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # 初始绘图
        self.plot_signals()

    def update_noise(self):
        """生成新的高斯噪声（固定强度 0.3）"""
        self.noise = 0.3 * torch.randn_like(self.clean_signal)
        self.noisy_signal = self.clean_signal + self.noise
        self.filtered_signal = None   # 重置降噪结果

    def update_noise_and_plot(self):
        """更换噪声并刷新图形"""
        self.update_noise()
        self.status_label.setText("已生成新噪声 | 点击按钮进行 FFT 降噪")
        self.plot_signals()

    def fft_denoise(self):
        """
        使用 PyTorch 进行 FFT 降噪（频域硬阈值 + 低通滤波）
        策略：保留幅度谱中最大的 30% 频率分量，其余置零。
        """
        if self.noisy_signal is None:
            return

        # 转为复数张量并做 FFT
        x = self.noisy_signal.clone()
        X = torch.fft.fft(x)                   # 复数频谱
        magnitude = torch.abs(X)               # 幅度谱
        phase = torch.angle(X)                 # 相位谱

        # 方法1：保留幅度最大的 30% 分量（自适应降噪）
        k = int(0.3 * len(magnitude))          # 保留的分量数量
        if k > 0:
            # 获取幅度最大的 k 个索引
            _, indices = torch.topk(magnitude, k)
            mask = torch.zeros_like(magnitude, dtype=torch.bool)
            mask[indices] = True
            # 其他位置置零
            X_filtered = X * mask
        else:
            X_filtered = X

        # 也可改用低通滤波器（保留前 50 个频率），取消下面注释并注释上面部分：
        # cutoff = 50
        # mask = torch.zeros_like(magnitude, dtype=torch.bool)
        # mask[:cutoff] = True
        # X_filtered = X * mask

        # 逆 FFT 得到时域信号（取实部，忽略微小虚部）
        filtered = torch.fft.ifft(X_filtered).real

        self.filtered_signal = filtered.numpy()   # 转为 numpy 用于绘图
        self.status_label.setText("已执行 FFT 降噪（保留幅度最大的30%频率分量）")
        self.plot_signals(show_filtered=True)

    def plot_signals(self, show_filtered=False):
        """更新图形：显示干净信号、含噪信号，以及可选降噪后信号"""
        self.figure.clear()

        # 转换为 numpy 用于绘图
        t_np = self.t.numpy()
        clean_np = self.clean_signal.numpy()
        noisy_np = self.noisy_signal.numpy()

        # 子图1：原始干净信号
        ax1 = self.figure.add_subplot(311)
        ax1.plot(t_np, clean_np, 'b-', linewidth=1.5, label='原始正弦波 (5Hz)')
        ax1.set_ylabel('幅值')
        ax1.set_title('原始信号 (无噪声)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 子图2：含噪信号
        ax2 = self.figure.add_subplot(312)
        ax2.plot(t_np, noisy_np, 'r-', linewidth=0.8, alpha=0.8, label='含噪信号')
        if show_filtered and self.filtered_signal is not None:
            ax2.plot(t_np, self.filtered_signal, 'g--', linewidth=1.5, label='降噪后信号')
        ax2.set_ylabel('幅值')
        ax2.set_title('含噪信号 (SNR ≈ 8 dB)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 子图3：频谱对比（可选，显示降噪前后的幅度谱）
        ax3 = self.figure.add_subplot(313)
        # 计算含噪信号的幅度谱
        noisy_fft = torch.fft.fft(self.noisy_signal)
        mag_noisy = torch.abs(noisy_fft).numpy()[:200]   # 只显示低频部分
        freqs = np.fft.fftfreq(len(self.t), d=1/self.fs)[:200]
        ax3.plot(freqs, mag_noisy, 'r-', alpha=0.6, label='含噪信号频谱')

        if show_filtered and self.filtered_signal is not None:
            filtered_tensor = torch.from_numpy(self.filtered_signal)
            filtered_fft = torch.fft.fft(filtered_tensor)
            mag_filtered = torch.abs(filtered_fft).numpy()[:200]
            ax3.plot(freqs, mag_filtered, 'g-', linewidth=1.5, label='降噪后频谱')
        ax3.set_xlabel('频率 (Hz)')
        ax3.set_ylabel('幅度')
        ax3.set_title('幅度谱 (局部)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = DenoiseDemo()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()