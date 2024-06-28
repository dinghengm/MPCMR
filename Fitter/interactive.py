import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QGridLayout
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import math

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("N x 2D Numpy Array Viewer")

        # 初始化一个列表，包含多个随机的2D numpy数组
        self.data_list = [np.random.rand(10, 10) for _ in range(7)]
        self.canvas_list = []

        self.grid_layout = QGridLayout()

        # 动态添加canvas并展示2D numpy数组
        for idx, data in enumerate(self.data_list):
            canvas = MplCanvas(self)
            canvas.ax.imshow(data, cmap='viridis')
            canvas.mpl_connect("button_press_event", lambda event, data=data: self.onclick(event, data))
            self.canvas_list.append(canvas)
            row = idx // 3
            col = idx % 3
            self.grid_layout.addWidget(canvas, row, col)

        self.value_label = QLabel("Click on a pixel to see its value")
        self.value_label.setAlignment(Qt.AlignCenter)

        self.canvas_gaussian = MplCanvas(self)

        left_layout = QVBoxLayout()
        left_layout.addLayout(self.grid_layout)
        left_layout.addWidget(self.value_label)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addWidget(self.canvas_gaussian)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def onclick(self, event, data):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            value = data[y, x]
            self.value_label.setText(f"Value at ({x}, {y}): {value}")

            # 生成一个以该像素点数值为输入的随机高斯图像
            size = 100
            mean = value * 10  # 举例：以像素值作为均值的一个因子
            stddev = 1
            gaussian_data = np.random.normal(mean, stddev, (size, size))

            # 在旁边的窗口中显示随机高斯图像
            self.canvas_gaussian.ax.clear()
            self.canvas_gaussian.ax.imshow(gaussian_data, cmap='viridis')
            self.canvas_gaussian.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())