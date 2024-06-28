import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton, QFileDialog,QGridLayout
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
sys.path.append(r'C:\Research\MRI\Ungated\MPCMR\MPEPI')
from libMapping_v14 import *

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPEPI's T1 viewer")

        #We don't have any obj at first
        self.obj = None
        self._map = None
        self.canvas_list = []

        self.grid_layout = QGridLayout()

        self.value_label = QLabel("Click on a pixel to see its value")
        self.value_label.setAlignment(Qt.AlignCenter)

        self.canvas_plot = MplCanvas(self)

        self.load_button = QPushButton("Load mapping object")
        self.load_button.clicked.connect(self.load_file)

        left_layout = QVBoxLayout()
        left_layout.addLayout(self.grid_layout)
        left_layout.addWidget(self.value_label)
        left_layout.addWidget(self.load_button)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addWidget(self.canvas_plot)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open mapping object", "", "mapping Files (*.mapping)")

        if file_path:
            self.obj= mapping(data=file_path)
            self._data=self.obj._data
            self._map=self.obj._map
            for canvas in self.canvas_list:
                canvas.ax.clear()

            self.imshow_map()

    
    def imshow_map(self):
        obj=self.obj

        crange=obj.crange
        cmap=obj.cmap
        Nz=obj.Nz
        map=self._map.squeeze()
        self.canvas_list=[]
        for sl in range(Nz):
            canvas=MplCanvas(self)
            map_show=map[...,sl]
            canvas.ax.imshow(map_show,vmin=crange[0],vmax=crange[1], cmap=cmap)
            canvas.ax.set_title(f'Slice{sl}')
            canvas.ax.axis('off')
            canvas.mpl_connect("button_press_event",lambda event, slice=sl: self.onclick(event, slice))
            self.canvas_list.append(canvas)
            row = sl // 2
            col = sl % 2
            self.grid_layout.addWidget(canvas, row, col)


    def onclick(self, event,slice):
        if self._map is not None and event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            map=self._map
            TIlist=self.obj.valueList
            map_show_value=map[y,x,slice].squeeze()
            data=self._data
            data_read=data[y,x,slice].squeeze()
            T1_final,ra_final,rb_final,returnInd,res=ir_fit(data_read,TIlist)

            self.value_label.setText(f"Value at ({x}, {y}): T1={map_show_value}\nT1= {T1_final},ra={ra_final}\nrb={rb_final},returnInd={returnInd},res={res} ")

            self.canvas_plot.ax.clear()
            x_plot=np.arange(start=1,stop=TIlist[-1],step=1)

            ydata_exp=abs(ir_recovery(x_plot,T1_final,ra_final,rb_final))
            self.canvas_plot.ax.plot(x_plot,ydata_exp)
            self.canvas_plot.ax.scatter(TIlist,data_read)
            self.canvas_plot.ax.legend(['Mz_Read'])
            self.canvas_plot.ax.set_xlabel('Time (ms)')
            self.canvas_plot.ax.set_ylabel('Magnetization')
            self.canvas_plot.ax.set_title(f'T1 value={T1_final}')
            self.canvas_plot.ax.grid('True')
            self.canvas_plot.draw()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())