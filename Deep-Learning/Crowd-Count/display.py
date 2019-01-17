import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore,QtWidgets
import os
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import time
import threading
import numpy as np
import cv2

class CrowdUI(QWidget):

    def __init__(self):
        super().__init__()
        self.datapath = ['/Users/wangqian/PycharmProjects/qt/data1/',
                     '/Users/wangqian/PycharmProjects/qt/data2/',
                     '/Users/wangqian/PycharmProjects/qt/data3/',
                     '/Users/wangqian/PycharmProjects/qt/data4/']
        self.den_path = ['/Users/wangqian/PycharmProjects/qt/data/den1/',
                     '/Users/wangqian/PycharmProjects/qt/data/den2/',
                     '/Users/wangqian/PycharmProjects/qt/data/den3/',
                     '/Users/wangqian/PycharmProjects/qt/data/den4/']
        init_image = QPixmap("data1/1.jpg").scaled(120, 100)


        wlayout = QHBoxLayout()  # 全局
        hlayout = QHBoxLayout()  # 水平
        self.playButton = QPushButton("启动")
        self.playButton.clicked.connect(self.frameUpdate)
        # control_box.addWidget(self.playButton)
        self.leftlist = QListWidget()
        self.leftlist.insertItem(0, '状态')
        # self.txt = QtWidgets.QTextEdit()
        hlayout.addWidget(self.leftlist)
        hlayout.addWidget(self.playButton)
        # hlayout.addWidget(self.txt)
        self.glayout = QGridLayout()
        self.setLayout(self.glayout)
        self.names = [' ', ' ', ' ', ' ',
                      'video1', 'video2', 'video3', 'video4',
                      ' ', ' ', ' ', ' ',
                      'densitymap1', 'densitymap2', 'densitymap3', 'densitymap4']
        self.positions = [(i, j) for i in range(4) for j in range(4)]
        self.label = [0]*8
        k = 0
        for position, name in zip(self.positions, self.names):
            # print(name)
            if (name == ' '):
                self.label[k] = QLabel()
                self.label[k].setPixmap(init_image)
                self.glayout.addWidget(self.label[k], *position)
                k += 1
            else:
                button = QPushButton(name)
                self.glayout.addWidget(button, *position)

        hwg = QWidget()
        gwg = QWidget()
        hwg.setLayout(hlayout)
        gwg.setLayout(self.glayout)
        wlayout.addWidget(hwg)
        wlayout.addWidget(gwg)
        self.setLayout(wlayout)
        self.show()

    def frameUpdate(self):
        # data_files = [filename for filename in os.listdir(self.datapath[i])]
        # data_files = [filename for filename in os.listdir(self.datapath[0])]

        for frame0,frame1,frame2,frame3,frame4,frame5,frame6,frame7 in zip(
                                        [filename for filename in os.listdir(self.datapath[0])],
                                        [filename1 for filename1 in os.listdir(self.datapath[1])],
                                        [filename2 for filename2 in os.listdir(self.datapath[2])],
                                        [filename3 for filename3 in os.listdir(self.datapath[3])],
                                        [filename for filename in os.listdir(self.den_path[0])],
                                        [filename1 for filename1 in os.listdir(self.den_path[1])],
                                        [filename2 for filename2 in os.listdir(self.den_path[2])],
                                        [filename3 for filename3 in os.listdir(self.den_path[3])]):
            path0 = self.datapath[0] + frame0
            path1 = self.datapath[1] + frame1
            path2 = self.datapath[2] + frame2
            path3 = self.datapath[3] + frame3

            path4 = self.den_path[0] + frame4
            path5 = self.den_path[1] + frame5
            path6 = self.den_path[2] + frame6
            path7 = self.den_path[3] + frame7
            density0 = np.load(path4)
            density1 = np.load(path5)
            density2 = np.load(path6)
            density3 = np.load(path7)
            plt.imsave("temp0.jpg", density0, cmap=CM.jet)
            plt.imsave("temp1.jpg", density1, cmap=CM.jet)
            plt.imsave("temp2.jpg", density2, cmap=CM.jet)
            plt.imsave("temp3.jpg", density3, cmap=CM.jet)

            # print(path1)
            self.label[0].setPixmap(QPixmap(path0).scaled(120, 100))
            self.label[1].setPixmap(QPixmap(path1).scaled(120, 100))
            self.label[2].setPixmap(QPixmap(path2).scaled(120, 100))
            self.label[3].setPixmap(QPixmap(path3).scaled(120, 100))

            self.label[4].setPixmap(QPixmap("temp0.jpg").scaled(120, 100))
            self.label[5].setPixmap(QPixmap("temp1.jpg").scaled(120, 100))
            self.label[6].setPixmap(QPixmap("temp2.jpg").scaled(120, 100))
            self.label[7].setPixmap(QPixmap("temp3.jpg").scaled(120, 100))

            QtWidgets.QApplication.processEvents()
            time.sleep(1/25)


    def framesUpdate(self):

        threads = []
        for i in range(4):
            data_files = [filename for filename in os.listdir(self.datapath[i])]
            t = threading.Thread(target=self.frameUpdate, args=(i,data_files))
            threads.append(t)
        # t1 = threading.Thread(target=self.frameUpdate, args=(0))
        # threads.append(t1)
        # t2 = threading.Thread(target=self.frameUpdate, args=(1))
        # threads.append(t2)
        # t3 = threading.Thread(target=self.frameUpdate, args=(2))
        # threads.append(t3)
        # t4 = threading.Thread(target=self.frameUpdate, args=(3))
        # threads.append(t4)

        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CrowdUI()
    sys.exit(app.exec_())