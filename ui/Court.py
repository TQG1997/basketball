from PyQt5.QtWidgets import *
from PyQt5 import QtCore

import os
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use("qt5agg")

COURT_PNG = os.path.join(os.path.dirname(__file__), "images/court.png")
LABEL_EFFECT = [pe.withStroke(linewidth=2, foreground='black')]

class Court(QWidget):
    def __init__(self,parent = None):
        QWidget.__init__(self)
        self.setLayout(QVBoxLayout())
        self.canvas = plotCanvas(self)
        self.cond_ = -1
        self.is_playing = False
        self.loaded = False

        self.setFixedSize(570,580)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        self.setStyleSheet("background-color: white;")
        self.ani_timer = QtCore.QTimer(self)
        self.sim_timer = QtCore.QTimer(self)

        self.ani_timer.timeout.connect(self.canvas.on_start)
        self.sim_timer.timeout.connect(self.start_sim)

    def start_sim(self):
        self.cond_ = 0
        output_path = os.path.join(os.path.dirname(__file__), 'Data', 'output', 'output.npy')
        if not os.path.exists(output_path):
            print(f"Output not found: {output_path} — run Generate first")
            return
        try:
            data = np.load(output_path)
            self.canvas.on_start_G(data, self.cond_)
        except Exception as e:
            print(f"Error loading simulation: {e}")

    def get_len(self, cond):
        l = 0
        base = os.path.dirname(__file__)
        try:
            if cond == 1:
                pts_path = os.path.join(base, 'Points', 'points2.npy')
                if os.path.exists(pts_path):
                    data = np.load(pts_path)
                    l = len(data)
            elif cond == 2:
                out_path = os.path.join(base, 'Data', 'output', 'output.npy')
                if os.path.exists(out_path):
                    data = np.load(out_path)
                    play = data[0, self.cond_]
                    l = len(play)
        except Exception:
            pass
        return l


class plotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5.7, height=6.0, dpi=100):
        self.fig = Figure(figsize=(width,height),dpi=dpi)
        self.frame_id = 0

        self.fig.patch.set_facecolor('none')
        self.axes = self.fig.add_subplot(111)

        court = plt.imread(COURT_PNG)
        self.axes.imshow(court, zorder=0, extent=[0, 94, 50, 0])
        self.axes.set_xlim(47,94)
        self.axes.axis('off')

        self.fig.tight_layout(rect=[0, 0,1, 0.97])

        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)

        FigureCanvas.resize(self,570,580)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


########################################################################################
    def update_all(self,frame_id,line_path, player_circles, ball_circle, annotations, data):
        # players
        for j, circle in enumerate(player_circles):
            circle.center = data[frame_id, 2 + j *
                                 2 + 0], data[frame_id, 2+ j * 2 + 1]
            annotations[j].set_position(circle.center)
        j = 0
        size_count = 2
        #23 lines, 5 players 1 ball, 4 lines each
        for i, line in enumerate(line_path):
            start = max((frame_id - size_count, 0))
            end = max((frame_id,0))
            if i > 5:
                j = i%6

            if j == 0:
                line.set_data(data[start:end, j * 2], data[start:end, j * 2 + 1])
            else:
                line.set_data(data[start:end, j * 2], data[start:end, j * 2 + 1])

            if (i+1) % 6 == 0:
                size_count += 1

            j += 1

        # ball
        ball_circle.center = data[frame_id, 0], data[frame_id, 1]

        return

    def update_all2(self,frame_id,line_path, player_circles, ball_circle, annotations, data):
        # players
        for j, circle in enumerate(player_circles):
            circle.center = data[frame_id, 2 + j *
                                 2 + 0], data[frame_id, 2+ j * 2 + 1]
            annotations[j].set_position(circle.center)

        j = 0
        size_count = 2
        #23 lines, 5 players 1 ball, 4 lines each
        for i, line in enumerate(line_path):
            start = max((frame_id - size_count, 0))
            end = max((frame_id,0))
            if i > 10:
                j = i%11

            if j == 0:
                line.set_data(data[start:frame_id, j * 2], data[start:frame_id, j * 2 + 1])
            else:
                line.set_data(data[start:end, j * 2], data[start:end, j * 2 + 1])

            if (i+1) % 11 == 0:
                size_count += 1

            j += 1

        ball_circle.center = data[frame_id, 0], data[frame_id, 1]

        return


    def plot_data(self,data, length):
        self.axes.cla()
        court = plt.imread(COURT_PNG)
        self.axes.imshow(court, zorder=0, extent=[0, 94, 50, 0])

        player_circles = []
        line_path = []

        ball_circle = plt.Circle(xy=(0, 0), radius=0.9,zorder=10,edgecolor=None,facecolor='g')

        name_list = ['PG', 'SG', 'SF', 'PF', 'C']
        # team A -> read circle, team B -> blue circle, ball -> small green circle
        [player_circles.append(plt.Circle(xy=(0, 0), radius=0.8, color='tomato', zorder=10))
         for _ in range(5)]

        for x in range (4):
            s = 1-(x* 0.25)
            l_s = 0.2 - (x*0.05)
            a = 0.1-(x*0.025)
            for i in range(6):
                if i == 0:
                    line, = self.axes.plot([], [], c='g',linewidth = l_s,zorder=1,solid_capstyle='round',markersize= 0,alpha=a)
                else:
                    line, = self.axes.plot([],[],c='r',zorder=1,solid_capstyle='round',marker='H',markeredgewidth = 0,markersize= 0,alpha=a,
                                                        linewidth =s)
                line_path.append(line)

        # plot
        self.axes.axis('off')
        self.axes.set_xlim(47, 94)

        for circle in player_circles:
            self.axes.add_patch(circle)
        self.axes.add_patch(ball_circle)
        # annotations on circles
        annotations = [self.axes.annotate(name_list[i], xy=[47., 0.],
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold',
                                   fontsize=7, color='white', zorder=11,
                                   path_effects=LABEL_EFFECT)
                       for i in range(5)]
        self.frame_id += 1
        if self.frame_id == length:
            self.frame_id = 0
        self.update_all(self.frame_id,line_path,player_circles, ball_circle, annotations, data)
        self.draw()


    def plot_data2(self,data, length):
        self.axes.cla()
        court = plt.imread(COURT_PNG)
        self.axes.imshow(court, zorder=0, extent=[0, 94, 50, 0])

        name_list = ['PG', 'SG', 'SF', 'PF', 'C',
                     'B1', 'B2', 'B3', 'B4', 'B5']
        player_circles = []
        line_path = []
        line_path2 = []
        line_path3 = []
        # team A -> read circle, team B -> blue circle, ball -> small green circle
        [player_circles.append(plt.Circle(xy=(0, 0), radius=0.8, color='tomato'))
         for _ in range(5)]
        [player_circles.append(plt.Circle(xy=(0, 0), radius=0.8, color='royalblue'))
         for _ in range(5)]
        ball_circle = plt.Circle(xy=(0, 0), radius=0.9, zorder=10, edgecolor=None, facecolor='lime')
        # plot
        self.axes.axis('off')
        self.axes.set_xlim(47, 94)

        for x in range (4):
            s = 1-(x* 0.25)
            l_s = 0.3 - (x*0.05)
            a = 0.1-(x*0.025)
            for i in range(11):
                if i == 0:
                    line, = self.axes.plot([], [], c='g',linewidth = l_s,zorder=1,solid_capstyle='round',markersize= 0,alpha=a)
                elif i < 6:
                    line, = self.axes.plot([],[],c='r',zorder=1,solid_capstyle='round',marker='H',markeredgewidth = 0,markersize= 0,alpha=a,
                                                        linewidth =s)
                else:
                    line, = self.axes.plot([], [], c='b', zorder=1,solid_capstyle='round',marker='H',markeredgewidth = 0,markersize= 0,alpha=a,
                                                        linewidth =s)
                line_path.append(line)


        for circle in player_circles:
            self.axes.add_patch(circle)
        self.axes.add_patch(ball_circle)
        # annotations on circles
        annotations = [self.axes.annotate(name_list[i], xy=[47., 0.],
                                          horizontalalignment='center',
                                          verticalalignment='center', fontweight='bold',
                                          fontsize=7, color='white', zorder=11,
                                          path_effects=LABEL_EFFECT)
                       for i in range(10)]

        self.frame_id += 1
        if self.frame_id == length:
            self.frame_id = 0
        self.update_all2(self.frame_id, line_path,player_circles,ball_circle, annotations, data)
        self.draw()

    def on_start(self):
        pts_path = os.path.join(os.path.dirname(__file__), 'Points', 'points2.npy')
        if not os.path.exists(pts_path):
            print(f"Sketch not found: {pts_path} — draw ball path first")
            return
        data = np.load(pts_path)
        data[:,[0,2,4,6,8,10]] = [x - 2.5 for x in data[:,[0,2,4,6,8,10]]]
        play_len = len(data)
        self.plot_data(data, play_len)

    def on_start_G(self, data, cond=0):
        play = data[0,cond]
        play[:, [0, 2, 4, 6, 8, 10]] = [x - 2.5 for x in play[:, [0, 2, 4, 6, 8, 10]]]
        play_len = len(play)
        self.plot_data2(play, play_len)





