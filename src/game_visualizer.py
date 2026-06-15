import argparse
import numpy as np

import os

import time
import matplotlib
matplotlib.use('agg')  # run backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc


# Note: plot_data() works on raw numpy arrays — no DataFactory needed.
# DataFactory is only used by training callers (train.py) for normalization.


def update_all(frame_id, player_circles, ball_circle, annotations, data):
    """ 
    Inputs
    ------
    frame_id : int
        automatically increased by 1
    player_circles : list of pyplot.Circle
        players' icon
    ball_circle : list of pyplot.Circle
        ball's icon
    annotations : pyplot.axes.annotate
        colors, texts, locations for ball and players
    data : float, shape=[amount, length, 23]
        23 = ball's xyz + 10 players's xy
    """
    max_length = data.shape[1]
    # players
    for j, circle in enumerate(player_circles):
        circle.center = data[frame_id, 2 + j * 2 + 0], data[frame_id, 2 +
                                                            j * 2 + 1]
        annotations[j].set_position(circle.center)
    # print("Frame:", frame_id)
    # ball
    ball_circle.center = data[frame_id, 0], data[frame_id, 1]
    annotations[10].set_position(ball_circle.center)

    return


def plot_data(data, length, file_path=None, if_save=False, fps=6, dpi=128):
    """
    Inputs
    ------
    data : float, shape=[amount, length, 23]
        23 = ball's xyz + 10 players's xy
    length : int
        how long would you like to plot
    file_path : str
        where to save the animation
    if_save : bool, optional
        save as .gif file or not
    fps : int, optional
        frame per second
    dpi : int, optional
        dot per inch
    Return
    ------
    """
    # Resolve court.png relative to this file
    _court_dir = os.path.dirname(os.path.abspath(__file__))
    court = plt.imread(os.path.join(_court_dir, "court.png"))  # 500*939
    name_list = [
        'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', '0'
    ]

    # team A -> red circle, ball -> small green circle
    player_circles = []
    [
        player_circles.append(plt.Circle(xy=(0, 0), radius=2.5, color='r'))
        for _ in range(5)
    ]
    [
        player_circles.append(plt.Circle(xy=(0, 0), radius=2.5, color='b'))
        for _ in range(5)
    ]
    ball_circle = plt.Circle(xy=(0, 0), radius=1.5, color='g')

    # plot
    ax = plt.axes(xlim=(0, 100), ylim=(50, 0))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)

    # annotations on circles
    annotations = [
        ax.annotate(name_list[i],
                    xy=[0., 0.],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='bold') for i in range(11)
    ]
    # animation
    anim = animation.FuncAnimation(fig,
                                   update_all,
                                   fargs=(player_circles, ball_circle,
                                          annotations, data),
                                   frames=length,
                                   interval=100)

    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
    if if_save:
        anim.save(file_path, fps=fps, dpi=dpi, writer='ffmpeg')
        print('!!!Animation is saved!!!')
    else:
        plt.show()
        print('!!!End!!!')

    # clear content
    plt.cla()
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NBA Games visualization')
    parser.add_argument('--save', type=bool, default=True, help='save as mp4')
    parser.add_argument('--amount', type=int, default=10, help='number of plays to plot')
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./samples/')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to .npy data file (shape [N, T, D] where D >= 22)')
    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)

    data = np.load(opt.data_path)
    if data.ndim == 2:
        data = data[None, :, :]  # [T, D] → [1, T, D]

    for i in range(min(opt.amount, len(data))):
        plot_data(data[i:i + 1], length=opt.seq_length,
                  file_path=os.path.join(opt.save_path, f'play_{i}.mp4'),
                  if_save=opt.save)
