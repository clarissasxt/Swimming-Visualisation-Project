import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from mediapipe_utils import write_pose_video
import threading
import numpy as np


plt.style.use('fivethirtyeight')

def animate(i, csv_file):
    data = pd.read_csv(csv_file)

    x = np.arange(len(data))
    y0 = data['0']
    y1 = data['1']
    y2 = data['2']


    plt.cla()
    plt.plot(x, y0, label='y0')
    plt.plot(x, y1, label='y1')
    plt.plot(x, y2, label='y2')

class ExitCommand(Exception):

    pass


if __name__ == '__main__':
    VIDEO_FILE = 'DJI_0883.MP4'
    CSV_FILE = 'pose.csv'


    t1 = threading.Thread(target=write_pose_video, args=(VIDEO_FILE, CSV_FILE))
    t1.start()

    ani = FuncAnimation(plt.gcf(), animate, fargs=(CSV_FILE,), interval=1000)
    plt.tight_layout()
    plt.show()
    plt.legend()

    t1.join()


