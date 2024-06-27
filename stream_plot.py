import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from mediapipe_utils import write_pose_video
import threading
import numpy as np
import cv2

plt.style.use('fivethirtyeight')

def animate(i, csv_file, ax1, ax2, ax3, frame_lock, current_frame):
    data = pd.read_csv(csv_file)

    x = np.arange(len(data))
    left_wrist_y = data['15'] # left wrist
    right_wrist_y = data['16'] # right wrist
    left_elbow_y = data['13']
    right_elbow_y = data['14']

    # plt.cla()
    # plt.plot(x, y0, label='y0')
    # plt.plot(x, y1, label='y1')
    # plt.plot(x, y2, label='y2')
    
    ax1.clear()
    ax1.plot(x, left_wrist_y, label='Left wrist', linewidth=1.5)
    ax1.plot(x, right_wrist_y, label='Right wrist', linewidth=1.5)
    ax1.set_title("Wrist Movements", fontsize = 14)
    ax1.legend()
    
    ax2.clear()
    ax2.plot(x, left_elbow_y, label='Left elbow', linewidth=1.5)
    ax2.plot(x, right_elbow_y, label='Right elbow', linewidth=1.5)
    ax1.set_title("Elbow Movements", fontsize = 14)
    ax2.legend()
    
    with frame_lock:
        if current_frame[0] is not None:
            ax3.clear()
            ax3.imshow(cv2.cvtColor(current_frame[0], cv2.COLOR_BGR2RGB))
            ax3.axis('off')
            ax3.set_title("Live Camera", fontsize = 14)
            

class ExitCommand(Exception):
    pass


if __name__ == '__main__':
    VIDEO_FILE = 'DJI_0883.MP4'
    CSV_FILE = 'pose.csv'
    OUT_VIDEO_FILE = 'output_with_landmarks.mp4'

    current_frame = [None]
    frame_lock = threading.Lock()

    t1 = threading.Thread(target=write_pose_video, args=(VIDEO_FILE, CSV_FILE, OUT_VIDEO_FILE, frame_lock, current_frame))
    t1.start()

    fig = plt.figure(figsize=(12, 8))

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ani = FuncAnimation(fig, animate, fargs=(CSV_FILE, ax1, ax2, ax3, frame_lock, current_frame), interval=100)
    plt.tight_layout()
    plt.show()
    #plt.legend()

    t1.join()


