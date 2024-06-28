import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation 
import pandas as pd
from mediapipe_utils import write_pose_video
import threading
import numpy as np
import cv2

plt.style.use('fivethirtyeight')

def animate(i, csv_file, ax1, ax2, ax3, ax4, frame_lock, current_frame):
    data = pd.read_csv(csv_file)

    x = np.arange(len(data))
    left_wrist_y = data['15'] 
    right_wrist_y = data['16'] 
    left_elbow_y = data['13']
    right_elbow_y = data['14']
    left_hip_y = data['23']
    right_hip_y = data['24']
    
    ax1.clear()
    ax1.plot(x, left_wrist_y, label='Left wrist', linewidth=1.5)
    ax1.plot(x, right_wrist_y, label='Right wrist', linewidth=1.5)
    ax1.set_title("Wrist Movements", fontsize = 14)
    ax1.set_xlabel('Frame Index', fontsize = 10)
    ax1.set_ylabel('Pixel Position', fontsize = 10)
    ax1.legend()
    
    ax2.clear()
    ax2.plot(x, left_elbow_y, label='Left elbow', linewidth=1.5)
    ax2.plot(x, right_elbow_y, label='Right elbow', linewidth=1.5)
    ax2.set_title("Elbow Movements", fontsize = 14)
    ax2.set_xlabel('Frame Index', fontsize = 10)
    ax2.set_ylabel('Pixel Position', fontsize = 10)
    ax2.legend()
    
    ax3.clear()
    ax3.plot(x, left_hip_y, label='Left hip', linewidth=1.5)
    ax3.plot(x, right_hip_y, label='Right hip', linewidth=1.5)
    ax3.set_title("Hip Movements", fontsize=14)
    ax3.set_xlabel('Frame Index', fontsize = 10)
    ax3.set_ylabel('Pixel Position', fontsize = 10)
    ax3.legend()
    
    with frame_lock:
        if current_frame[0] is not None:
            ax4.clear()
            ax4.imshow(cv2.cvtColor(current_frame[0], cv2.COLOR_BGR2RGB))
            ax4.axis('off')
            # ax4.set_title("Live Camera", fontsize = 14)
            

class ExitCommand(Exception):
    pass


if __name__ == '__main__':
    VIDEO_FILE = 'DJI_0886.MP4'
    CSV_FILE = 'pose.csv'
    OUTPUT_VISUALIZATION_FILE = 'visualization_output.mp4'

    current_frame = [None]
    frame_lock = threading.Lock()

    t1 = threading.Thread(target=write_pose_video, args=(VIDEO_FILE, CSV_FILE, OUTPUT_VISUALIZATION_FILE, frame_lock, current_frame))
    t1.start()

    # for smaller video and larger graph
    # fig = plt.figure(figsize=(14, 10))

    # ax1 = plt.subplot2grid((3, 2), (0, 0))
    # ax2 = plt.subplot2grid((3, 2), (0, 1))
    # ax3 = plt.subplot2grid((3, 2), (1, 0))
    # ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    
    fig = plt.figure(figsize=(12, 8))

    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (0, 1))
    ax3 = plt.subplot2grid((3, 3), (0, 2))
    ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2)

    ani = FuncAnimation(fig, animate, fargs=(CSV_FILE, ax1, ax2, ax3, ax4, frame_lock, current_frame), interval=100)
    plt.tight_layout()
    plt.show()

    
    # ani.save(OUT_VIDEO_FILE, writer='ffmpeg', fps=30)
    # plt.close()
    
    ani.save(OUTPUT_VISUALIZATION_FILE, writer='ffmpeg', fps=30) 
    plt.close() 
    
    t1.join()


