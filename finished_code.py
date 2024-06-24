import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()


def animate(i):
    left_data = np.load('data/left_arm_angle.npy')
    right_data = np.load('data/right_arm_angle.npy')
    
    x = left_data[0::2]  
    y_left = left_data[1::2] 
    y_right = right_data[1::2]  
    # x = data[0::2]  
    # y1 = data[1::2]  
    #data = pd.read_csv('data.csv')
    # x = data['x_value']
    # y1 = data['total_1']
    #y2 = data['total_2']

    plt.cla()

    plt.plot(x, y_left, label='Left Arm Angle')
    plt.plot(x, y_right, label='Right Arm Angle')
    #plt.plot(x, y1, label='Left Arm Angle')
    #plt.plot(x, y2, label='Right Arm Angle')

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()