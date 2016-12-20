# Author: Zongchang (Jim) Chen
# Junior at Haverford College, Com-Sci & Math
# Date: Dec, 2016

# main.py 
# Program entrance file. Initialize global variables and call main functions.



import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from camera_pose_estimate import visualize3D

# Import all global constants and coefficients
from settings import *
# Import global model figure
import global_figure as gf



# Establish global figure
gf.fig = plt.figure()
gf.ax = gf.fig.add_subplot(111, projection='3d')

# Set axes labels
gf.ax.set_xlabel('X')
gf.ax.set_ylabel('Y')
gf.ax.set_zlabel('Z')


# Edit the codes block below to choose images for cam pose estimation.
# You can pick one file only, or severals at the same time.
# You can also write a loop to handle this.
visualize3D('../images/1.jpg')
visualize3D('../images/2.jpg')
visualize3D('../images/3.jpg')
# visualize3D('../images/4.jpg')
# visualize3D('../images/5.jpg')
# visualize3D('../images/6.jpg')
# visualize3D('../images/7.jpg')
# visualize3D('../images/8.jpg')
# visualize3D('../images/9.jpg')


# Set axes unit length equal, some embellishment
plt.gca().set_aspect('equal', adjustable='box')

# Show the plots
plt.show()



