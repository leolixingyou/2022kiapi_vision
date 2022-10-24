from inspect import classify_class_attrs
import os

from cmath import exp
import os
import time
import math
import numpy as np
import copy
import cv2
import argparse
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

from infer_no_nms import YOLOV7
from calibration import Calibration

###
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
###

with open('./car.txt','r') as f :
    data = f.readlines()

val = {}
for ii  in data:
    content = ii.split(':')
    val[content[0]] = [float(x) for x in content[1].split()]
val["H"] = sorted(val["H"],reverse=True)
fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(val['H'],val['GT'])
ax.plot(val['H'],val['GT'],'o-')
ax.set_xlabel("Heights (pixel)")
ax.set_ylabel("Distance (m)")

plt.show()








