import os
from cv2 import threshold
import numpy as np
import rospy
from std_msgs.msg import Int32MultiArray

class Visual_jurdge:
    def __init__(self):
        self.threshold = 0

        rospy.init_node('Visual_jurdge')
        self.pub_bump = rospy.Publisher('/camera_ob_bump', Int32MultiArray, queue_size=1)
        
    def main(self,bbox):
        while not rospy.is_shutdown():
            height = self.bboxes[1]-self.bboxes[3]
            on_off = Int32MultiArray()
            print(height)
            if self.threshold < height:
                on_off.data.append(1)
            else:
                on_off.data.append(0)
            self.pub_bump.publish(on_off)

if __name__ == "__main__":
    Visual_jurdge = Visual_jurdge()
    Visual_jurdge.main([0,5,0,0,0])

