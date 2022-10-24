import numpy as np
import cv2

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray

class Reci:
    def __init__(self):
        self.Dis = None
        self.sid = None

        rospy.init_node('mark_receive')

        rospy.Subscriber('/camera_ob_bump', Float32MultiArray,self.receiver)

    def receiver(self,msg):
        self.Dis = msg.data[0]

    def strategy(self,sid):
        if sid != None and sid != []:
            print(sid)

    def main(self):
        while not rospy.is_shutdown():
            self.sid = self.Dis
            self.strategy(self.sid)


if __name__ == "__main__":

    Reci = Reci()
    Reci.main()

