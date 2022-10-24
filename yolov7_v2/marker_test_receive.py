import numpy as np
import cv2

import rospy
from visualization_msgs.msg import Marker, MarkerArray

class Marker:
    def __init__(self):
        self.Marker = None
        self.marker = None

        rospy.init_node('mark_receive')

        rospy.Subscriber('/camera_ob_marker', MarkerArray, self.marker_receiver)

    def marker_receiver(self,msg):
        self.Marker = msg.markers

    def strategy(self,marker):
        if marker != None:
            for obj in marker:
                id_num = obj.id
                x,y,z = obj.pose.position.x,obj.pose.position.y,obj.pose.position.z
                print(id_num)
                print(x,y,z)


    def main(self):
        while not rospy.is_shutdown():
            self.marker = self.Marker
            self.strategy(self.marker)


if __name__ == "__main__":

    Marker = Marker()
    Marker.main()

