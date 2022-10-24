import numpy as np
import cv2

import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from jsk_recognition_msgs.msg import BoundingBoxArray,BoundingBox
from visualization_msgs.msg import Marker, MarkerArray

from calibration import Calibration

class LiDAR_Cam:
    def __init__(self):
        self.LiDAR_bbox = None
        self.Camera_60_bbox = None
        self.bbox_60 = []
        self.Camera_190_bbox = None

        self.cam = None
        self.lidar = None

        rospy.init_node('LiDAR_Cam')
        path = ['./calibration_data/front_60.txt','./calibration_data/camera_lidar.txt']
        self.calib = Calibration(path)

        self.pub_camera_ob_marker = rospy.Publisher('/camera_ob_marker', MarkerArray, queue_size=1)
        self.pub_od = rospy.Publisher('/Camera/Front60/bboxes_info', Float32MultiArray,queue_size=1)
        rospy.Subscriber('/Camera/Front60/od_bbox', Int32MultiArray, self.IMG_60_callback)
        rospy.Subscriber('/Camera/Front190/od_bbox', Float32MultiArray, self.IMG_190_callback)
        rospy.Subscriber('/lidar/cluster_box', BoundingBoxArray, self.LiDAR_bboxes_callback)
        
    def pub_arry_msg(self, bboxes_info):
        pub_box = []
        for bbox in bboxes_info:
            pub_box.extend([bbox[0], bbox[1], bbox[2], bbox[3], bbox[6]])
        return Float32MultiArray(data=pub_box)

    def LiDAR_bboxes_callback(self,msg):
        lidar_temp = []
        for object in msg.boxes:
            obj = object.pose.position
            if obj.y< 3 and obj.y >-3 and obj.x>2:
                lidar_temp.append([obj.x,obj.y,obj.z])

        self.LiDAR_bbox = lidar_temp

    def IMG_60_callback(self, msg):
        print(msg.data)
        self.Camera_60_bbox = msg.data
    
    def IMG_190_callback(self, msg):
        self.Camera_190_bbox = msg.data
    
    def Marker(self,predict_3d,label):
        
        camera_ob_marker_array = MarkerArray()
        marker_ob = Marker()

        ob_id = 0
        ##marker
        marker_ob.header.frame_id = 'os_sensor'
        marker_ob.type = marker_ob.SPHERE
        marker_ob.action = marker_ob.ADD
        marker_ob.scale.x = 1.0
        marker_ob.scale.y = 1.0
        marker_ob.scale.z = 1.0
        marker_ob.color.a = 1.0
        marker_ob.color.r = label/10
        marker_ob.color.g = 1.0
        marker_ob.color.b = 0.0
        marker_ob.id = ob_id
        marker_ob.pose.orientation.w = 1.0
        marker_ob.pose.position.x = predict_3d[0]#########
        marker_ob.pose.position.y = predict_3d[1]
        marker_ob.pose.position.z = predict_3d[2]
        marker_ob.lifetime = rospy.Duration.from_sec(0.3)
        camera_ob_marker_array.markers.append(marker_ob)
        self.pub_camera_ob_marker.publish(camera_ob_marker_array)
        camera_ob_marker_array = MarkerArray()

    def Cam2LiDAR(self,Cam):
        ### 2d -> 3d : Cam -> LiDAR
        predict_3d = np.dot(self.calib.M, Cam)
        return predict_3d

    def LiDAR2Cam(self,LiDAR):
        ### 3d -> 2d : LiDAR-> Cam 
        predict_2d = self.calib.lidar_project_to_image(LiDAR.transpose(), self.calib.proj_lidar2cam)
        return predict_2d
   
    def strategy(self,lidar): 
        
        if self.Camera_60_bbox != None:
            bboxes_60 = list(self.Camera_60_bbox)
            bboxes = np.array(bboxes_60).reshape(-1,5)
            print(bboxes)
            for bbox in bboxes:
                bbox_mid=[[bbox[0] + bbox[2]],[bbox[1]+bbox[3]],[1]]
        
            if lidar != None:
                predict_2d = self.LiDAR2Cam(np.array(lidar)).reshape(-1,2)
                for box in bboxes:
                    for t,poi_2d in enumerate(predict_2d):
                        # if box[0] <= poi_2d[0] <= box[2] and box[1] <= poi_2d[1] <= box[3]:
                        self.Marker(lidar[t],box[4])

    def main(self):
        while not rospy.is_shutdown():
            self.lidar = self.LiDAR_bbox
            self.strategy(self.lidar)

if __name__ == "__main__":

    LiDAR_Cam = LiDAR_Cam()
    LiDAR_Cam.main()

