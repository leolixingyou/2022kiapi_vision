import os
import time
import math
import numpy as np
import copy
import cv2
import argparse

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from jsk_recognition_msgs.msg import BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray

from infer_no_nms import YOLOV7
from calibration import Calibration

class LiDAR_Cam:
    def __init__(self,args, image_shape):
        self.LiDAR_bbox = None
        self.Camera_60_bbox = None
        self.bbox_60 = []

        self.cam = None
        self.threshold = 0

        rospy.init_node('LiDAR_Cam_fusion')

        common_path = os.getcwd() + '/calibration_data'
        camera_path = [
                    '/home/cvlab/catkin_build_ws/src/yolov7/calibration_data/front_60.txt',
                    '/home/cvlab/catkin_build_ws/src/yolov7/calibration_data/camera_lidar.txt', 
                    '/home/cvlab/catkin_build_ws/src/yolov7/calibration_data/camera.txt',
                    '/home/cvlab/catkin_build_ws/src/yolov7/calibration_data/lidar.txt'
                    ]
        self.calib = Calibration(camera_path)

        self.args = args
        self.img_shape = image_shape

        self.yolov7 = YOLOV7(args,image_shape)

        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_60_img = {'img':None, 'header':None}

        self.get_new_image = False

        self.camera_ob_marker_array = MarkerArray()
        self.pub_camera_ob_marker = rospy.Publisher('/camera_ob_marker', MarkerArray, queue_size=1)
        self.pub_bump = rospy.Publisher('/camera_ob_bump', Float32MultiArray, queue_size=1)
        self.cam = rospy.Publisher('/dectection_test', Float32MultiArray, queue_size=1)
        
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_60_callback)
        rospy.Subscriber('/lidar/cluster_box', BoundingBoxArray, self.LiDAR_bboxes_callback)



    def IMG_60_callback(self,msg):
        img_start = time.time()
        np_arr = np.fromstring(msg.data, np.uint8)
        front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        front_img = cv2.resize(front_img, (self.img_shape))
        img_state1 = time.time()
        
        self.cur_f60_img['img'] = self.calib.undistort(front_img)
        self.cur_f60_img['header'] = msg.header
        img_state2 = time.time()
        self.get_new_image=True

    def LiDAR_bboxes_callback(self,msg):
        count = 0
        lidar_temp = []
        if msg.boxes != []:
            for obj in msg.boxes:
                obj_info = obj.pose.position
                
                marker_ob = Marker()
                ##marker
                marker_ob.type = marker_ob.SPHERE
                marker_ob.action = marker_ob.ADD
                marker_ob.scale.x = 1.0
                marker_ob.scale.y = 1.0
                marker_ob.scale.z = 1.0
                marker_ob.color.a = 1.0
                marker_ob.color.r = 1.0
                marker_ob.color.g = 1.0
                marker_ob.color.b = 1.0
                marker_ob.pose.orientation.w = 1.0

                marker_ob.pose.position.x = obj_info.x
                marker_ob.pose.position.y = obj_info.y
                marker_ob.pose.position.z = obj_info.z

                marker_ob.lifetime = rospy.Duration.from_sec(0.3)
                marker_ob.header.frame_id = 'os_sensor'
                marker_ob.id = count
                count +=1
                self.camera_ob_marker_array.markers.append(marker_ob)
                print(self.camera_ob_marker_array)
                self.pub_camera_ob_marker.publish(self.camera_ob_marker_array)
                self.camera_ob_marker_array = MarkerArray()

    def Marker(self,obj_list):
    

        marker_ob.lifetime = rospy.Duration.from_sec(0.3)
        self.camera_ob_marker_array.markers.append(marker_ob)
        count +=1

    def LiDAR2Cam(self,LiDAR):
        ### 3d -> 2d : LiDAR-> Cam 
        predict_2d = np.dot(LiDAR, self.calib.homo_lidar2cam)
        predict_2d [:,:2] /= predict_2d[:,2].reshape(predict_2d.shape[0],-1)
        return predict_2d[:,:2]

    def image_process(self):
        if self.get_new_image:
            img_test = Float32MultiArray()
            img_test.data.append(1.)

            self.cam.publish(img_test)

            self.sub_60_img['img'] = self.cur_f60_img['img']
            orig_im = copy.copy(self.sub_60_img['img']) 
            state3 = time.time()
            boxwclass,_ = self.yolov7.detect(orig_im,is_save = True)
            state4 = time.time()
            print('detect() time :',round(state4 - state3,5))
            return boxwclass

    def Visual_jurdge(self,bbox):
        if bbox[4] ==80:
            height = bbox[3]-bbox[1]
            on_off = Float32MultiArray()
            distance = 67.941*math.exp( 1 )**(-0.017*height)
            if distance > 15:
                distance = distance
            print('height is :' ,height)
            print('distance is :',distance)
            on_off.data.append(distance)
            on_off.data.append(bbox[4])
            self.pub_bump.publish(on_off)
        
        if bbox[4] ==2:
            height = bbox[3]-bbox[1]
            width = bbox[2]-bbox[0]
            area = height * width
            bbox_mid = (bbox[0] +bbox[2])/2
            self.height.append(height)

            if  420 < bbox_mid < 840:
                print(2)
                print('height is :' ,height)
                print('width is :',width)
                print('area is :' ,area)
                distance = 39.074*math.exp( 1 )**(-0.007*height)
                print('distance is :',distance)

    def main(self):
        ave_fps = 0.0
        count = 0
        while not rospy.is_shutdown():
            state = time.time()
            lidar = self.LiDAR_bbox
            if lidar == None and lidar == []:
                print(lidar)
                self.Camera_60_bbox = self.image_process()
                if self.Camera_60_bbox == None or self.Camera_60_bbox == []:
                    print('num of boxes :',0)
                else:
                    print('num of boxes :',len(self.Camera_60_bbox))
                    cam_box = self.Camera_60_bbox
                    # self.Visual_jurdge(cam_box)
                    self.Marker(lidar)
                try:
                    if 1./(time.time() - state) <200:
                        ave_fps += 1./(time.time() - state)
                        count +=1
                except:
                    pass
                print('fusion() fps :',1./(time.time() - state))
        print(ave_fps/count)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weightfile', default="/home/cvlab/catkin_build_ws/src/yolov7/weights/yolov7-transfer.trt")
    # parser.add_argument('--weightfile', default="/home/cvlab/catkin_build_ws/src/yolov7/weights/yolov7tiny-transfer.trt")
    # parser.add_argument('--weightfile', default="/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/weights/yolov7-transfer.trt")  
    parser.add_argument('--interval', default=1, help="Tracking interval")
    
    parser.add_argument('--namesfile', default="data/coco.names", help="Label name of classes")
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by cla ss: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()
    image_shape=(1280, 720)

    LiDAR_Cam = LiDAR_Cam(args, image_shape)
    LiDAR_Cam.main()