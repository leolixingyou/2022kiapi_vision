import os
import time
import math
import numpy as np
import copy
import cv2
import argparse
import re

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from jsk_recognition_msgs.msg import BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray

from infer_no_nms import YOLOV7
from calibration import Calibration

###
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
###

class LiDAR_Cam:
    def __init__(self,args, image_shape):
        self.Camera_60_bbox = None
        self.bbox_60 = []

        self.cam = None
        self.obj_info = []
        self.marker_info = []
        self.sup = []
        self.count= 0
        self.car_count= 0
        self.ped_count= 0

        rospy.init_node('LiDAR_Cam_fusion')

        camera_path = [
                    # '/home/cvlab/catkin_build_ws/src/yolov7/calibration_data/front_60.txt',
                    # '/home/cvlab/catkin_build_ws/src/yolov7/calibration_data/camera_lidar.txt', 
                    # '/home/cvlab/catkin_build_ws/src/yolov7/calibration_data/camera.txt',
                    # '/home/cvlab/catkin_build_ws/src/yolov7/calibration_data/lidar.txt'
                    '/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/calibration_data/front_60.txt',
                    '/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/calibration_data/camera_lidar.txt',
                    '/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/calibration_data/camera.txt',
                    '/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/calibration_data/lidar.txt'
                    ]
        self.calib = Calibration(camera_path)

        self.args = args
        self.img_shape = image_shape

        self.yolov7 = YOLOV7(args,image_shape)

        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_60_img = {'img':None, 'header':None}

        self.get_new_image = False
        self.is_bump = False

        self.camera_ob_marker_array = MarkerArray()
        self.on_off = Float32MultiArray()
        self.pub_camera_ob_marker = rospy.Publisher('/camera_ob_marker', MarkerArray, queue_size=1)
        self.pub_bump = rospy.Publisher('/camera_ob_bump', Float32MultiArray, queue_size=1)
        self.cam = rospy.Publisher('/dectection_test', Float32MultiArray, queue_size=1)
        
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_60_callback)
        rospy.Subscriber('/lidar/cluster_box', BoundingBoxArray, self.LiDAR_bboxes_callback)

       ##########################
        self.pub_cam = rospy.Publisher('/cam/result', Image, queue_size=1)
        self.bridge = CvBridge()
        self.is_save =False
        self.sup = []
        ##########################
        
    def IMG_60_callback(self,msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        front_img = cv2.resize(front_img, (self.img_shape))
        
        self.cur_f60_img['img'] = self.calib.undistort(front_img)
        self.cur_f60_img['header'] = msg.header
        self.get_new_image=True

    def LiDAR_bboxes_callback(self,msg):
        if msg.boxes != []:
            for obj in msg.boxes:
                self.obj_info = obj.pose.position


    def LiDAR2Cam(self,LiDAR):
        ### 3d -> 2d : LiDAR-> Cam 
        predict_2d = np.dot(LiDAR, self.calib.homo_lidar2cam)
        predict_2d [:,:2] /= predict_2d[:,2].reshape(predict_2d.shape[0],-1)
        return predict_2d[:,:2]

    def supervisor(self,obj):
        self.sup.append(obj)
        temp =  np.unique(self.sup, axis=0)
        if len(temp) >3: 
            tm = temp[temp[:, 0].argsort()]
            return tm[0]
        else:
            tm =obj
            return tm
    
    def Makerpub(self,obj_info,idx):
        count_car = 0
        count_ped = 0
        color_list = [1.,1.,1.]
        obj = []
        obj = self.supervisor([round(obj_info.x,2),round(obj_info.y,2)])
        print('obj is :',obj)
        if obj != []:
            flag = True
            if self.count > 50:
                self.count =0
            if idx == 0:
                count_ped +=1
                flag = True
                color_list = [.0,.0,1.]
                self.ped_count
                if count_ped > self.ped_count:
                    color_list = [1.,1.,1.]
            elif idx == 2:
                count_car +=1
                flag = True
                color_list = [.0,1.,.0]
                if count_car > self.ped_count:
                    color_list = [1.,1.,1.]
            if flag: 
                marker_ob = Marker()

                ##marker
                marker_ob.type = marker_ob.SPHERE
                marker_ob.action = marker_ob.ADD
                marker_ob.scale.x = 1.0
                marker_ob.scale.y = 1.0
                marker_ob.scale.z = 1.0
                marker_ob.color.a = 1.0
                marker_ob.color.r = color_list[0]
                marker_ob.color.g = color_list[1]
                marker_ob.color.b = color_list[2]
                marker_ob.pose.orientation.w = 1.0

                marker_ob.pose.position.x = obj[0]
                marker_ob.pose.position.y = obj[1]
                marker_ob.pose.position.z = 0

                marker_ob.lifetime = rospy.Duration.from_sec(0.3)
                marker_ob.header.frame_id = 'os_sensor'
                marker_ob.id = self.count
                self.count +=1
                self.camera_ob_marker_array.markers.append(marker_ob)
                self.pub_camera_ob_marker.publish(self.camera_ob_marker_array)
                self.camera_ob_marker_array = MarkerArray()
                self.sup =[]
                
    def image_process(self):
        if self.get_new_image:
            img_test = Float32MultiArray()
            img_test.data.append(1.)

            self.cam.publish(img_test)

            self.sub_60_img['img'] = self.cur_f60_img['img']
            orig_im = copy.copy(self.sub_60_img['img']) 
            boxwclass,draw_img = self.yolov7.detect(orig_im,is_save = True)
            
            ######
            print('box is :',boxwclass)
            msg = None
            try:
                msg = self.bridge.cv2_to_imgmsg(draw_img, "bgr8")
                self.sub_60_img['header'] = msg.header

            except CvBridgeError as e:
                print(e)
            self.pub_cam.publish(msg)
            ######
            return boxwclass

    def Visual_jurdge(self,cam_box):
        if self.obj_info != []:
            lidar = self.obj_info
            lidar_dist =  math.sqrt((lidar.x)**2+(lidar.y)**2)
            is_marker = False
            for bbox in cam_box:
                if bbox[4] ==80:
                    self.ped_count +=1
                    height = bbox[3]-bbox[1]
                    distance = 67.941*math.exp( 1 )**(-0.017*height)
                    if distance > 15:
                        distance = distance -5
                    # print('height is :' ,height)
                    # print('distance is :',distance)
                    self.on_off.data.append(distance)
                    self.pub_bump.publish(self.on_off)
                    self.is_bump = True

                if bbox[4] ==2:
                    height = bbox[3]-bbox[1]
                    width = bbox[2]-bbox[0]
                    area = height * width
                    bbox_mid = (bbox[0] +bbox[2])/2

                    if  420 < bbox_mid < 840:
                        self.car_count +=1
                        # print(2)
                        # print('height is :' ,height)
                        # print('width is :',width)
                        # print('area is :' ,area)
                        distance = 39.074*math.exp( 1 )**(-0.007*height)
                        # print('distance is :',distance)
                        offset = 1
                        if lidar_dist-offset  < distance <lidar_dist+offset :
                            self.marker_info.append([lidar,2])
                            is_marker = True

            if not is_marker:
                self.marker_info.append([lidar,99])

    def main(self):
        while not rospy.is_shutdown():
            self.Camera_60_bbox = self.image_process()
            if self.Camera_60_bbox == None or self.Camera_60_bbox == []:
                print('num of boxes :',0)
            else:
                print('num of boxes :',len(self.Camera_60_bbox))
                cam_box = self.Camera_60_bbox
                self.Visual_jurdge(cam_box)
                for lidar,idx in self.marker_info:
                    self.count= 0
                    self.car_count= 0
                    self.ped_count= 0
                    self.Makerpub(lidar,idx)

            if not self.is_bump:
                self.on_off.data=[1000.]
                self.pub_bump.publish(self.on_off)
            self.is_bump = False
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--weightfile', default="/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/weights/yolov7-transfer.trt")  
    parser.add_argument('--weightfile', default="/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/weights/yolov7-transfer-v2.trt")  
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

