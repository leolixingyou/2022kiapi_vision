from asyncio import FastChildWatcher
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
        self.get_new_image = False
        self.Camera_60_bbox = None
        self.bbox_60 = []

        self.cam = None
        self.obj_info = []
        self.marker_info = []
        self.sup = []
        self.count= 0
        self.img_flag = [1,0,0]

        rospy.init_node('Camemra_Node')

        self.args = args
        self.img_shape = image_shape

        self.yolov7 = YOLOV7(args,image_shape)

        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_60_img = {'img':None, 'header':None}

        self.bboxes = Float32MultiArray()
        self.pub_od = rospy.Publisher('/camera_od', Float32MultiArray, queue_size=1)
        
        rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, self.IMG_f60_callback)

       ##########################
        self.pub_cam = rospy.Publisher('/cam/result', Image, queue_size=1)
        self.bridge = CvBridge()
        self.is_save =False
        self.sup = []
        ##########################
        
    def IMG_f60_callback(self,msg):
        if self.img_flag[0] == 1:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            front_img = cv2.resize(front_img, (self.img_shape))
            
            self.cur_f60_img['img'] = front_img
            self.cur_f60_img['header'] = msg.header
            self.get_new_image = True
            # self.img_flag = [0,1,0]

    def IMG_f120_callback(self,msg):
        if self.img_flag[0] == 1:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            front_img = cv2.resize(front_img, (self.img_shape))
            
            self.cur_f60_img['img'] = front_img
            self.cur_f60_img['header'] = msg.header
            self.get_new_image = True
            self.img_flag = [0,0,1]

    def IMG_r120_callback(self,msg):
        if self.img_flag[0] == 1:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            front_img = cv2.resize(front_img, (self.img_shape))
            
            self.cur_f60_img['img'] = front_img
            self.cur_f60_img['header'] = msg.header
            self.get_new_image = True
            self.img_flag = [1,0,0]

    def image_process(self):
        if self.get_new_image:
            self.sub_60_img['img'] = self.cur_f60_img['img']
            orig_im = copy.copy(self.sub_60_img['img']) 
            boxwclass,draw_img = self.yolov7.detect(orig_im,is_save = True)
            
            ###publist

            ######
            print('box is :',boxwclass)
            msg = None
            try:
                self.bboxes.data = boxwclass
                msg = self.bridge.cv2_to_imgmsg(draw_img, "bgr8")
                self.sub_60_img['header'] = msg.header
                msg_boxes = self.bboxes
                self.pub_od.publish(msg_boxes)
                self.bboxes = Float32MultiArray()
                self.pub_cam.publish(msg)
            except CvBridgeError as e:
                print(e)
            ######
            return boxwclass

    def main(self):
        while not rospy.is_shutdown():
                self.Camera_60_bbox = self.image_process()
                if self.Camera_60_bbox == None or self.Camera_60_bbox == []:
                    print('num of boxes :',0)
                else:
                    print('num of boxes :',len(self.Camera_60_bbox))
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--weightfile', default="/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/weights/yolov7-transfer.trt")  
    parser.add_argument('--weightfile', default="/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7_trt/weights/yolov7-transfer-v2.trt")  
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

