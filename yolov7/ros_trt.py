import os
import cv2
import time
import numpy as np
import argparse
import copy

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray

from calibration import Calibration
from infer_no_nms import YOLOV7

class Detection:
    def __init__(self, args, image_shape):

        rospy.init_node('object_detection_front')
        camera_path = [os.getcwd() + '/calibration_data/front_60.txt',os.getcwd()+'/calibration_data/camera_lidar.txt']
        self.calib = Calibration(camera_path)

        self.args = args
        self.img_shape = image_shape

        self.yolov7 = YOLOV7(args,image_shape)

        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_60_img = {'img':None, 'header':None}

        self.cur_f190_img = {'img':None, 'header':None}
        self.sub_190_img = {'img':None, 'header':None}
        
        self.get_new_image = False

        self.pub_camera_60_ob_marker = rospy.Publisher('/Camera/Front60/od_bbox', Int32MultiArray, queue_size=1)
        # self.pub_camera_190_ob_marker = rospy.Publisher('/Camera/Front190/od_bbox', MarkerArray, queue_size=30)
        
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_60_callback)
        rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_190_callback)

    def IMG_60_callback(self,msg):
        
        img_start = time.time()
        np_arr = np.fromstring(msg.data, np.uint8)
        front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        front_img = cv2.resize(front_img, (self.img_shape))
        img_state1 = time.time()
        
        self.cur_f60_img['img'] = self.calib.undistort(front_img)
        self.cur_f60_img['header'] = msg.header
        print('callback1() time :',round(img_state1 - img_start,5))
        img_state2 = time.time()

        self.get_new_image = True

    def IMG_190_callback(self,msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        front_img = cv2.resize(front_img, (self.img_shape))
        
        self.cur_f190_img['img'] = self.calib.undistort(front_img)
        self.cur_f190_img['header'] = msg.header

    def to_msg(self,arr):
        arr_msg = Int32MultiArray()
        for rr in arr:
           arr_msg.data.extend(rr)
        return arr_msg
    
    def main(self):
        start = time.time()
        try:
            while not rospy.is_shutdown():
                state1 = time.time()
                if self.get_new_image:
                    state2 = time.time()
                    self.sub_60_img['img'] = self.cur_f60_img['img']
                    orig_im = copy.copy(self.sub_60_img['img']) 
                    state3 = time.time()

                    boxwclass,draw_img = self.yolov7.detect(orig_im,is_save = True)
                    state4 = time.time()

                    msg_boxes = []
                    msg_boxes = self.to_msg(boxwclass)
                    state5 = time.time()
    
                    self.pub_camera_60_ob_marker.publish(msg_boxes)

                    # print('start to first loop',round(state1 - start,3))
                    # print('state1',round(state2 - state1,5))
                    # print('state2',round(state3 - state2,5))
                    print('detect() time :',round(state4 - state3,5))
                    # print('state4',round(state5 - state4,5))

        except rospy.ROSInterruptException:
            rospy.logfatal("{object_detection} is dead.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--weightfile', default=os.getcwd()+"/weights/yolov7-tiny-no-nms.trt")  
    # parser.add_argument('--weightfile', default=os.getcwd()+"/weights/yolov7-no-nms_swlee.trt")  
    parser.add_argument('--weightfile', default=os.getcwd()+"/weights/yolov7-transfer.trt")  
    parser.add_argument('--interval', default=1, help="Tracking interval")
    
    parser.add_argument('--namesfile', default="data/coco.names", help="Label name of classes")
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by cla ss: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()

    image_shape=(1280, 720)
    Detection = Detection(args, image_shape)
    Detection.main()
