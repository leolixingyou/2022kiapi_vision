import os
import time
import numpy as np
import copy
import cv2
import argparse

import rospy
from sensor_msgs.msg import CompressedImage
from jsk_recognition_msgs.msg import BoundingBoxArray,BoundingBox
from visualization_msgs.msg import Marker, MarkerArray

from infer_no_nms import YOLOV7
from calibration import Calibration

class LiDAR_Cam:
    def __init__(self,args, image_shape):
        self.LiDAR_bbox = None
        self.Camera_60_bbox = None
        self.bbox_60 = []
        self.Camera_190_bbox = None

        self.cam = None
        self.lidar = None

        rospy.init_node('LiDAR_Cam_fusion')
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

        self.pub_camera_ob_marker = rospy.Publisher('/camera_ob_marker', MarkerArray, queue_size=1)
        # self.pub_camera_190_ob_marker = rospy.Publisher('/Camera/Front190/od_bbox', MarkerArray, queue_size=30)
        
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_60_callback)
        # rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_190_callback)
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
        lidar_temp = []
        for object in msg.boxes:
            obj = object.pose.position
            if obj.y< 3 and obj.y >-3 and obj.x>2:
                lidar_temp.append([obj.x,obj.y,obj.z])

        self.LiDAR_bbox = lidar_temp

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

    def LiDAR2Cam(self,LiDAR):
        ### 3d -> 2d : LiDAR-> Cam 
        predict_2d = self.calib.lidar_project_to_image(LiDAR.transpose(), self.calib.proj_lidar2cam)
        return predict_2d

    def image_process(self):
        if self.get_new_image:
            self.sub_60_img['img'] = self.cur_f60_img['img']
            orig_im = copy.copy(self.sub_60_img['img']) 
            state3 = time.time()
            boxwclass,_ = self.yolov7.detect(orig_im,is_save = True)
            state4 = time.time()
            print('detect() time :',round(state4 - state3,5))
            return boxwclass

    def strategy(self,lidar): 
        if self.Camera_60_bbox != None and self.Camera_60_bbox != []:
            bboxes_60 = self.Camera_60_bbox
            bboxes = np.array(bboxes_60).reshape(-1,5)
            print('num of boxes :',len(bboxes))
            if lidar != None:
                predict_2d = self.LiDAR2Cam(np.array(lidar)).reshape(-1,2)
                print('predict_2d')
                print(predict_2d)
                for box in bboxes:
                    for t,poi_2d in enumerate(predict_2d):
                        # if box[0] <= poi_2d[0] <= box[2] and box[1] <= poi_2d[1] <= box[3]:
                        self.Marker(lidar[t],box[4])

    def main(self):
        print('lidar_cam')
        ave_fps = 0.0
        count = 0
        while not rospy.is_shutdown():
            state = time.time()
            self.Camera_60_bbox = self.image_process()
            self.lidar = self.LiDAR_bbox
            self.strategy(self.lidar)
            if 1./(time.time() - state) <200:
                ave_fps += 1./(time.time() - state)
                count +=1
            print('fusion() fps :',1./(time.time() - state))
        print(ave_fps/count)
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

    LiDAR_Cam = LiDAR_Cam(args, image_shape)
    LiDAR_Cam.main()

