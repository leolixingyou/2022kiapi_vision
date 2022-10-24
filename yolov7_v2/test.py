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


class LiDAR_Cam:
    def __init__(self,args, image_shape):
        self.LiDAR_bbox = None
        self.Camera_60_bbox = None
        self.bbox_60 = []
        self.Camera_190_bbox = None

        self.cam = None

        rospy.init_node('LiDAR_Cam_fusion')

        camera_path = [
                    '/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/calibration_data/front_60.txt',
                    '/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/calibration_data/camera_lidar_5.txt',
                    '/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/calibration_data/camera.txt',
                    '/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/calibration_data/lidar.txt'
                    ]
        self.calib = Calibration(camera_path)

        self.args = args
        self.img_shape = image_shape

        self.yolov7 = YOLOV7(args,image_shape)

        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_60_img = {'img':None, 'header':None}

        self.cur_f190_img = {'img':None, 'header':None}
        self.sub_190_img = {'img':None, 'header':None}
        
        self.get_new_image = False

        self.pub_bump = rospy.Publisher('/camera_ob_bump', Float32MultiArray, queue_size=1)
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_60_callback)
        rospy.Subscriber('/os_cloud_node/points', PointCloud2, self.lidar_callback)
        self.xyz = None
        self.poi_tar_bump = None
        self.poi_tar_car = []
        self.car_height = []

        ##########################
        self.pub_pro = rospy.Publisher('/cam/color', Image, queue_size=1)
        self.pub_cam = rospy.Publisher('/cam/result', Image, queue_size=1)
        self.bridge = CvBridge()
        self.height = []
        self.is_save =False
        ##########################

    def IMG_60_callback(self,msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        front_img = cv2.resize(front_img, (self.img_shape))
        
        self.cur_f60_img['img'] = self.calib.undistort(front_img)
        self.cur_f60_img['header'] = msg.header
        self.get_new_image=True

    def lidar_callback(self,msg):
        points = []
        for point in pc2.read_points(msg, skip_nans=True):
            temp = [round(point[0],3),round(point[1],3),round(point[2],3)]
            points.append(temp)
        if points != []:
            self.xyz = points

    def image_process(self):
        if self.get_new_image:
            self.sub_60_img['img'] = self.cur_f60_img['img']
            orig_im = copy.copy(self.sub_60_img['img']) 
            state3 = time.time()
            boxwclass,draw_img = self.yolov7.detect(orig_im,is_save = True)
            state4 = time.time()

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
            
            print('detect() time :',round(state4 - state3,5))
            return boxwclass

    def Visual_jurdge(self,bbox):
        if bbox[4] == 80:
            print('aass')
        elif bbox[4] == 2:
            height = bbox[3]-bbox[1]
            width = bbox[2]-bbox[0]
            area = height * width
            bbox_mid = (bbox[0] +bbox[2])/2
            
            if 11058 < area < 56832 and 320 < bbox_mid < 640 :
                self.car_height.append(height)

    def define_area(self,point1, point2, point3):
        """
        法向量    ：n={A,B,C}
        空间上某点：p={x0,y0,z0}
        点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)
        https://wenku.baidu.com/view/12b44129af45b307e87197e1.html
        :param point1:
        :param point2:
        :param point3:
        :param point4:
        :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        point3 = np.asarray(point3)
        AB = np.asmatrix(point2 - point1)
        AC = np.asmatrix(point3 - point1)
        N = np.cross(AB, AC)  # 向量叉乘，求法向量
        # Ax+By+Cz
        Ax = N[0, 0]
        By = N[0, 1]
        Cz = N[0, 2]
        D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
        return Ax, By, Cz, D

    #### 7.3986,-3.0141,0.0459
    def point2area_distance(self,point1, point2, point3, point4):
        Ax, By, Cz, D = self.define_area(point1, point2, point3)
        mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
        mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
        d = abs(mod_d) / mod_area
        return d

    def lidar_pro(self,class_id):
        temp_bump = []
        temp_car = []
        if class_id == 80:
            for poi in self.xyz:
                poi_1=[7.863,-0.952,-2.25]
                #right
                poi_2=[7.358,-2.99,0.34]
                poi_3=[7.02,-3.32,-0.287]
                
                ##left
                poi_4=[7.197,-2.53,-0.235]
                poi_5=[7.358,-2.99,0.34]

                ##bottom
                poi_6=[7.02,-3.32,-0.287]
                poi_7=[7.197,-2.53,-0.235]

                # poi_test_1 = [7.3986,-3.0141,0.0459]
                # poi_test_2 = [7.3986,-3.0141,0.0459]
                base_poi_1 = self.point2area_distance(poi_1,poi_2,poi_3,poi)
                base_poi_2 = self.point2area_distance(poi_1,poi_4,poi_5,poi)
                base_poi_3 = self.point2area_distance(poi_1,poi_6,poi_7,poi)
                
                if base_poi_1 <1 and base_poi_2 <1 and base_poi_3 <1:
                    dis = math.sqrt((poi[0])**2+(poi[1])**2)
                    temp_bump.append(dis)
            if temp_bump != []:
                self.poi_tar_bump = sum(temp_bump)/len(temp_bump)  

        elif class_id == 2:
            for poi in self.xyz:
                if -.5< poi[1] < .5 and -1.0 < poi[2] <-0.2 and poi[0]>5:
                    dis = math.sqrt((poi[0])**2+(poi[1])**2)
                    temp_car.append(dis)
            if temp_car != [] and temp_car != None:
                self.poi_tar_car.append(round(sum(temp_car)/len(temp_car),3))

    def LiDAR2Cam(self,LiDAR):
        ### 3d -> 2d : LiDAR-> Cam 
        predict_homo_2d = np.dot(LiDAR, self.calib.homo_lidar2cam)
        predict_proj_2d = np.dot(LiDAR, self.calib.proj_lidar2cam)
        predict_homo_2d [:,:2] /= predict_homo_2d[:,2].reshape(predict_homo_2d.shape[0],-1)
        predict_proj_2d [:,:2] /= predict_proj_2d[:,2].reshape(predict_proj_2d.shape[0],-1)
        return predict_proj_2d[:,:2]

    def projection_2d(self):
        if self.get_new_image and self.xyz != None and self.xyz != []:
            img = self.cur_f60_img['img']
            pts_lidar = np.asarray(self.xyz)
            # apply projection
            pts_proj_2d= np.array(self.LiDAR2Cam(pts_lidar)).reshape(2,-1)
            img_height, img_width= [1080,1920]
            # Filter lidar points to be within image FOV
            inds = np.where((pts_proj_2d[0, :] < img_width) & (pts_proj_2d[0, :] >= 0) &
                            (pts_proj_2d[1, :] < img_height) & (pts_proj_2d[1, :] >= 0) &
                            (pts_lidar[:, 0] > 1) & (pts_lidar[:, 2] < 15))[0]

            # Filter out pixels points
            imgfov_pc_pixel = pts_proj_2d[:, inds]

            # Retrieve depth from lidar
            imgfov_pc_lidar = pts_lidar[inds, :]
            imgfov_pc_lidar = np.hstack((imgfov_pc_lidar, np.ones((imgfov_pc_lidar.shape[0], 1))))
            imgfov_pc_cam2 = np.dot(self.calib.proj_lidar2cam, imgfov_pc_lidar.transpose())

            cmap = plt.cm.get_cmap('hsv', 256)
            cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

            for i in range(imgfov_pc_pixel.shape[1]):
                depth = imgfov_pc_cam2[2, i]
                
                color = cmap[min(int(1280.0 / depth), 255), :]
                cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                                    int(np.round(imgfov_pc_pixel[1, i]))),
                            1, color=tuple(color), thickness=3)
            msg = None
            try:
                msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
                self.sub_60_img['header'] = msg.header

            except CvBridgeError as e:
                print(e)
            self.pub_pro.publish(msg)

    def main(self):
        is_write = False
        while not rospy.is_shutdown():
            self.Camera_60_bbox = self.image_process()
            if self.Camera_60_bbox == None or self.Camera_60_bbox == []:
                print('num of boxes :',0)
            else:
                # self.projection_2d()
                print('num of boxes :',len(self.Camera_60_bbox))
                for cam_box in self.Camera_60_bbox :
                    # self.lidar_pro(cam_box[4])
                    # if self.poi_tar_car != [] and self.poi_tar_car != None :
                    #     self.Visual_jurdge(cam_box)
                    self.Visual_jurdge(cam_box)
                    if self.car_height != [] and self.car_height != None :
                        self.lidar_pro(cam_box[4])
        gt_car = sorted(set(self.poi_tar_car))
        hi_car = sorted(set(self.car_height))
        print('poi_tar_car is :',gt_car,'num is :',len(gt_car))
        print('car_height is :',hi_car,'num is :',len(hi_car))

        if self.sssis_write:
            with open('./car.txt','w') as f:
                for i,ii in enumerate(gt_car):
                    if i==0:
                        f.write('GT:' + str(ii)+' ')
                    elif i < len(gt_car)-1:
                        f.write(str(ii)+' ')
                    else:
                        f.write(str(ii)+'\n')
                for o,oo in enumerate(hi_car):
                    if o==0:
                        f.write('H:' + str(oo)+' ')
                    elif o < len(hi_car )-1:
                        f.write(str(oo)+' ')
                    else:
                        f.write(str(oo)+'\n')


            # with open('./car_gt.txt','w')as f:

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weightfile', default="/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/weights/yolov7-transfer-v2.trt")  
    parser.add_argument('--interval', default=1, help="Tracking interval")
    
    parser.add_argument('--namesfile', default="data/coco.names", help="Label name of classes")
    parser.add_argument('--conf-thres', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by cla ss: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()
    image_shape=(1280, 720)

    LiDAR_Cam = LiDAR_Cam(args, image_shape)
    LiDAR_Cam.main()