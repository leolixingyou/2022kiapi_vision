from cmath import exp
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
        self.LiDAR_bbox = None
        self.Camera_60_bbox = None
        self.bbox_60 = []
        self.Camera_190_bbox = None

        self.cam = None
        self.lidar = None
        self.threshold = 0

        rospy.init_node('LiDAR_Cam_fusion')

        common_path = os.getcwd() + '/calibration_data'
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

        self.cur_f190_img = {'img':None, 'header':None}
        self.sub_190_img = {'img':None, 'header':None}
        
        self.get_new_image = False

        self.camera_ob_marker_array = MarkerArray()
        self.pub_camera_ob_marker = rospy.Publisher('/camera_ob_marker', MarkerArray, queue_size=1)
        self.pub_bump = rospy.Publisher('/camera_ob_bump', Float32MultiArray, queue_size=1)
        # self.pub_camera_190_ob_marker = rospy.Publisher('/Camera/Front190/od_bbox', MarkerArray, queue_size=30)
        
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_60_callback)
        # rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_190_callback)
        rospy.Subscriber('/lidar/cluster_box', BoundingBoxArray, self.LiDAR_bboxes_callback)

        ##########################
        self.pub_cam = rospy.Publisher('/cam/result', Image, queue_size=1)
        self.bridge = CvBridge()
        self.height = []
        self.is_save =False
        ##########################

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
            if obj.y< 4 and obj.y >-4 and obj.x>2:
                lidar_temp.append([obj.x,obj.y,obj.z])

        self.LiDAR_bbox = lidar_temp

    def Marker(self,obj_list):
        marker_ob = Marker()

        obj_list = np.array(obj_list).reshape(-1,2)
        for obj in obj_list:
            print('obj is :',obj)
            predict_3d = obj[0]
            label = obj[1]
            if label == 0 :
                ob_id = 1
                color_list = [.0,.0,1.]
            elif label == 2 or label == 7:
                ob_id = 2
                color_list = [.0,1.,.0]
            else:
                ob_id = 0
                color_list = [1.,1.0,1.0]
            
            ##marker
            marker_ob.header.frame_id = 'os_sensor'
            marker_ob.type = marker_ob.SPHERE
            marker_ob.action = marker_ob.ADD
            marker_ob.scale.x = 1.0
            marker_ob.scale.y = 1.0
            marker_ob.scale.z = 1.0
            marker_ob.color.a = 1.0
            marker_ob.color.r = color_list[0]
            marker_ob.color.g = color_list[1]
            marker_ob.color.b = color_list[2]
            marker_ob.id = ob_id
            marker_ob.pose.orientation.w = 1.0

            marker_ob.pose.position.x = predict_3d[0]
            marker_ob.pose.position.y = predict_3d[1]
            marker_ob.pose.position.z = predict_3d[2]

            marker_ob.lifetime = rospy.Duration.from_sec(0.3)
            self.camera_ob_marker_array.markers.append(marker_ob)
        print('marker is : ',self.camera_ob_marker_array)
        self.pub_camera_ob_marker.publish(self.camera_ob_marker_array)

    def LiDAR2Cam(self,LiDAR):
        ### 3d -> 2d : LiDAR-> Cam 
        predict_2d = np.dot(LiDAR, self.calib.homo_lidar2cam)
        predict_2d [:,:2] /= predict_2d[:,2].reshape(predict_2d.shape[0],-1)
        return predict_2d[:,:2]

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

    def strategy(self,lidar): 
        bboxes_60 = self.Camera_60_bbox
        bboxes = np.array(bboxes_60).reshape(-1,5)
        bboxes = bboxes.tolist()
        if lidar != None and lidar != []:
            predict_2d = self.LiDAR2Cam(np.array(lidar)).tolist()
            obj_list =[]
            for box in bboxes:
                flag= True
                print('box is :',box)
                box_mid =[(box[0]+box[2])/2,box[3]]
                while flag:
                    try:
                        if len(bboxes) != 0:
                            dis = [math.sqrt((poi_2d[0]-box_mid[0])**2+(poi_2d[1]-box_mid[1])**2) for poi_2d in predict_2d]
                            min_idx = dis.index(min(dis))  
                            obj_list.append(lidar[min_idx])
                            obj_list.append(box[4])
                            bboxes.remove(box)
                            lidar.remove(lidar[min_idx])
                            flag =False
                    except:
                        flag =False
            for temp_poi in lidar:
                obj_list.append(temp_poi)
                obj_list.append(88)

            self.Marker(obj_list)

    def Visual__bump_jurdge(self,bbox):
        ### for bump the maximum dis is 37,-2.97
        ### for bump the minimum dis is 8
        ### for bump the stable range is 37-8
        height = bbox[3]-bbox[1]
        on_off = Float32MultiArray()
        self.height.append(height)
        ###-52.12547702777454 -6.847777832724959 72102.25440487654
        ### 19.25095528,-0.54844003,-15583.8388745
        ##0.00399,-0.98462,49766/703

        ### 0.0013  x2 - 0.5303x + 52.726 
        ### 0.00399,-0.98462,49766/703

        ### 0.3238x + 45.888
        # weight = [ 0.0013,-0.5303,52.726]
        # weight = [0.00399,-0.98462,49766/703]
        # distance =  weight[0] * height**2 + weight[2]* height  + weight[1]
        # distance =  0.3238 * height  + 45.888
        distance = 67.941*math.exp( 1 )**(-0.017*height)
        # distance = 0.00399 * height**2 -0.98462 * height  + (49766/703)
        # distance = 0.0013  * height**2 -0.5303 * height  + 52.726

        print('height is :' ,height)
        print('distance is :',distance)
        
        if distance > 15:
            distance = distance - 4
        print('height is :' ,height)
        print('distance is :',distance)

        on_off.data.append(distance)
        self.pub_bump.publish(on_off)



    def Visual_car_jurdge(self,bbox):
        height = bbox[3]-bbox[1]
        width = bbox[2]-bbox[0]
        area = height * width
        bbox_mid = (bbox[0] +bbox[2])/2
        # on_off_car = Float32MultiArray()
        self.height.append(height)
        #422.02e-0.071x
        
        ### range 20 - 9
        if  420 < bbox_mid < 840:
            print(2)
            print('height is :' ,height)
            print('width is :',width)
            print('area is :' ,area)
            #39.074e-0.007x
            distance = 39.074*math.exp( 1 )**(-0.007*height)
            ### 0.1232x2 - 13.666x + 324.79
            # distance = 422.02*math.exp( 1 )**(-0.071*height) 
            print('distance is :',distance)


        # on_off.data.append(distance)
        # self.pub_bump.publish(on_off)

    def main(self):
        print('lidar_cam')
        ave_fps = 0.0
        count = 0
        once =True
        flag = True
        while not rospy.is_shutdown():
            state = time.time()
            self.Camera_60_bbox = self.image_process()
            if self.Camera_60_bbox == None or self.Camera_60_bbox == []:
                print('num of boxes :',0)
            else:
                print('num of boxes :',len(self.Camera_60_bbox))
                if once:
                    if flag:
                        for cam_box in self.Camera_60_bbox :
                            if cam_box[4] == 80:
                                self.Visual__bump_jurdge(cam_box)
                                flag = False
                                break
                            elif cam_box[4] == 2:
                                self.Visual_car_jurdge(cam_box)
                                flag = False
                                break
                            else:
                                self.lidar = self.LiDAR_bbox
                                self.strategy(self.lidar)
                                self.camera_ob_marker_array = MarkerArray()
                                try:
                                    if 1./(time.time() - state) <200:
                                        ave_fps += 1./(time.time() - state)
                                        count +=1
                                except:
                                    pass
                            print('fusion() fps :',1./(time.time() - state))
                        break
                else:
                    for cam_box in self.Camera_60_bbox :
                        if cam_box[4] == 80:
                            self.Visual__bump_jurdge(cam_box)
                        elif cam_box[4] == 2:
                            self.Visual_car_jurdge(cam_box)
                        else:
                            self.lidar = self.LiDAR_bbox
                            self.strategy(self.lidar)
                            self.camera_ob_marker_array = MarkerArray()
                            try:
                                if 1./(time.time() - state) <200:
                                    ave_fps += 1./(time.time() - state)
                                    count +=1
                            except:
                                pass
                        print('fusion() fps :',1./(time.time() - state))
        try:
            print(ave_fps/count)    
        except:
            pass
        if self.is_save:
            save_txt = './heights.txt'
            # print([x+'\n' for x in re.split('[ ] ,',str(self.height))])
            for t,x in enumerate(str(set(self.height)).split('{')[1].split('}')[0].split(',')):
                if t == 0:
                    with open(save_txt,'w') as f:
                        f.write(x + '\n')
                else:
                    with open(save_txt,'a') as f:
                        f.write(x + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--weightfile', default=os.getcwd()+"/weights/yolov7-tiny-no-nms.trt")  
    # parser.add_argument('--weightfile', default=os.getcwd()+"/weights/yolov7-no-nms_swlee.trt")  
    # parser.add_argument('--weightfile', default="/home/cvlab/catkin_build_ws/src/yolov7/weights/yolov7-transfer.trt")  
    parser.add_argument('--weightfile', default="/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/weights/yolov7-transfer.trt")  
    # parser.add_argument('--weightfile', default="/home/cvlab-swlee/Desktop/competition/git/2022kiapi_vision/yolov7/weights/yolov7tiny-transfer.trt")  
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

