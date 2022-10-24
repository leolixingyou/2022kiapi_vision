import os
import cv2
import numpy as np

class Calibration:
    def __init__(self, path):
        # camera parameters
        path1,path2,path3,path4 = path
        cam_param = []
        with open(path1, 'r') as f:
            for i in f.readlines():
                for val in i.split(','):
                    cam_param.append(float(val))

        self.camera_matrix = np.array([[cam_param[0], cam_param[1], cam_param[2]], 
                                       [cam_param[3], cam_param[4], cam_param[5]], 
                                       [cam_param[6], cam_param[7], cam_param[8]]])
        self.dist_coeffs = np.array([[cam_param[9]], [cam_param[10]], [cam_param[11]], [cam_param[12]]])

        # calibration parameters
        calib_param = []
        with open(path2, 'r') as f:
            for line in f.readlines():
                calib_param.extend([float(i) for i in line.split(',')])

        RPT = np.array([[calib_param[0], calib_param[1], calib_param[2], calib_param[9]],
                        [calib_param[3], calib_param[4], calib_param[5], calib_param[10]],
                        [calib_param[6], calib_param[7], calib_param[8], calib_param[11]]])

        self.proj_lidar2cam = np.dot(self.camera_matrix, RPT)

        src_pt = []###camera points
        with open(path3,'r') as f:
            for ii in f.readlines():
                src_pt.append([int(x) for x in ii.split(',')])

        dst_pt = []###lidar points
        with open(path4,'r') as f:
            for oo in f.readlines():
                dst_pt.append([float(x) for x in oo.split(',')[:2]])

        ### [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        print(src_pt)
        print(dst_pt)
        src_pt = np.float32(src_pt)
        dst_pt = np.float32(dst_pt)
        self.homo_lidar2cam = cv2.getPerspectiveTransform(dst_pt,src_pt)


    def lidar_project_to_image(self, points, proj_mat):
        num_pts = points.shape[1]
        points = np.vstack((points, np.ones((1, num_pts))))
        points = np.dot(proj_mat, points)
        points[:2, :] /= points[2, :]
        return points[:2, :]

    def undistort(self, img):
        w,h = (img.shape[1], img.shape[0])
        ### The only one camera which is needed is FOV 190 camera
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 0)
        # result_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        result_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

        return result_img

    def topview(self, img):
        topveiw_img = cv2.warpPerspective(img, self.M, (self.grid_size[0], self.grid_size[1]))
        return topveiw_img 


if __name__ == "__main__":

    common_path = os.getcwd() + '/yolov7/calibration_data'
    camera_path = [common_path + '/front_60.txt',common_path +'/camera_lidar.txt',common_path + '/camera.txt',common_path + '/lidar.txt']
    LiDAR_Cam = Calibration(camera_path)