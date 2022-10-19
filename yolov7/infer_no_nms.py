import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from pathlib import Path
from collections import OrderedDict,namedtuple
import argparse
from utils.general import non_max_suppression

class YOLOV7:
    def __init__(self, args, image_shape):

        self.args = args
        self.img_shape = image_shape
        self.device = torch.device('cuda:0')
        self.yolov7 = self.LoadModel()
        
        self.init_flag = 0

    def LoadModel(self):
        # Infer TensorRT Engine
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(self.args.weightfile, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()


    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def postprocess(self,boxes,r,dwdh):
        dwdh = torch.tensor(dwdh*2).to(boxes.device)
        boxes -= dwdh
        boxes /= r
        return boxes

    def drawer(self,img,boxes,scores,classes,names,ratio,dwdh,is_save = False):
        #colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
        #print(colors)
        colors = {'person':(120,50,10),'car':(12,150,10),'truck':(10,250,10),'traffic sign':(120,150,255)}
        count = 0
        bboxwclass = []
        c = 0

        start = time.time()
        for box,score,cl in zip(boxes,scores,classes):
            box = self.postprocess(box,ratio,dwdh).round().int()
            name = names[int(cl)]
            color = colors[name]
            boxe_fix = [x if x > 0 else 0 for x in box.tolist()]
            bboxwclass.append(boxe_fix)
            bboxwclass[c].append(int(cl.tolist()))
            c+=1
            name += ' ' + str(round(float(score),3))
            cv2.rectangle(img,(boxe_fix[0],boxe_fix[1]),(boxe_fix[2],boxe_fix[3]),color,2)
            cv2.putText(img,name,(int(boxe_fix[0]), int(boxe_fix[1]) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)
        # print("drawer() for loop time : {:.3f}".format(time.time()-start))
        if is_save:
            #print(img.shape)
            count += 1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"./inference/frame {count}.jpg",img)
        return img, bboxwclass

    def detect(self,img,is_save=False):
        names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
                 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                 'hair drier', 'toothbrush','traffic sign']

        start = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_img = img
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im = torch.from_numpy(im).to(self.device)
        im/=255
        print(im.shape)

        # warmup for 10 times
        if self.init_flag == 0:
            for _ in range(10):
                tmp = torch.randn(1,3,640,640).to(self.device)
                self.binding_addrs['images'] = int(tmp.data_ptr())
                self.context.execute_v2(list(self.binding_addrs.values()))
                self.init_flag = 1

        cost_start = time.time()  
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        cost_end = time.time()
        print("\nCost time : {:.3f}sec".format(cost_end - cost_start))

        output = self.bindings['output'].data
        
        start = time.time()

        pred = non_max_suppression(output, self.args.conf_thres, self.args.iou_thres, classes=self.args.classes, agnostic=self.args.agnostic_nms)

        boxes = []
        scores = []
        classes = []

        # height, width = dwdh
        for i in range(len(pred[0])):
            boxes.append(pred[0][i][:4])
            scores.append(pred[0][i][4])
            classes.append(pred[0][i][5])
        draw_img,bboxwclass = self.drawer(img,boxes,scores,classes,names,ratio,dwdh,is_save)
        # print("test time {:.3f}sec".format(time.time() - start))

        return bboxwclass,draw_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weightfile', default="./weights/yolov7-tiny-no-nms.trt")  
    parser.add_argument('--namesfile', default="data/coco.names", help="Label name of classes")

    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()

    image_shape=(1280, 720)
    img = cv2.imread('inference/images/bus.jpg')

    Detection = YOLOV7(args, image_shape)
    boexs,classes = Detection.detect(img,is_save=True)

