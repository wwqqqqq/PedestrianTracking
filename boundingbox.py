import cv2
import sys
import time
import math

from tf_pose import common
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') 

def recognize(img):
    w = 432
    h = 368
    model = "cmu"
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    '''
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        {18, "Bkg"}
    '''
    if(len(humans) == 0):
        return (0,0,0,0)
    human = humans[0]

    height, width, channels = frame.shape
    #return human.get_upper_body_box(width, height)

    x_min = 1
    y_min = 1
    x_max = 0
    y_max = 0

    print(human.body_parts)

    necessary_parts = [0,9,10,12,13,14,15]

    for parts in necessary_parts:
        if parts not in human.body_parts:
            return (0,0,0,0)
    
    eye_y = (human.body_parts[14].y + human.body_parts[15].y) / 2
    #neck_y = human.body_parts[1].y
    nose_y = (human.body_parts[0].y)
    y_min = eye_y - (nose_y - eye_y) * 3.2
    #y_max = 1 - y_max
    if(human.body_parts[13].y < human.body_parts[10].y):
        y_max = human.body_parts[13].y + 0.32 * (human.body_parts[13].y - human.body_parts[12].y)
    else:
        y_max = human.body_parts[10].y + 0.32 * (human.body_parts[10].y - human.body_parts[9].y)
    #y_min = 1 - y_min
    for k in human.body_parts:
        bodypart = human.body_parts[k]
        x = bodypart.x
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
    x_max = math.ceil(x_max * width)
    x_min = math.floor(x_min * width)
    y_max = math.ceil(y_max * height)
    y_min = math.floor(y_min * height)
    bbox = (x_min, y_min, (x_max - x_min), (y_max - y_min))
    print(bbox)
    print(x_min, x_max, y_min, y_max)
    print("eye",human.body_parts[14].y)
    print("ankle",human.body_parts[10].y)
    print("lWrist",human.body_parts[4].x)
    print("rWrist",human.body_parts[7].x)
    return bbox


if __name__ == '__main__' :
    frame = cv2.imread("./images/test.jpg")
    '''video = cv2.VideoCapture(0)
 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()'''
        
    while True:
        #ok, frame = video.read()
        bbox = recognize(frame)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1)