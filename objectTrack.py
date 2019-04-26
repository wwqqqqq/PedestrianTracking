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
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
 
    # Read video
    video = cv2.VideoCapture(0)
 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Define an initial bounding box
    # bbox = (287, 23, 86, 320)
 
    # Uncomment the line below to select a different bounding box
    bbox = recognize(frame)
    #bbox = (bbox['x'], bbox['y'], bbox['w'], bbox['h'])
    print(bbox)
    

    #bbox = cv2.selectROI(frame, False)
    #print(bbox)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break