import cv2
import time
import numpy as np
import math

def distance(point1, point2):
    return math.sqrt( ((point1[0]-point2[0])**2)+((point1[1]-point2[1])**2) )

MODE = "COCO"

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


inWidth = 368
inHeight = 368
threshold = 0.2


input_source = "/Users/kawewutchujit/github.com/contest-cv/Examples/21.mp4"
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

# prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
begin_right = False
start_right = False
end_right = False
count_right = 0
begin_left = False
start_left = False
end_left = False
count_left = 0
while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    hasFrame, frame = cap.read()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    
    # Empty list to store the detected keypoints
    points = []
    p2 = (-1,-1)
    p5 = (-1,-1)
    p3 = (-1,-1)
    p4 = (-1,-1)
    p6 = (-1,-1)
    p7 = (-1,-1)
    # probMap2 = output[0, 2, :, :]
    # minVal2, prob2, minLoc2, point2 = cv2.minMaxLoc(probMap2)
    # if prob2 > threshold : 
    #     p2 = point2
    # else:
    #     pass
    # probMap3 = output[0, 3, :, :]
    # minVal3, prob3, minLoc3, point3 = cv2.minMaxLoc(probMap3)
    # if prob3 > threshold : 
    #     p3 = point3
    # else:
    #     pass
    # probMap4 = output[0, 4, :, :]
    # minVal4, prob4, minLoc4, point4 = cv2.minMaxLoc(probMap4)
    # if prob4 > threshold : 
    #     p4 = point4
    # else:
    #     pass
    # probMap5 = output[0, 5, :, :]
    # minVal5, prob5, minLoc5, point5 = cv2.minMaxLoc(probMap5)
    # if prob5 > threshold : 
    #     p5 = point5
    # else:
    #     pass
    # probMap6 = output[0, 6, :, :]
    # minVal6, prob6, minLoc6, point6 = cv2.minMaxLoc(probMap6)
    # if prob6 > threshold : 
    #     p6 = point6
    # else:
    #     pass
    # probMap7 = output[0, 7, :, :]
    # minVal7, prob7, minLoc7, point7 = cv2.minMaxLoc(probMap7)
    # if prob7 > threshold : 
    #     p7 = point7
    # else:
    #     pass
    
    # print("start_left", start_left)
    # print("end_left", end_left)
    # print("count_left", count_left)

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            # cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            # if(i == 2):
            #     print(x,y)
            #     p2 = (x,y) 
            # elif(i == 3):
            #     print(x,y)
            #     p3 = (x,y)
            # elif(i == 4):
            #     print(x,y)
            #     p4 = (x,y)
            # elif(i == 5):
            #     print(x,y)
            #     p5 = (x,y)
            # elif(i == 6):
            #     print(x,y)
            #     p6 = (x,y)
            # elif(i == 7):
            #     print(x,y)
            #     p7 = (x,y)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    p2 = points[2]
    p3 = points[3]
    p4 = points[4]
    p5 = points[5]
    p6 = points[6]
    p7 = points[7]
    # print(p2,p3,p4,p5)
    if(p2 != None and p3 != None and p4 != None and p5 != None and p6 != None and p7 != None):
        distance_shoulder = distance(p2, p5) 
        distance_right_elbow = distance(p2, p3)
        distance_left_elbow = distance(p5, p6)
        distance_right_arm = distance(p2, p4)
        distance_left_arm = distance(p5, p7)
        distance_right = distance_right_elbow + distance_right_arm
        distance_left = distance_left_elbow + distance_left_arm
        print("distance_shoulder", distance_shoulder)
        print("distance_right", distance_right)
        print("distance_left", distance_left)
        
        if(distance_shoulder < distance_right):
            begin_right = True
        if(distance_shoulder > distance_right and begin_right):
            start_right = True
        if(distance_shoulder < distance_right and start_right):
            end_right = True
        if(begin_right and end_right and start_right):
            count_right += 1
            begin_right = False
            end_right = False
            start_right = False
            begin_left = False
            end_left = False
            start_left = False
        
        if(distance_shoulder < distance_left):
            begin_left = True
        if(distance_shoulder > distance_left and begin_left):
            start_left = True
        if(distance_shoulder < distance_left and start_left):
            end_left = True
        if(start_left and end_left):
            count_left += 1
            begin_right = False
            end_right = False
            start_right = False
            begin_left = False
            end_left = False
            start_left = False
        

        print("begin_right", begin_right)
        print("start_right", start_right)
        print("end_right", end_right)
        print("count_right", count_right)
        print("")
        print("begin_left", begin_left)
        print("start_left", start_left)
        print("end_left", end_left)
        print("count_left", count_left)
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        # print(partA)
        if points[partA] and points[partB]:
            # print(points[partA])
            # print(points[partB])
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    height, width, layer = frame.shape
    resized_height = int(height/3)
    resized_width = int(width/3)
    frame = cv2.resize(frame, (resized_width, resized_height))
    # frameCopy = cv2.resize(frameCopy, (resized_width, resized_height))
    cv2.putText(frame, str(count_right), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, str(count_left), (150, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)

    # vid_writer.write(frame)
# https://drive.google.com/drive/folders/1RY-N84-crH8SrZRHN_5aUxHww3nd5bck?usp=sharing
# vid_writer.release()