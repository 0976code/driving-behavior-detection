from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import pygame
import math
import goto
from dominate.tags import label
from goto import with_goto
import requests




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


t_end = time.time()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[5], mouth[8])
    B = dist.euclidean(mouth[1], mouth[11])	
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C) 



def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret1, frame1 = cap.read()
        
        if not ret1:
            break
        frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame1)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break
        
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break
        
    cap.release()

'''def playmusic():
    clip = mp3play.load("C:/darknet-master/build/darknet/x64/")
    clip.play()
    time.sleep(10)   
    clip.stop()
'''


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors

    ear = 0
    mar = 0
    a=0
    
    label = Queue()
    COUNTER_FRAMES_EYE = 0
    image=Queue()
        
    COUNTER_BLINK = 0
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 7
    MOUTH_AR_THRESH = 0.4
        

    COUNTER_FRAMES_MOUTH = 0
    COUNTER_MOUTH = 0    
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    while cap.isOpened():
        frame1 = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        
        key = cv2.waitKey(1) & 0xFF
        
    
        if frame1 is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame1, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame1, class_colors)
               
            if args.out_filename is not None:
                video.write(image)
            #if cv2.waitKey(fps) == 27:
            if key == ord('q'):
                cv2.destroyAllWindows()
                camera.release()
                break
       
        print("HHHHHHHHHHHHHHHHHHHHHH")   
        
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        

    
#        if ret1 != True:
#            print('read frame failed')
#            continue
        size = frame1.shape
        ret1, image_points1 = get_image_points(frame1)
        if ret1 != 0:
            print('get_image_points failed')
            continue
        ret1, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size1, image_points1)
        if ret1 != True:
            print('get_pose_estimation failed')
            continue
        
        ret1, pitch, yaw, roll = get_euler_angle(rotation_vector)        
        
        
        for rect in rects:
            shape1 = predictor(gray, rect)
            shape1 = face_utils.shape_to_np(shape1)
            leftEye = shape1[lStart:lEnd]
            rightEye = shape1[rStart:rEnd]
            jaw = shape1[48:61]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye) 
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(jaw)

            image_points = np.array([
                                (shape1[30][0], shape1[30][1]),
                                (shape1[8][0], shape1[8][1]),
                                (shape1[36][0], shape1[36][1]),
                                (shape1[45][0], shape1[45][1]),
                                (shape1[48][0], shape1[48][1]),
                                (shape1[54][0], shape1[54][1])
                                ], dtype="double")


            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            if SHOW_POINTS_FACE:
                for p in image_points:
                    cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            
            if SHOW_CONVEX_HULL_FACE: 
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                jawHull = cv2.convexHull(jaw)

            

                cv2.drawContours(image, [leftEyeHull], 0, (255, 255, 255), 1)
                cv2.drawContours(image, [rightEyeHull], 0, (255, 255, 255), 1)
                cv2.drawContours(image, [jawHull], 0, (255, 255, 255), 1)
                cv2.line(image, p1, p2, (255,255,255), 2)


            
            
            if(str(label) == "phone"):
                detect.put('phone')
                label=""
            if (COUNTER_FRAMES_EYE >= EYE_AR_CONSEC_FRAMES*4):
                cv2.putText(image, "Sleeping Driver!", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                detect.put('sleep')
            if(str(label) == "drink"):
                detect.put('drink')
                label=""
            if(str(label) != "seat belt"):
                detect.put('seatbelt')
                label=""
            if yaw>=35 or yaw<=-35:
                cv2.putText(image, "care full!", (200, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                detect.put('eye')

            
            if ear < EYE_AR_THRESH:
                COUNTER_FRAMES_EYE += 1
            else:
                if COUNTER_FRAMES_EYE > 2:
                    COUNTER_BLINK += 1
                COUNTER_FRAMES_EYE = 0
            
            if mar >= MOUTH_AR_THRESH:
                COUNTER_FRAMES_MOUTH += 1
            else:
                if COUNTER_FRAMES_MOUTH > 5:
                    COUNTER_MOUTH += 1
        
                COUNTER_FRAMES_MOUTH = 0
            
            if (time.time() - t_end) > 60:
    #            t_end = time.time()
                COUNTER_BLINK = 0
                COUNTER_MOUTH = 0
                
            
        if SHOW_INFO:
            '''cv2.putText(image, "EAR: {:.2f}".format(ear), (30, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, "MAR: {:.2f}".format(mar), (200, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, "Blinks: {}".format(COUNTER_BLINK), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, "Mouths: {}".format(COUNTER_MOUTH), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)'''
                
        cv2.imshow('test', image)    
                
        key = cv2.waitKey(1) & 0xFF
        print("-----------------------------")


def _largest_face(rects):
    
    if len(rects) == 1:
        return 0
    face_areas = [(det.right()-det.left())*(det.bottom()-det.top()) for det in rects]
    
    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(rects)):
        if face_areas[index] > largest_area :
            largest_index = index
            largest_area = face_areas[index]
    print("largest_face index is {} in {} faces".format(largest_index, len(rects)))
    return largest_index
# 從dlib的檢測結果抽取姿態估計需要的點座標
def get_image_points_from_landmark_shape(landmark_shape):
    #2D image points. If you change the image, you need to change vector
    image_points1 = np.array([
                                (landmark_shape.part(30).x, landmark_shape.part(30).y),     # Nose tip
                                (landmark_shape.part(8).x, landmark_shape.part(8).y),     # Chin
                                (landmark_shape.part(36).x, landmark_shape.part(36).y),     # Left eye left corner
                                (landmark_shape.part(45).x, landmark_shape.part(45).y),     # Right eye right corne
                                (landmark_shape.part(48).x, landmark_shape.part(48).y),     # Left Mouth corner
                                (landmark_shape.part(54).x, landmark_shape.part(54).y)      # Right mouth corner
                                ], dtype="double")
    return 0, image_points1
# 用dlib檢測關鍵點，返回姿態估計需要的幾個點座標
def get_image_points(gray):
#gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 圖片調整爲灰色
    dets = detector( gray, 0 )
    if 0 == len( dets ):
        print( "ERROR: found no face" )
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]
    landmark_shape = predictor(gray, face_rectangle)
    return get_image_points_from_landmark_shape(landmark_shape)
   
    
# 獲取翻轉向量和平移向量                        
def get_pose_estimation(img_size, image_points1 ):
  # Camera internals
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    print("Camera Matrix :{}".format(camera_matrix))
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points1, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )
    print("Rotation Vector:\n {}".format(rotation_vector))
    print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs
# 從翻轉向量轉換爲歐拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
# transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    
    # 單位轉換：將弧度轉換爲度
    Y = int((pitch/math.pi)*180)
    X = int((yaw/math.pi)*180)
    Z = int((roll/math.pi)*180)
    return 0, Y, X, Z


def lineNotifyMessage(token, msg):
    headers = {
        "Authorization": "Bearer " + token, 
        "Content-Type" : "application/x-www-form-urlencoded"
    }

    payload = {'message': msg}
    r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
    return r.status_code
   

def alert(detect,):
    while cap.isOpened():
        detect1 = detect.get()
        if(detect1 == "phone"):
            message = '請別玩手機'
            lineNotifyMessage(token, message)
            pygame.mixer.music.play(1,0.0)
            detect1 = ""
        #    clock = pygame.time.Clock()
        #    clock.tick(2)
        elif(detect1 == "sleep"):
            message = '請不要睡覺'
        #   lineNotifyMessage(token, message)
#            pygame.mixer.music.play(1,0.0)
            detect1 = ""
            #clock = pygame.time.Clock()
            #clock.tick(2)
        elif(detect1 == "drink"):
            message = '請別喝飲料'
        #   lineNotifyMessage(token, message)
            detect1 = ""
            pygame.mixer.music.play(1,0.0)
#            clock = pygame.time.Clock()
        elif(detect1 == "seat belt"):
            message = '請繫安全帶'
#            time.sleep(0)
#            lineNotifyMessage(token, message)
            detect1 = ""
#            pygame.mixer.music.play(1,0.0)
            #clock = pygame.time.Clock()
        #else:
            #pygame.mixer.music.stop()   
        
        elif(detect1 == "eye"):
            message = '請注意視線'
            #lineNotifyMessage(token, message)
#            pygame.mixer.music.play(1,0.0)
            detect1 = ""
            #clock = pygame.time.Clock()
            #clock.tick(2)

        
        if detect.full():
            detect.clear()
        print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"+detect1)
       


    

if __name__ == '__main__':

    ear = 0
    mar = 0

    SHOW_POINTS_FACE = True
    SHOW_CONVEX_HULL_FACE = True
    SHOW_INFO = True

    token = ''

    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    
    detect = Queue()
    
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1)
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    
    cap = cv2.VideoCapture(input_path)
    ret1, frame1 = cap.read()
    size1 = frame1.shape
        
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    
    #pygame.init()
    pygame.mixer.init()
    #track = pygame.mixer.music.load('C:/yolov4/darknet/build/darknet/x64/14849.mp3')
    file = ''
    track = pygame.mixer.music.load(file)
    
    
    model_points = np.array([(0.0, 0.0, 0.0),
                         (0.0, -330.0, -65.0),        
                         (-225.0, 170.0, -135.0),     
                         (225.0, 170.0, -135.0),      
                         (-150.0, -150.0, -125.0),    
                         (150.0, -150.0, -125.0)])

    focal_length = size1[1]
    center = (size1[1]/2, size1[0]/2)

    camera_matrix = np.array([[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")

    dist_coeffs = np.zeros((4,1))
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=alert, args=(detect,)).start()    
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()

