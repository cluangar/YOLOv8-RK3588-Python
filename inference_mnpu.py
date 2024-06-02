import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite
from imutils.video import FPS
import time
from lib.postprocess import yolov5_post_process, letterbox_reverse_box, post_process
import lib.config as config
import argparse

from lib.rknnpool import rknnPoolExecutor

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputtype", required=False, default="cam2",
	help="Select input cam(gstreamer), cam2, file(gstreamer), file2")
ap.add_argument("-f", "--filename", required=False, default="skyfall.mp4",
	help="file video (.mp4)")
ap.add_argument("-n", "--npu", required=False, default="3",
	help="Number NPU Threads (default=3")
ap.add_argument("-y", "--yolo", required=False, default="8",
	help="YOLO Version 5 or 8")
args = vars(ap.parse_args())

IMG_SIZE = config.IMG_SIZE

CLASSES = config.CLASSES

# decice tree for rk356x/rk3588
DEVICE_COMPATIBLE_NODE = config.DEVICE_COMPATIBLE_NODE

RK356X_RKNN_MODEL = config.RK356X_RKNN_MODEL
if args["yolo"] == '8':
    RK3588_RKNN_MODEL = config.RK3588_RKNN_MODEL_V8
else:
    RK3588_RKNN_MODEL = config.RK3588_RKNN_MODEL_V5

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK356x'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

#def draw(image, boxes, scores, classes):
def draw(image, boxes, scores, classes, dw, dh):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box

        ##Transform Box to original image
        x1, y1, x2, y2 = letterbox_reverse_box(x1, y1, x2, y2, image.shape[1], image.shape[0], config.IMG_SIZE, config.IMG_SIZE, dw, dh)

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, '{} {:.2f}'.format(CLASSES[cl], score),
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    
    if args["inputtype"] == 'cam':
        gst_str = ("uvch264src device=/dev/video{} ! "
               "image/jpeg, width={}, height={}, framerate=30/1 ! "
               "jpegdec ! "
               "video/x-raw, format=BGR ! "
               "appsink").format(dev, width, height)
        vs = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    elif args["inputtype"] == 'file':
        gst_str = ("filesrc location={} ! "
               "qtdemux name=demux demux. ! queue ! faad ! audioconvert ! audioresample ! autoaudiosink demux. ! "
               "avdec_h264 ! videoscale ! videoconvert ! "
               "appsink").format(args["filename"])		
        vs = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    elif args["inputtype"] == 'cam2':
        vs = cv2.VideoCapture(dev)
        vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    elif args["inputtype"] == 'file2':
        vs = cv2.VideoCapture(args["filename"])


    return vs

##########################
def YoloFunc(rknn_lite, frame):

#Show FPS in Pic
    #new_frame_time = time.time()
    #show_fps = 1/(new_frame_time-prev_frame_time)
    #prev_frame_time = new_frame_time
    #show_fps = int(show_fps)
    #show_fps = str("{} FPS".format(show_fps))

    ori_frame = frame

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame, ratio, (dw, dh) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
#    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Inference
    outputs = rknn_lite.inference(inputs=[frame])


    if args["yolo"] == '8':
        boxes, classes, scores = post_process(outputs)

    else:

        # post process
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))


        boxes, classes, scores = yolov5_post_process(input_data)

#    boxes = None

#    img_1 = frame
#    img_1 = ori_frame
    if boxes is not None:

        # post process

        draw(ori_frame, boxes, scores, classes, dw, dh)


    return ori_frame
##########################


modelPath = RK3588_RKNN_MODEL

#Create Stream from Webcam
vs = open_cam_usb(config.CAM_DEV, config.CAM_WIDTH, config.CAM_HEIGHT)


time.sleep(2.0)
#fps = FPS().start()

if not vs.isOpened():
    print("Cannot capture from camera. Exiting.")
    quit()
	
#prev_frame_time = 0
#new_frame_time = 0



TPEs = int(args["npu"])

pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=YoloFunc)


if (vs.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = vs.read()
        if not ret:
            vs.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()

while (vs.isOpened()):
    frames += 1

    ret, frame = vs.read()
    if not ret:
        break
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break

    
    # show output
    if args["yolo"] == '8':
        cv2.imshow("YOLOv8 post process result", frame)		
    else:
        cv2.imshow("YOLOv5 post process result", frame)
#        cv2.imshow('test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("30 AVR FPS:\t", 30 / (time.time() - loopTime), "Frame")
        loopTime = time.time()

print("Overall AVR FPS\t", frames / (time.time() - initTime))


vs.release()
cv2.destroyAllWindows()
pool.release()





