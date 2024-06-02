# YOLOv8-RK3588-Python
Modify Code From rknn-toolkit2

# Getting Started
You can change source cam, resolution from capture in config.py

# Prerequisites
1. must compile opencv support gstreamer
2. opencv
3. gstreamer
4. rknnlite2 from rknn-toolkit2
5. usb webcam

# Running the program
python inference_mnpu.py

# This branch
1. fix incorrect High detect box
2. add multithread rknn pool
3. support inference YOLO v5 and v8
