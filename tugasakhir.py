import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import zipfile
import serial 
import time 
import cv2 
import RPi.GPIO as GPIO
import operator


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
import numpy 
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# script repurposed from sentdex's edits and TensorFlow's example script. Pretty messy as not all unnecessary
# parts of the original have been removed




# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.



# What model to download.
MODEL_NAME = 'new_graph'  # change to whatever folder has the new graph
# MODEL_FILE = MODEL_NAME + '.tar.gz'   # these lines not needed as we are using our own model
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')  # our labels are in training/object-detection.pbkt

NUM_CLASSES = 4  # we only are using one class at the moment (mask at the time of edit)


# ## Download Model


# opener = urllib.request.URLopener()   # we don't need to download model since we have our own
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
# ArduinoSerial = serial.Serial('com3',9600)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)




# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_gambar'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpeg'.format(i)) for i in range(1, 2)]  # adjust range for # of images in folder

# Size, in inches, of the output images.
# IMAGE_SIZE = (16, 12)
cam = cv2.VideoCapture(0)
IMAGE_SIZE = (12, 8)
# cv2.namedWindow("test")

img_counter = 1
global GPIOpin
pin = 4
GPIOpin = pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIOpin,GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
R_EN = 21
L_EN = 22
RPWM = 23
LPWM = 24
GPIO.setup(R_EN, GPIO.OUT)
GPIO.setup(RPWM, GPIO.OUT)
GPIO.setup(L_EN, GPIO.OUT)
GPIO.setup(LPWM, GPIO.OUT)
GPIO.output(R_EN, True)
GPIO.output(L_EN, True)
r=GPIO.PWM(RPWM,100)
p=GPIO.PWM(LPWM,100)
r.start(0)  
p.start(0) 
LED_PIN1 = 26 #red
LED_PIN2 = 25 #green
LED_PIN3 = 8 #blue
LED_PIN4 = 7 #white
def neutral():
    r.ChangeDutyCycle(0)
def right():
    r.ChangeDutyCycle(10)
   

def left():
    p.ChangeDutyCycle(10)

channel = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN1, GPIO.OUT)
GPIO.setup(LED_PIN2, GPIO.OUT)
GPIO.setup(LED_PIN3, GPIO.OUT)
GPIO.setup(LED_PIN4, GPIO.OUT)
# def motor_on(pin):
#     GPIO.output(pin, GPIO.HIGH)


# def motor_off(pin):
#     GPIO.output(pin, GPIO.LOW)
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        u = 1
        print("Finished Initiation")
        print(GPIOpin)
        i = 0
    
        while True :
            u+=1
            start_time_detectwash = time.time()
            GPIO.output(LED_PIN1, GPIO.LOW)
            GPIO.output(LED_PIN2, GPIO.LOW)
            GPIO.output(LED_PIN3, GPIO.LOW)
            GPIO.output(LED_PIN4, GPIO.LOW)
            # print(GPIO.input(GPIOpin))
            #p.ChangeDutyCycle(0)
            # cam = cv2.VideoCapture(0)
            # print("ulang")
            # print(u)
            ret, frame = cam.read()
            state = GPIO.input(GPIOpin)
            if not ret:
                print("failed to grab frame")
                break 
            # if (state==0):
            #   print("Terdeteksi Benda")

            # if (state==1):
            #   print("tidak terdeteksi")
           
         
            if (u==6):
                u-=5
            # cv2.imshow("test", frame)
       
            if(state == 1):
                left()
                # print("tidak terdeteksi")
        
            elif (state == 0 and u==5):
                # time.sleep(2)
                # SPACE pressed
                GPIO.setup(channel, GPIO.IN) 
                
                # time.sleep(2)
                start_time = time.time()
             
                
                p.ChangeDutyCycle(0)
                time.sleep(4)
                for image_path in TEST_IMAGE_PATHS:

                    img_name = "test_gambar/image{}.jpeg".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    img_counter += 0
                    GPIO.output(LED_PIN1, GPIO.HIGH)
                    GPIO.output(LED_PIN2, GPIO.HIGH)
                    GPIO.output(LED_PIN3, GPIO.HIGH)
                    GPIO.output(LED_PIN4, GPIO.HIGH)
                    # ret, frame = cam.read()
                    # if not ret:
                    #     print("failed to grab frame")
                    #     break 
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    # print([category_index.get(i) for i in classes[0] scores[0,i]])
                    # print("\n")
                    # print(scores)
                    # print("\n")
                    # The following code replaces the 'print ([category_index...' statement
                    threshold = 0.5 
                    objects = []
                    test = []
                    for index, value in enumerate(classes[0]):
                        object_dict = {}
                        test_dict = {}
                        if scores[0, index] > threshold :
                            object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                                                scores[0, index]
                            test_dict['output'] =  category_index.get(value).get('name') #float(scores[0, index])
                            if((category_index.get(value)).get('name') != "piring"):
                                # print((category_index.get(value)).get('name'))
                                # print(float(scores[0, index]))
                                # calculate = test_dict.values()
                                # print(calculate)
                                objects.append(object_dict)
                                keys, values = zip(*test_dict.items())
                                test.append(values)
                                
                                # print(values)
                                # ArduinoSerial.write(str.encode((category_index.get(value)).get('name'))) #send 1
                                # time.sleep(5)
                            
                    
                    # print(test)
                    print(objects)
                    print("%s Detik" % (time.time() - start_time))
                    if (len(test)>0): 
                        calculate=test[0]  
                        print(calculate)
                        kata = " "
                        isikata=(kata.join(calculate))
                        print(isikata)
                        # print(len(isikata))
                        if(isikata=='noda'):
                            # state = GPIO.input(GPIOpin)
                            GPIO.output(LED_PIN1, GPIO.HIGH)
                            GPIO.output(LED_PIN2, GPIO.LOW)
                            GPIO.output(LED_PIN3, GPIO.LOW)
                            GPIO.output(LED_PIN4, GPIO.LOW)
                            print("BERAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATT")
                            GPIO.setup(channel, GPIO.LOW)
                            time.sleep(30)
                            GPIO.setup(channel, GPIO.IN) 
                            # left()
                            # time.sleep(3)
                            u-=5
                            
                        if(isikata=='noda sedang'):
                            # state = GPIO.input(GPIOpin)
                            GPIO.output(LED_PIN1, GPIO.LOW)
                            GPIO.output(LED_PIN2, GPIO.LOW)
                            GPIO.output(LED_PIN3, GPIO.HIGH)
                            GPIO.output(LED_PIN4, GPIO.LOW)
                            print("SEDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANGG")
                            GPIO.setup(channel, GPIO.LOW)
                            time.sleep(25)
                            GPIO.setup(channel, GPIO.IN) 
                            # left()
                            # time.sleep(3)
                            u-=5
                            
                        if(isikata=='noda kecil'):
                            # state = GPIO.input(GPIOpin)
                            GPIO.output(LED_PIN1, GPIO.LOW)
                            GPIO.output(LED_PIN2, GPIO.HIGH)
                            GPIO.output(LED_PIN3, GPIO.LOW)
                            GPIO.output(LED_PIN4, GPIO.LOW)
                            print("KECILLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
                            GPIO.setup(channel, GPIO.LOW)
                            time.sleep(20)
                            GPIO.setup(channel, GPIO.IN) 
                            # left()
                            # time.sleep(3)
                            u-=5
                    
                    else:
                        GPIO.output(LED_PIN1, GPIO.LOW)
                        GPIO.output(LED_PIN2, GPIO.LOW)
                        GPIO.output(LED_PIN3, GPIO.LOW)
                        GPIO.output(LED_PIN4, GPIO.HIGH)
                        state = GPIO.input(GPIOpin)
                        left()
                        time.sleep(2)
                        u-=5
                    #    p.ChangeDutyCycle(0)
                    # time.sleep(5)

            #              if(isikata=='noda'):
            #                 print("BERAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATT")
            #   #              GPIO.setup(channel, GPIO.LOW)
            #    #             time.sleep(10)
            #                 left()
            #                 time.sleep(2)
                          
            #             if(isikata=='noda sedang'):
            #                 print("SEDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANGG")
            #     #            GPIO.setup(channel, GPIO.LOW)
            #      #           time.sleep(5)
            #                 left()
            #                 time.sleep(2)
                         
            #             if(isikata=='noda kecil'):
            #                 print("KECILLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            #       #          GPIO.setup(channel, GPIO.LOW)
            #        #         time.sleep(2)
            #                 left()
            #                 time.sleep(2)
                    
            #         else:
            #             left()
            #             time.sleep(2)
           
                    # os.remove("test_gambar/image1.jpeg")
                        # calculate = test[0].split(": ")
                        # print(calculate)
                    # unique, counts = numpy.unique([category_index.get(i) for i in classes[0]], return_counts=True)
                    # dict(zip(unique, counts))
                    # print("\n")
                    # print(dict(zip(unique, counts)))
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                
                

                 
                    # # print("\n")
                    # # min_score_thresh = 0.5
                    # # print(category_index.get(i) for i in classes[0] if scores[0, i] > min_score_thresh)
                    plt.figure(figsize=IMAGE_SIZE)
                    plt.imshow(image_np)    # matplotlib is configured for command line only so we save the outputs instead
                    plt.savefig("outputs/detection_output{}.png".format(i))  # create an outputs folder for the images to be saved
                    i = i+1  # this was a quick fix for iteration, create a pull request if you'd like
                    print("%s DETIK PENDETEKSIAN TOTAL" % (time.time() - start_time_detectwash))
            
             
        # cam.release()
        
        # cv2.destroyAllWindows()

                



# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("test", frame)

#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         # SPACE pressed
#         img_name = "test_gambar/image{}.jpeg".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         img_counter += 0
# tf.gfile.GFile 

