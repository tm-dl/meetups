import sys
import tools_matrix as tools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
from save_weights import retrieve_original_weights_as_dict
from python_model import custom_Onet_original


# for saving the weights and biases dictionary
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

Pnet = create_Kao_Pnet(r'12net.h5')
Rnet = create_Kao_Rnet(r'24net.h5')
Onet = create_Kao_Onet(r'48net.h5')  # will not work. caffe and TF incompatible


RUN_VIDEO = False
RUN_webcam = True
RUN_picture = False
use_custom_model = True

# Save the weights for the Onet network, for later use
save_weights_as_dict = False
weights_dict_file = 'ONet_weights_dict.p' 
VERBOSE = False

if(save_weights_as_dict):
    all_weights_original_dict = retrieve_original_weights_as_dict(Onet, VERBOSE)
    with open(weights_dict_file, 'wb') as fp:
        pickle.dump(all_weights_original_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

if(use_custom_model):
    with open(weights_dict_file, 'rb') as fp:
        weights_biases_original_model = pickle.load(fp)
    
def detectFace(img, threshold):

    caffe_img = (img.copy() - 127.5) / 127.5
    origin_h, origin_w, ch = caffe_img.shape
    scales = tools.calculateScales(img)
    
    print("scales: ",scales)
    out = []
    t0 = time.time()
    # del scales[:4]
    
    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)
        scale_img = cv2.resize(caffe_img, (ws, hs))
        input = scale_img.reshape(1, *scale_img.shape)
        #print("DEBUG 00: input.shape: ",input.shape)
        ouput = Pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
        #------------------------------
        # --------- rmaria -----------
        # For every scale, the we have 2 predictions and 4 bbox regressor coordinates
        
        classifier = Pnet.predict(input)[0]                    # shape (1,outshape1, outshape2,2)     where outshape1 = the output of the image after applying KaoPnet forward pass
        bbox_regressor = Pnet.predict(input)[1]                # shape (1, outshape1, outshape2,4)
        #print("DEBUG 0")
        classifier = np.array(classifier)
        bbox_regressor = np.array(bbox_regressor)
        #print("classifier.shape: ",classifier.shape)
        #print("bbox_regressor.shape: ",bbox_regressor.shape)
        #print("------------------------")
        #print("classifier = :",classifier)
        #print("------------------------")
        # ---------------------------
        
        out.append(ouput)
        
        
    
    image_num = len(scales)
    
    rectangles = []
    for i in range(image_num):
        cls_prob = out[i][0][0][:, :,1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
        
        #print("cls_prob.shape: ",cls_prob.shape)
        roi = out[i][1][0]
        out_h, out_w = cls_prob.shape
        
        out_side = max(out_h, out_w)
        # print('calculating img scale #:', i)
        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles, 0.7, 'iou')

    t1 = time.time()
    print ('time for 12 net is: ', t1-t0)

    if len(rectangles) == 0:
        return rectangles

    crop_number = 0
    out = []
    predict_24_batch = []
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (24, 24))
        predict_24_batch.append(scale_img)
        crop_number += 1

    predict_24_batch = np.array(predict_24_batch)

    out = Rnet.predict(predict_24_batch)

    cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
    cls_prob = np.array(cls_prob)  # convert to numpy
    roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
    roi_prob = np.array(roi_prob)
    rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
    t2 = time.time()
    print ('time for 24 net is: ', t2-t1)


    if len(rectangles) == 0:
        return rectangles


    crop_number = 0
    predict_batch = []
    for rectangle in rectangles:
        # print('calculating net 48 crop_number:', crop_number)
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (48, 48))
        predict_batch.append(scale_img)
        crop_number += 1

    predict_batch = np.array(predict_batch)
    
    if(use_custom_model):
        output = custom_Onet_original(weights_biases_original_model, predict_batch)
    else:
        output = Onet.predict(predict_batch)
    
    
    cls_prob = output[0]
    roi_prob = output[1]
    pts_prob = output[2]  # index
    # rectangles = tools.filter_face_48net_newdef(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h,
    #                                             threshold[2])
    rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    t3 = time.time()
    print ('time for 48 net is: ', t3-t2)

    return rectangles


threshold = [0.6,0.6,0.7]

if(RUN_VIDEO):
    video_path = '/home/merlin/learn_opencv/VideoReadWriteDisplay/chaplin.mp4'


    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        # Display the resulting frame
        #cv2.imshow('Frame',frame)
        print("frame.shape: ",frame.shape)
        rectangles = detectFace(frame, threshold)
        draw = frame.copy()
        
        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = frame[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)
    
                for i in range(5, 15, 2):
                    cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
                    
        cv2.imshow("test", draw)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
      else:
         break
        
    cap.release()
    cv2.destroyAllWindows()
    
elif(RUN_picture):
    #while(True):
    img = cv2.imread('1_AndrewNg_0.jpg')
    print("img.shape: ",img.shape)
    rectangles = detectFace(img, threshold)    # threshold = [0.6,0.6,0.7]
    draw = img.copy()

    for rectangle in rectangles:
        if rectangle is not None:
            W = -int(rectangle[0]) + int(rectangle[2])
            H = -int(rectangle[1]) + int(rectangle[3])
            paddingH = 0.01 * W
            paddingW = 0.02 * H
            crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            if crop_img is None:
                continue
            if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                continue
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

            for i in range(5, 15, 2):
                cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
    while(True):
        cv2.imshow("test", draw)
        plt.show()
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            cv2.destroyAllWindows()
            break
            
    # cv2.imwrite('test.jpg', draw)
    
elif(RUN_webcam):
    #cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
     
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        draw = img.copy()
        
       
        
        # run on each frame
        rectangles = detectFace(img, threshold) 
        print("rectangles: ",rectangles)
        
        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0] + int(rectangle[2]))
                H = -int(rectangle[1] + int(rectangle[3]))
                paddingH = 0.001 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3]-paddingH),
                               int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), 
                              (255,0,0), 1)
                
                for i in range(5, 15, 2):
                    cv2.circle(draw, (int(rectangle[i+0]), int(rectangle[i+1])), 2, (0, 255, 0))
                    
        
        cv2.imshow("test",draw)
        plt.show()
            # wait for a key to stop the collection
        key = cv2.waitKey(100)
        #cv2.imshow("preview", img)
        if key == 27: # exit on ESC
            break
        
    vc.release()
    cv2.destroyWindow("test")