#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:29:44 2018

@author: merlin
"""

from python_model_utils import padding, padding_for_maxpool, conv_single_step
from python_model_utils import conv_forward, prelu, prelu_forward, prelu_FC_forward
from python_model_utils import pool_forward
import numpy as np

def custom_Onet_original(weights_biases_original_model, input_img):
    
    '''
    Layer 1
    '''
    #lay_1 = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    hparameters = {"pad" : 0, "stride": 1, "padding_value":0}
    W = weights_biases_original_model['conv1'][0]
    b = weights_biases_original_model['conv1'][1]
    
    lay_1_out = conv_forward(input_img, W, b, hparameters)
    # because lay_1_out[1] is the cache for backprop
    lay_1_out_np = np.array(lay_1_out[0])
    print("lay_1_out_np.shape: ",lay_1_out_np.shape)
    
    
    '''
    Layer 2
    '''
    #lay_2 = PReLU(shared_axes=[1,2],name='prelu1')(lay_1)    don't forget Prelu has param too
    prelu1 = weights_biases_original_model['prelu1'] 
    prelu1_np = np.array(prelu1)
    lay_2_out = prelu_forward(lay_1_out_np, prelu1_np)
    lay_2_out_np = np.array(lay_2_out)
    print("lay_2_out_np.shape: ",lay_2_out_np.shape)
    
    '''
    Layer 3
    '''
    # lay_3 = MaxPool2D(pool_size=3, strides=2, padding='same')(lay_2)
    hparameters = {"f":3, "pad" : 1, "stride": 2, "padding_value":0}
    lay_3_out = pool_forward(lay_2_out_np, hparameters, mode = "max")
    # because lay_3_out[1] is the cache
    lay_3_out_np = np.array(lay_3_out[0])
    print("lay_3_out_np.shape: ",lay_3_out_np.shape)
    
    '''
    Layer 4
    '''
    # lay_4 = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(lay_3)
    hparameters = {"pad" : 0, "stride": 1, "padding_value":0}
    W = weights_biases_original_model['conv2'][0]
    b = weights_biases_original_model['conv2'][1]
    
    lay_4_out = conv_forward(lay_3_out_np, W, b, hparameters)
    # because lay_1_out[1] is the cache for backprop
    lay_4_out_np = np.array(lay_4_out[0])
    print("lay_4_out_np.shape: ",lay_4_out_np.shape)
    
    '''
    Layer 5
    '''
    #lay_5 = PReLU(shared_axes=[1,2],name='prelu2')(lay_4)
    prelu2 = weights_biases_original_model['prelu2'] 
    prelu2_np = np.array(prelu2)
    lay_5_out = prelu_forward(lay_4_out_np, prelu2_np)
    lay_5_out_np = np.array(lay_5_out)
    print("lay_5_out_np.shape: ",lay_5_out_np.shape)
    
    '''
    Layer 6
    '''
    #lay_6 = MaxPool2D(pool_size=3, strides=2)(lay_5)
    hparameters = {"f":3, "pad" : 0, "stride": 2, "padding_value":0}
    lay_6_out = pool_forward(lay_5_out_np, hparameters, mode = "max")
    # because lay_3_out[1] is the cache
    lay_6_out_np = np.array(lay_6_out[0])
    print("lay_6_out_np.shape: ",lay_6_out_np.shape)
    
    '''
    Layer 7
    '''
    # lay_7 = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(lay_6)
    hparameters = {"pad" : 0, "stride": 1, "padding_value":0}
    W = weights_biases_original_model['conv3'][0]
    b = weights_biases_original_model['conv3'][1]
    
    lay_7_out = conv_forward(lay_6_out_np, W, b, hparameters)
    # because lay_1_out[1] is the cache for backprop
    lay_7_out_np = np.array(lay_7_out[0])
    print("lay_7_out_np.shape: ",lay_7_out_np.shape)
    
    '''
    Layer 8
    '''
    #lay_8 = PReLU(shared_axes=[1,2],name='prelu3')(lay_7)
    prelu3 = weights_biases_original_model['prelu3'] 
    prelu3_np = np.array(prelu3)
    lay_8_out = prelu_forward(lay_7_out_np, prelu3_np)
    lay_8_out_np = np.array(lay_8_out)
    print("lay_8_out_np.shape: ",lay_8_out_np.shape)
    
    '''
    Layer 9
    '''
    # lay_9 = MaxPool2D(pool_size=2)(lay_8)
        # strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size
    hparameters = {"f":2, "pad" : 0, "stride": 2, "padding_value":0}
    lay_9_out = pool_forward(lay_8_out_np, hparameters, mode = "max")
    lay_9_out_np = np.array(lay_9_out[0])
    print("lay_9_out_np.shape: ",lay_9_out_np.shape)
    
    '''
    Layer 10
    '''
    # lay_10 = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(lay_9)
    hparameters = {"pad" : 0, "stride": 1, "padding_value":0}
    W = weights_biases_original_model['conv4'][0]
    b = weights_biases_original_model['conv4'][1]
    
    lay_10_out = conv_forward(lay_9_out_np, W, b, hparameters)
    lay_10_out_np = np.array(lay_10_out[0])
    print("lay_10_out_np.shape: ",lay_10_out_np.shape)
    
    '''
    Layer 11
    '''
    # lay_11 = PReLU(shared_axes=[1,2],name='prelu4')(lay_10)
    prelu4 = weights_biases_original_model['prelu4'] 
    prelu4_np = np.array(prelu4)
    lay_11_out = prelu_forward(lay_10_out_np, prelu4_np)
    lay_11_out_np = np.array(lay_11_out)
    print("lay_11_out_np.shape: ",lay_11_out_np.shape)
    
    
    '''
    Layer 12
    '''
    # lay_12 = Permute((3,2,1))(lay_11)
    lay_12_out_np = np.transpose(lay_11_out_np, (0,3,2,1))
    print("lay_12_out_np.shape: ",lay_12_out_np.shape)
    
    '''
    Layer 13
    '''
    # lay_13 = Flatten()(lay_12)
    # Only flatten, does not work because it also collapses the first dimension (batch size)
    #lay_13_out_np = lay_12_out_np.flatten()
    # try reshaping
    batch_size, height, width, channels = lay_12_out_np.shape
    #print("batch_size: {}, height: {}, width: {}, channels: {}".format(batch_size, height, width, channels))
    
    product_shape = height*width*channels
    lay_13_out_np = lay_12_out_np.reshape(batch_size, product_shape)
    print("lay_13_out_np.shape: ",lay_13_out_np.shape)
    
    
    '''
    Layer 14
    '''
    # lay_14 = Dense(256, name='conv5') (lay_13)
    W = weights_biases_original_model['conv5'][0]
    b = weights_biases_original_model['conv5'][1]
    lay_14_out_np = np.dot(lay_13_out_np, W) + b
    print("lay_14_out_np.shape: ",lay_14_out_np.shape)
    
    
    '''
    Layer 15
    '''
    #lay_15 = PReLU(name='prelu5')(lay_14)
    prelu5 = weights_biases_original_model['prelu5'] 
    prelu5_np = np.array(prelu5)
    lay_15_out = prelu_FC_forward(lay_14_out_np, prelu5_np)
    lay_15_out_np = np.array(lay_15_out)
    print("lay_15_out_np.shape: ",lay_15_out_np.shape)
    
    
    '''
    classifier
    '''
    # classifier = Dense(2, activation='softmax',name='conv6-1')(lay_15)
    W = weights_biases_original_model['conv6-1'][0]
    b = weights_biases_original_model['conv6-1'][1]
    classifier = np.dot(lay_15_out_np, W) + b
    print("classifier.shape: ",classifier.shape)
    
    '''
    bbox_regress
    '''
    # bbox_regress = Dense(4,name='conv6-2')(lay_15)
    W = weights_biases_original_model['conv6-2'][0]
    b = weights_biases_original_model['conv6-2'][1]
    bbox_regress = np.dot(lay_15_out_np, W) + b
    print("bbox_regress.shape: ",bbox_regress.shape)
    
    
    '''
    landmark_regress
    '''
    # landmark_regress = Dense(10,name='conv6-3')(lay_15)
    W = weights_biases_original_model['conv6-3'][0]
    b = weights_biases_original_model['conv6-3'][1]
    landmark_regress = np.dot(lay_15_out_np, W) + b
    print("landmark_regress.shape: ",landmark_regress.shape)
    
    return classifier, bbox_regress, landmark_regress