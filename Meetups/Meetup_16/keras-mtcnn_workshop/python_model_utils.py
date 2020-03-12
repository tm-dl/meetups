#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:27:09 2018

@author: merlin
"""
import numpy as np

def padding(X, pad, padding_value):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (padding_value,padding_value))

    return X_pad


def padding_for_maxpool(X, pad, padding_value):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

   
    X_pad = np.pad(X, ((0,0), (0,pad), (0,pad), (0,0)), 'constant', constant_values = (padding_value,padding_value))
   

    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev,W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z+b

    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    #print("A_prev.shape: ",A_prev.shape)
    
    (f, f, n_C_prev, n_C) = W.shape
    #print("W.shape: ",W.shape)
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    padding_value = hparameters["padding_value"]
 
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev-f+2*pad) / stride+1)
    n_W = int((n_W_prev-f+2*pad) / stride+1) 
    
  
    A_prev_pad = padding(A_prev, pad, padding_value)
    
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m,n_H,n_W,n_C))
    
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:,:]              # Select ith training example's padded activation
        
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" 
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). 
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. 
                    Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:,:,:,c], b[c])
                     
                    
    cache = (A_prev, W, b, hparameters)               
    return Z, cache


def prelu(x, alpha, deriv=False):
    c = np.zeros_like(x).astype('float32')
    
    #slope = 1e-1
    if deriv:
        c[x<=0] = alpha
        c[x>0] = 1
    else:
        c[x>0] = x[x>0]
        c[x<=0] = alpha*x[x<=0]
    return c


def prelu_forward(A_prev, prelu_np_array):
    """
    Implements the forward propagation for a Prelu function: 

    Arguments:
    A_prev -- output of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    prelu_layer -- numpy array of shape (n_C_prev,) with all the alpha params, 1 per channel

    
    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    nb_prelu_inputs = prelu_np_array.shape[3]
    
    
    '''
    A_prev.shape:  (1, 48, 48, 32)
    prelu_np_array.shape:  (1, 1, 1, 32)
    '''
    assert (A_prev.shape[3] == prelu_np_array.shape[3])
    
    # initialize the output with 0s
    out = np.zeros_like(A_prev).astype('float32')
    
    # we have to squeeze the prelu, to get only the values
    prelu_squeeze = np.squeeze(prelu_np_array)
    
    
    for i in range(nb_prelu_inputs):
        channel_slice_A_prev = A_prev[...,i]
        
        out[...,i] = prelu(channel_slice_A_prev, prelu_squeeze[i], deriv=False)

    return out



def prelu_FC_forward(A_prev, prelu_np_array):
    """
    Implements the forward propagation for a Prelu function for a Fully Connected layer: 

    Arguments:
    A_prev -- output of the previous layer, numpy array of shape (m, n_C_prev)
    
    prelu_layer -- numpy array of shape (1, n_C_prev,) with all the alpha params, 1 per channel

    
    """
    
    # Retrieve dimensions from the input shape
    m  = A_prev.shape[1]
    nb_prelu_inputs = prelu_np_array.shape[1]
    
    
    '''
    A_prev.shape:  (None, 256)
    prelu_np_array.shape:  (1, 256) 
    '''
    assert (m == nb_prelu_inputs)
    
    # initialize the output with 0s
    out = np.zeros_like(A_prev).astype('float32')
    
    # we have to squeeze the prelu, to get only the values
    prelu_squeeze = np.squeeze(prelu_np_array)
    
    
    for i in range(nb_prelu_inputs):
        
        out[...,i] = prelu(A_prev[...,i], prelu_squeeze[i], deriv=False)

    return out

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    padding_value = hparameters["padding_value"]
    
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f + 2*pad) / stride)
    n_W = int(1 + (n_W_prev - f + 2*pad) / stride)
    n_C = n_C_prev
    
    #print("n_H: ",n_H)
    #print("n_W: ",n_W)
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))    
    
    # pad or not, dep on pad value
    A_prev_pad = padding_for_maxpool(A_prev, pad, padding_value)
    
   
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)


    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache