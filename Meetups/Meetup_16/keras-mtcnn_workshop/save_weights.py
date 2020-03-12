#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:17:15 2018

@author: merlin
"""
import numpy as np

def retrieve_original_weights_as_dict(my_original_model, VERBOSE):

    all_weights_original_dict = {}
    
    for idx,layer in enumerate(my_original_model.layers):
        if(VERBOSE):
            print("idx = ",idx)

        layer_name = layer.name
        layer_prefix = layer_name[:4]
        
        if(VERBOSE):
            print("layer.name: ",layer_name)
            #print("layer_prefix: ",layer_prefix)

        if(layer_prefix == 'conv'):
            if(VERBOSE):
                print("---------------->")
                print("conv layyyyerr")
            weights_biases = my_original_model.layers[idx].get_weights()
            weights_biases_np = np.array(weights_biases)
            weights = weights_biases_np[0]
            biases = weights_biases_np[1]

            all_weights_original_dict[layer_name] = weights_biases_np
            if(VERBOSE):
                print("weights_biases_np.shape: ",weights_biases_np.shape)
                print("weights.shape: ",weights.shape)
                print("biases.shape: ",biases.shape)
                print("<-----------")
        if(layer_prefix == 'prel'):
            if(VERBOSE):
                print("---------------->")
                print("prelu layyyyerr")
            params = my_original_model.layers[idx].get_weights()
            params_np = np.array(params)

            all_weights_original_dict[layer_name] = params_np
            if(VERBOSE):
                print("params.shape: ",params_np.shape)
                print("<-----------")
        if(VERBOSE):
            print("")
        
    return all_weights_original_dict