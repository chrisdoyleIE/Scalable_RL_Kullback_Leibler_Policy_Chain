#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:31:18 2019

@author: CP
"""
import numpy as np
import matplotlib as plt
import tensorflow as tf

def KL(p,q):
    
    #KL Divergence = - Sum[ p(x) log (q(x) / p(x))  ]
    return - np.dot(p, np.log( np.divide(q,p) ) )
    

p1 = np.asarray([0.1,0.1,0.75,0.05])
p2 = np.asarray([0.15,0.05,0.75,0.05])

D_kl = KL(p1,p2)

p3 = (p1+p2)/2

