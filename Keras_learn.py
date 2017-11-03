# -*- coding: utf-8 -*-
from __future__ import division
import os
os.getcwd()
import numpy as np
import pandas as pd
import random as rd
from sklearn import datasets
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def zscore(data, mu, sigma):
    assert isinstance(data, np.ndarray), "type of data_obj must be np.ndarray"
    res = (data-mu) / (sigma+10**-8)
    return res

def BN(data, gamma, kappa):
    assert isinstance(data, np.ndarray), "type of data must be np.ndarray"
    zscore = (data-np.mean(data, axis=0, keepdims=True)) / (np.std(data, axis=0, keepdims=True)+10**-8)
    batch_norm = gamma*zscore + kappa
    return zscore, batch_norm, np.mean(data, axis=0, keepdims=True), np.std(data, axis=0, keepdims=True)

# active_func
def sigmiod(z):
    res = 1.0 / (1.0 + np.exp(-z))
    return res
    
def tanh(z):
    res = 2.0 / (1.0 + np.exp(-z)) - 1
    return res
    
def relu(z):
    res = np.maximum(0, z)
    return res
    
def leaky_relu(z):
    res = np.maximum(0.01*z, z)
    return res
    
def softmax(z):
    res = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return res
    
def tanh_dv(z):
    res = 1-(np.tanh(z))**2
    return res
    
def relu_dv(z):
    res = np.where(z < 0, 0, 1)
    return res
    
def leaky_relu_dv(z):
    res = np.where(z < 0, 0.01, 1)
    return res

# alpha_decay
def alpha_decay_1(alpha, epoch, decay_rate):
    alpha_update = decay_rate**epoch * alpha
    return alpha_update

def alpha_decay_2(alpha, epoch, decay_rate):
    alpha_update = 1.0 / (1.0 + decay_rate*epoch) * alpha
    return alpha_update

def unit_vec(x):
    res = x/np.sqrt(np.sum(x**2))
    return res

def gradient_descent(alpha, param, param_dv):
    param -= alpha*unit_vec(param_dv)
    return param

def Momentum(w, dw, v_dw, beta_1, alpha, i):
    v_dw = beta_1*v_dw + (1-beta_1)*dw
    v_dw_corr = v_dw/(1-beta_1**i)
    w -= alpha * unit_vec(v_dw_corr)
    return w, v_dw_corr

def Adam(w, dw, v_dw, s_dw, beta_1, beta_2, alpha, i):
    v_dw = beta_1*v_dw + (1-beta_1)*dw
    s_dw = beta_2*s_dw + (1-beta_2)*(dw**2)
    w -= alpha * v_dw/(np.sqrt(s_dw)+10**-8)
    return w, v_dw, s_dw

#def Adam(w, dw, v_dw, s_dw, beta_1, beta_2, alpha, i):
#    v_dw = beta_1*v_dw + (1-beta_1)*dw
#    s_dw = beta_2*s_dw + (1-beta_2)*(dw**2)
#    v_dw_corr = v_dw/(1-beta_1**i)
#    s_dw_corr = s_dw/(1-beta_2**i)
#    w -= alpha * v_dw_corr/(np.sqrt(s_dw_corr)+10**-8)
#    return w, v_dw_corr, s_dw_corr

# drop_out
keep_prob = 0.8
def drop_out(a, keep_prob):
    d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
    a *= d
    a /= keep_prob
    return a
