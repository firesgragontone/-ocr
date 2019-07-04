# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 23:22:09 2019

@author: Admin
"""

#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
import cv2

from keras.layers import Input
from keras.models import Model
# import keras.backend as K

from nets import keys
import densenet

reload(densenet)

num_to_char = {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
               11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 
               20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 
               29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z',37:'#'}


characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
#nclass = len(characters)
nclass = 37
input1 = Input(shape=(64, None, 1), name='the_input')
y_pred= densenet.dense_cnn(input1, nclass)
basemodel = Model(inputs=input1, outputs=y_pred)

mo = os.path.join(os.getcwd(), 'models/model1/weights_densenet-13-0.73.h5')
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

def predict(img):
    
    #width, height = img.size[0], img.size[1]
    #scale = height * 1.0 / 32
    #width = int(width / scale)
    
    #img = img.resize([width, 32], Image.ANTIALIAS)

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    #img = np.array(img).astype(np.float32) / 255.0 - 0.5
    
    #X = img.reshape([1, 32, width, 1])
    
    #y_pred = basemodel.predict(X)
    batchsize = 1
    x = np.zeros((batchsize, img.size[1], img.size[0], 1), dtype=np.float)
    x[0] = np.expand_dims(img, axis=2)
    #X  = []
    #X.append(img)
    y_pred = basemodel.predict(x)
    
    print(y_pred.shape)
    
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    #out = decode(y_pred)
    #print(out)
    #return out
    print(len(y_pred),len((y_pred[0])),y_pred.shape)
    y_pred = y_pred.tolist()
    res = []
    re = ''
    for k in range(len(y_pred)):
        for j in range(len(y_pred[0])):
            #print(j)
            i = y_pred[k][j]
        #i = y_pred[j].tolist()
            res.append(num_to_char[1 + i.index(max(i[0:-1]))])
    for i in res:
        re += i
    print(re)
    return res

folder = './test_images'
paths = os.listdir(folder)
path = './images/S1UA8LQC.jpg'
for i in paths:
    print(i)
    i = folder + '/' + i
    #im = cv2.imread(path)
    im = Image.open(i).convert('L')
    img = im.resize((80, 64),Image.ANTIALIAS)
    #im = np.array(im)
    predict(img)
