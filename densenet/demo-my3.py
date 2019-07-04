# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:53:35 2019

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:33:43 2019

@author: Admin
"""
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
import cv2

from keras.layers import Input, Dense, Flatten
from keras.models import load_model,Model

# import keras.backend as K

from nets import keys
import densenet


num_to_char = {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
               11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 
               20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 
               29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z',37:'#'}
               
char_to_num ={'0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10, 
              'A': 11, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
              'K': 21, 'L': 22, 'M': 23, 'N': 24, 'O': 25, 'P': 26, 'Q': 27, 'R': 28, 'S': 29, 'T': 30, 
              'U': 31, 'V': 32, 'W': 33, 'X': 34, 'Y': 35, 'Z': 36}               
               
nclass = 36
maxlabellength = 10

model_file = 'models/weights_densenet-1865-2.74.h5' 




def readcsvfile_test(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(',')
        dic[p[0]] = p[1].strip()
    return dic




label = readcsvfile_test('train_id_label.csv')
#print(label['01AUV9WG.jpg'])

input1 =Input(shape=(64, 128, 1), name='the_input')
y = densenet.dense_cnn2(input1, nclass = nclass*maxlabellength)
predictions = y #Dense(36, activation='softmax')(y )
model = Model(input1, outputs=predictions)
model.load_weights(model_file)


t = 0
folder = './images'
paths = os.listdir(folder)
path = './images/S1UA8LQC.jpg'
nn = []
for i in paths[20000:21000]:
    orig = i
    print(i)
    i = folder + '/' + i
    #im = cv2.imread(path)
    img = Image.open(i).convert('L')
    img = img.resize((128, 64),Image.ANTIALIAS)
    img = np.array(img)
    #print(img.shape)
    batchsize = 1
    #print(img.size)
    x = np.zeros((batchsize, img.shape[0], img.shape[1], 1), dtype=np.float)
    x[0] = np.expand_dims(img, axis=2)
    y_pred = model.predict(x)
    #print(y_pred)
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

    print('predict',re)
    print('label  ',label[orig])
    lab = list(label[orig])
    n = 0
    for i in range(10):
        if res[i] == lab[i]:
            n += 1
    nn.append(n)
    if re == label[orig]:
        print(orig)
        print(True)
        t += 1
    print('-------------------------')
print('\nt',t)
print('\nnn',nn)
total = 0
al = 0
for i in nn:
    total += i
    al += 10
print('per',total/al)
print('#################################')