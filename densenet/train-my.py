#-*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
import random
from PIL import Image

import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten,Softmax
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload
import densenet


char_to_num ={'0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10, 
              'A': 11, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
              'K': 21, 'L': 22, 'M': 23, 'N': 24, 'O': 25, 'P': 26, 'Q': 27, 'R': 28, 'S': 29, 'T': 30, 
              'U': 31, 'V': 32, 'W': 33, 'X': 34, 'Y': 35, 'Z': 36}
num_to_char = {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
               11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 
               20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 
               29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'}

'''
img_h = 32
img_w = 280
batch_size = 128
maxlabellength = 10
'''
img_h = 64
img_w = 128
batch_size = 32
maxlabellength = 10
nclass = 36



def get_session(gpu_fraction=1.0):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic

#a = readfile('data_test.txt')
#print(len(a))


def readcsvfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines[1:-3200]:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(',')
        label = []
        for i in p[1].strip():
            label.append(str(char_to_num[i]))
        if p[0]=='name':continue
        dic[p[0]] = label
    return dic

def readcsvfile_test(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines[-3200:]:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(',')
        label = []
        for i in p[1].strip():
            label.append(str(char_to_num[i]))
        dic[p[0]] = label
    return dic

#a = readcsvfile('train_id_label.csv')
#print(a)

class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n

############################################################


###########################################################3
        
    
    
    
def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            print(os.path.join(image_path, j))
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str1 = image_label[j]
            label_length[i] = len(str1)

            if(len(str1) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str1)] = [int(k) - 1 for k in str1]
            print(labels)
        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length, 
                'label_length': label_length,
                }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)



def gen_my1(data_file, image_path, batchsize=batch_size, maxlabellength=10, imagesize=(img_h,img_w)):
    #image_label = readfile(data_file)
    image_label = readcsvfile_test(data_file)
    
    _imagefile = [i for i, j in image_label.items()]
    
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength,nclass])# * 10000
    #print('labedls',labels.shape)
    input_length = np.zeros([batchsize, maxlabellength])
    label_length = np.zeros([batchsize, maxlabellength])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f')# / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str1 = image_label[j]
            '''
            label_length[i] = len(str1)
            if len(str1)==11:
                print(j,len(str1))
            if(len(str1) <= 0):
                print("len < 0", j)
            '''
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str1)] = np.array([[1 if i ==(int(k) - 1) else 0 for i in range(nclass)] for k in str1])
            label_length[i] = [len([1 if i ==(int(k) - 1) else 0 for i in range(nclass)]) for k in str1]
        #print('gen label',labels.shape)
        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
        #outputs = {'ctc': np.zeros([batchsize])}
        #yield (inputs, outputs)
        #print("my1.x,labels,input_length,label_length",x.shape,labels.shape,input_length.shape,label_length.shape)
        #yield (inputs, np.zeros([batchsize, maxlabellength,37]))
        yield (inputs,labels)
        #yield(x,labels)

def gen_my(data_file, image_path, batchsize=batch_size, maxlabellength=10, imagesize=(img_h,img_w)):
    print('22222222222222222222222')
    #image_label = readfile(data_file)
    image_label = readcsvfile(data_file)
    
    _imagefile = [i for i, j in image_label.items()]
    
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength,nclass])# * 10000
    
    input_length = np.zeros([batchsize, maxlabellength])
    label_length = np.zeros([batchsize, maxlabellength])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile) 
    
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f')# / 255.0 - 0.5
            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str1 = image_label[j]
            #label_length[i] = len(str1)

            #if(len(str1) <= 0):
            #    print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            #labels[i, :len(str1)] = [int(k) - 1 for k in str1]
            labels[i, :len(str1)] = np.array([[1 if i ==(int(k) - 1) else 0 for i in range(nclass)] for k in str1])
            label_length[i] = [len([1 if i ==(int(k) - 1) else 0 for i in range(nclass)]) for k in str1]
#            print(labels[1,1])
        #print(os.path.join(image_path, j))
        #print(labels)
        #print('label',labels.shape)
        #print(input_length,label_length)
        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
        '''
        f = open('test.txt','a')
        f.write('label;;\n')
        f.write(str(labels.shape))
        f.write('\n')
        f.write(str(labels[5]))
        f.write('\n')
        f.write('x;;\n')
        f.write(str(x.shape))
        f.write('\n')
        f.write(str(x[5][0]))
        f.write('\n')
        f.close()
        '''
        
        ##outputs = {'ctc': np.zeros([batchsize])}
        #yield (inputs, outputs)
        #print('labels[i, :len(str1)].shape')
        #print("my,x,labels,input_length,label_length",x.shape,labels.shape,input_length.shape,label_length.shape)
        #yield (inputs,np.zeros([batchsize, maxlabellength,37]))
        #print('\nx[0]',x[0])
        #print('\nlabel',labels[0])
        yield (inputs,labels)
        #yield(x,labels)
#print('11111111')
#gen_my('train_id_label.csv', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
#print('444444444444444')
#img1 = Image.open('images\HG53ZLWR.jpg').convert('L')
#img = np.array(img1, 'f') / 255.0 - 0.5


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model1(img_h, nclass):
    input_ = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn2(input_, nclass)
    print('y_pred',y_pred.shape)
    basemodel = Model(inputs=input_, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input_, labels, input_length, label_length], outputs=loss_out)
    model.summary()
    #model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return basemodel, model


def get_model(img_h, nclass):
    input_ = Input(shape=(img_h, img_w, 1), name='the_input')
    print('nclass',nclass)
    y_pred = densenet.dense_cnn2(input_, nclass)
    print('y_pred',y_pred.shape)
    #basemodel = Model(inputs=input, outputs=y_pred)
    #basemodel.summary()
    '''
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    '''
    '''
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    '''
    labels = Input(name='the_labels', shape=[None,None], dtype='float32')
    input_length = Input(name='input_length', shape=[None], dtype='int64')
    label_length = Input(name='label_length', shape=[None], dtype='int64')
#    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    #model = Model(inputs=[input_, labels, input_length, label_length], outputs=loss_out)
    #y_pred = Softmax(axis=-1)(y_pred)#,activation='softmax'
    #print('y_pred',y_pred.shape)
    #predictions = Reshape((-1, maxlabellength, nclass))(y_pred)#, activation='softmoid'
    
    model = Model(inputs=[input_, labels, input_length, label_length], outputs=y_pred)
    model.summary()
    #model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    model.compile(loss='hinge', optimizer=Adam(lr=1e-4), metrics=['accuracy'])#categorical_crossentropy
    return model


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.9)
        print("lr changed to {}".format(lr * 0.9))
    return K.get_value(model.optimizer.lr)




if __name__ == '__main__':
    #char_set = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
    #char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    #nclass = len(char_set)
    
    K.set_session(get_session())
    reload(densenet)
    model = get_model(img_h, nclass*maxlabellength)
    #weights_densenet-02-0.91.h5
    modelPath = './models/weights_densenet-1363-0.05.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        model.load_weights(modelPath)
        print('done!')
    else:
        print('not found model')
    
    
    '''
    modelPath = './models/model2/weights_densenet-11-0.76.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')
        
        
        
        
    '''
    #modelPath = './models/model2/weights_densenet-11-0.76.h5'
    #if os.path.exists(modelPath):
    #    print("Loading model weights...")
    #    model.load_weights(modelPath)
    #        print('done!')
    #train_loader = gen('data_train.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    #test_loader = gen('data_test.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    train_loader = gen_my('train_id_label_39616.csv', './images2', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen_my1('train_id_label_39616.csv', './images2', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    checkpoint = ModelCheckpoint(filepath='./models/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=False)
    lr_schedule = lambda epoch: 0.01 * 0.4**epoch if epoch>4 or random.randint(0,20)!=5 else  0.01
    #lr_schedule = lambda epoch: 0.0001 * 0.4**epoch
    #learning_rate = np.array([lr_schedule(i) for i in range(1000)])
    #changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    changelr = LearningRateScheduler(scheduler)
    #earlystop = EarlyStopping(monitor='val_loss', patience=1000, verbose=1)sparse_categorical_crossentropy
    earlystop = EarlyStopping(monitor='val_loss', patience=1000, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')

    model.fit_generator(train_loader,
    	steps_per_epoch = (39616-3200) // batch_size,
    	epochs = 3000,
    	initial_epoch = 1363,
    	validation_data = test_loader,
    	validation_steps = 3200 // batch_size,
        callbacks = [checkpoint, changelr, tensorboard]
    	)
    '''
     model.fit_generator(train_loader,
    	steps_per_epoch = (39620-3200) // batch_size,
    	epochs = 1000,
    	initial_epoch = 0,
    	validation_data = test_loader,
    	validation_steps = 3200 // batch_size,
    	callbacks = [checkpoint, earlystop, changelr, tensorboard])
    '''

