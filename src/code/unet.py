# -*- coding: utf-8 -*-
# """
# Created on April 19th 2024

# @author: VIVEK OOMMEN
# """

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D,  Dense, Activation, GroupNormalization

class unet(tf.keras.Model):
    def __init__(self, Par):
        super(unet, self).__init__()
        np.random.seed(23)        
        tf.random.set_seed(23)

        self.Par = Par
        
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        n_kernels = 32
        kx=3
        ky=3
        activation='gelu'
        padding='same'
        
        #Defining the unet layers
        self.enc0 = self.block(int(n_kernels/2), kx,ky, activation, padding, name='conv0', flag=True)                        #[_,256,256, n_kernels/2]
        self.pool0= MaxPooling2D(pool_size = (2,2), strides = 2, name='pool0')                                               #[_,128,128, n_kernels/2]
        self.enc1 = self.block(1*n_kernels, kx,ky, activation, padding, name='conv1')                                        #[_,128,128, n_kernels]
        self.pool1= MaxPooling2D(pool_size = (2,2), strides = 2, name='pool1')                                               #[_,64,64, n_kernels]
        self.enc2 = self.block(2*n_kernels, kx,ky, activation, padding, name='conv2')                                        #[_,64,64, 2*n_kernels]
        self.pool2= MaxPooling2D(pool_size = (2,2), strides = 2, name='pool2')                                               #[_,32,32, 2*n_kernels]
        self.enc3 = self.block(4*n_kernels, kx,ky, activation, padding, name='conv3')                                        #[_,32,32, 4*n_kernels]
        self.pool3= MaxPooling2D(pool_size = (2,2), strides = 2, name='pool3')                                               #[_,16,16, 4*n_kernels]
        self.enc4 = self.block(8*n_kernels, kx,ky, activation, padding, name='conv4')                                        #[_,16,16, 8*n_kernels]
        self.pool4= MaxPooling2D(pool_size = (2,2), strides = 2, name='pool4')                                               #[_,8,8, 8*n_kernels] 

        self.bottleneck = self.block(16*n_kernels, kx,ky, activation, padding, name='bottleneck')                            #[_,8,8, 16*n_kernels]

        self.tconv4 = Conv2DTranspose(8*n_kernels, kernel_size=(2,2), strides=(2,2), activation = 'gelu',  name='tconv4')    #[_,16,16, 8*n_kernels]
        self.dec4   = self.block(8*n_kernels, kx,ky, activation, padding, name='dec4')                                       #[_,16,16, 8*n_kernels]
        self.tconv3 = Conv2DTranspose(4*n_kernels, kernel_size=(2,2), strides=(2,2), activation = 'gelu',  name='tconv3')    #[_,32,32, 4*n_kernels]
        self.dec3   = self.block(4*n_kernels, kx,ky, activation, padding, name='dec3')                                       #[_,32,32, 4*n_kernels]
        self.tconv2 = Conv2DTranspose(2*n_kernels, kernel_size=(2,2), strides=(2,2), activation = 'gelu',  name='tconv2')    #[_,64,64, 2*n_kernels]
        self.dec2   = self.block(2*n_kernels, kx,ky, activation, padding, name='dec2')                                       #[_,64,64, 2*n_kernels]
        self.tconv1 = Conv2DTranspose(1*n_kernels, kernel_size=(2,2), strides=(2,2), activation = 'gelu',  name='tconv1')    #[_,128,128, 1*n_kernels]
        self.dec1   = self.block(1*n_kernels, kx,ky, activation, padding, name='dec1')                                       #[_,128,128, 1*n_kernels]
        self.tconv0 = Conv2DTranspose(int(n_kernels/2), kernel_size=(2,2), strides=(2,2), activation = 'gelu',  name='tconv0')   #[_,256,256, n_kernels/2]
        self.dec0   = self.block(int(n_kernels/2), kx,ky, activation, padding, name='dec0')                                       #[_,256,256, n_kernels/2]

        self.final_norm = GroupNormalization( int(n_kernels/8) )
        self.final  = Conv2D(self.Par['nf'], (1,1))                                                #[_,128,128, self.nf]   

        #Defining the trunk network
        self.trunk_net = Sequential(name='trunk_net')
        self.trunk_net.add(Dense(128))
        self.trunk_net.add(Activation(tf.math.sin))
        self.trunk_net.add(Dense(128))
        self.trunk_net.add(Activation(tf.math.sin))
        
        self.dense0 = Dense(int(n_kernels/2))
        self.dense1 = Dense(1*n_kernels)
        self.dense2 = Dense(2*n_kernels)
        self.dense3 = Dense(4*n_kernels)
        self.dense4 = Dense(8*n_kernels)
        self.dense5 = Dense(16*n_kernels)

    # Q: ordering of conv, norm, activation
    def block(self, n_kernels, kx,ky, activation, padding, name, flag=False, n_groups=1):
        block = Sequential(name=name)
        if flag:
            block.add( Conv2D(n_kernels, (kx,ky), padding=padding, input_shape=[self.Par['nx'],self.Par['ny'],self.Par['lb']*self.Par['nf']] ) )
        else:
            block.add( Conv2D(n_kernels, (kx,ky), padding=padding ) )

        block.add( GroupNormalization(n_groups) )
        block.add( Activation(activation) )

        return block

    # @tf.function()
    def call(self, x, dt):
    # x - [_,128,128,lb,nf]
    # dt- [nt,1]
        nt = tf.shape(dt)[0]        

        x = (x - self.Par['inp_shift'])/self.Par['inp_scale']
        x = tf.reshape(x, (-1, self.Par['nx'], self.Par['ny'], self.Par['lb']*self.Par['nf']))

        e0 = self.enc0(x)                                                                               #[_,256,256, n_kernels/2]
        e1 = self.enc1(self.pool0(e0))                                                                  #[_,128,128, 1*n_kernels]
        e2 = self.enc2(self.pool1(e1))                                                                  #[_,64,64, 2*n_kernels]
        e3 = self.enc3(self.pool2(e2))                                                                  #[_,32,32, 4*n_kernels]
        e4 = self.enc4(self.pool3(e3))                                                                  #[_,16,16, 8*n_kernels]

        bottleneck = self.bottleneck(self.pool4(e4))                                                    #[_,8,8, 16*n_kernels]
 
        f_dt = self.trunk_net(dt)                                                                         #[nt, 128]
        f0_dt = self.dense0(f_dt)                                                                         #[nt, n_kernels/2]
        f1_dt = self.dense1(f_dt)                                                                         #[nt, 1*n_kernels]
        f2_dt = self.dense2(f_dt)                                                                         #[nt, 2*n_kernels]
        f3_dt = self.dense3(f_dt)                                                                         #[nt, 4*n_kernels]
        f4_dt = self.dense4(f_dt)                                                                         #[nt, 8*n_kernels]
        f5_dt = self.dense5(f_dt)                                                                         #[nt, 16*n_kernels]

        temp = tf.einsum('ijkl,pl->ipjkl', bottleneck,f5_dt)                                            #[_,nt,8,8,16*n_kernels]
        BOTTLENECK = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3], tf.shape(temp)[4]])    #[_*nt,8,8,16*n_kernels]

        d4 = self.tconv4(BOTTLENECK)                                                                    #[_*nt,16,16, 8*n_kernels]
        temp = tf.einsum('ijkl,pl->ipjkl', e4,f4_dt)                                                    #[_,nt,16,16,8*n_kernels]
        E4 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3], tf.shape(temp)[4]])            #[_*nt,16,16,8*n_kernels]
        d4 = tf.concat([d4,E4], axis=-1)                                                                #[_*nt,16,16, 2*(8*n_kernels)]
        d4 = self.dec4(d4)                                                                              #[_*nt,16,16, 8*n_kernels]

        d3 = self.tconv3(d4)                                                                            #[_*nt,32,32, 4*n_kernels]
        temp = tf.einsum('ijkl,pl->ipjkl', e3,f3_dt)                                                    #[_,nt,32,32,4*n_kernels]
        E3 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3], tf.shape(temp)[4]])            #[_*nt,32,32,4*n_kernels]
        d3 = tf.concat([d3,E3], axis=-1)                                                                #[_*nt,32,32, 2*(4*n_kernels)]
        d3 = self.dec3(d3)                                                                              #[_*nt,32,32, 4*n_kernels]

        d2 = self.tconv2(d3)                                                                            #[_*nt,64,64, 2*n_kernels]
        temp = tf.einsum('ijkl,pl->ipjkl', e2,f2_dt)                                                    #[_,nt,64,64,2*n_kernels]
        E2 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3], tf.shape(temp)[4]])            #[_*nt,64,64,2*n_kernels]
        d2 = tf.concat([d2,E2], axis=-1)                                                                #[_*nt,64,64, 2*(2*n_kernels)]
        d2 = self.dec2(d2)                                                                              #[_*nt,64,64, 2*n_kernels]

        d1 = self.tconv1(d2)                                                                            #[_*nt,128,128, 1*n_kernels]
        temp = tf.einsum('ijkl,pl->ipjkl', e1,f1_dt)                                                    #[_,nt,128,128,1*n_kernels]
        E1 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3], tf.shape(temp)[4]])            #[_*nt,128,128,1*n_kernels]
        d1 = tf.concat([d1,E1], axis=-1)                                                                #[_*nt,128,128, 2*(1*n_kernels)]
        d1 = self.dec1(d1)                                                                              #[_*nt,128,128, 1*n_kernels]

        d0 = self.tconv0(d1)                                                                            #[_*nt,256,256, n_kernels/2]
        temp = tf.einsum('ijkl,pl->ipjkl', e0,f0_dt)                                                    #[_,nt,256,256, n_kernels/2]
        E0 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3], tf.shape(temp)[4]])            #[_*nt,256,256, n_kernels/2]
        d0 = tf.concat([d0,E0], axis=-1)                                                                #[_*nt,256,256, 2*(n_kernels/2)]
        d0 = self.dec0(d0)                                                                              #[_*nt,256,256, n_kernels/2] 

        out = self.final_norm(d0)
        out = self.final(out)                                                                           #[_*nt,256,256, self.nf]
        out = tf.reshape(out, [-1,nt,tf.shape(out)[1],tf.shape(out)[2],tf.shape(out)[3] ])              #[_,nt,256,256, self.nf]
        
        out = out*self.Par['out_scale'] + self.Par['out_shift']

        return out


    def loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_train-y_pred))/tf.reduce_mean(tf.square(y_train))

        return([train_loss])
        
