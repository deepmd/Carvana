from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Activation, Lambda
from keras.layers.merge import Average, Add
from keras import backend as K

def conv_block(inputs, filters, acti_layer, init, bn, do=0):
    n = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(inputs)
    n = acti_layer(n)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(n)
    n = acti_layer(n)
    n = BatchNormalization()(n) if bn else n
    return n

def level_block(inputs, filters, depth, acti_layer, init, do, dbo, bn, mp, up):
    if depth > 0:
        n = conv_block(inputs, filters, acti_layer, init, bn, 0 if dbo else do)
        m = MaxPooling2D()(n) if mp else Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(n)
        seg, con = level_block(m, 2*filters, depth-1, acti_layer, init, do, dbo, bn, mp, up)
        if up:
            # Segmentation expanding path
            seg = UpSampling2D()(seg)
            seg = Conv2D(filters, (2, 2), kernel_initializer=init, padding='same')(seg)
            seg = acti_layer(seg)
            
            # Countour expanding path
            con = UpSampling2D()(con)
            con = Conv2D(filters, (2, 2), kernel_initializer=init, padding='same')(con)
            con = acti_layer(con)
        else:
            # Segmentation expanding path
            seg = Conv2DTranspose(filters, (3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(seg)
            seg = acti_layer(seg)
            
            # Countour expanding path
            con = Conv2DTranspose(filters, (3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(seg)
            con = acti_layer(seg)
            
        seg = Concatenate()([n, seg])
        seg = conv_block(seg, filters, acti_layer, init, bn)
        
        con = Concatenate()([n, con])
        con = conv_block(con, filters, acti_layer, init, bn)
    else:
        seg = conv_block(inputs, filters, acti_layer, init, bn, do)
        seg = Dropout(0.2)(seg)
        con = seg
        
    return (seg, con)

def DCAN(img_shape, num_classes=1, filters=64, depth=4, activation=lambda x: Activation('relu')(x),
         init='glorot_uniform', dropout=0.2, dropout_base_only=False, batchnorm=True, maxpool=True, upconv=True):
    '''
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (https://arxiv.org/abs/1505.04597)
    ---
    img_shape: (height, width, channels)
    num_classes: number of output channels
    filters: number of filters of the first conv
    depth: zero indexed depth of the U-structure
    activation: activation layer function after convolutions    default: Relu
    init: kernels initialization function
    dropout: amount of dropout in the contracting part
    batchnorm: adds Batch Normalization if true
    maxpool: use strided conv instead of maxpooling if false
    upconv: use transposed conv instead of upsamping + conv if false
    '''
    
    i = Input(shape=img_shape)
    (seg, con) = level_block(i, filters, depth, activation, init, dropout, dropout_base_only, batchnorm, maxpool, upconv)
    seg_out = Conv2D(num_classes, (1, 1), activation='sigmoid', name='seg_out')(seg)
    con_out = Conv2D(num_classes, (1, 1), activation='sigmoid', name='contour_out')(con)
    return Model(inputs=i, outputs=[seg_out, con_out])

    #seg_con = Add()([seg, con])
    #seg_con = Average()([seg, con])
    #seg_con = Lambda(lambda x: x // 2)(seg_con)
    #final_out = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_seg')(seg_con)
    #return Model(inputs=i, outputs=[seg_out, con_out, final_out])
