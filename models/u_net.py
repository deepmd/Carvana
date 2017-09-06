from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Activation

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
        m = level_block(m, 2*filters, depth-1, acti_layer, init, do, dbo, bn, mp, up)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(filters, (2, 2), kernel_initializer=init, padding='same')(m)
            m = acti_layer(m)
        else:
            m = Conv2DTranspose(filters, (3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(m)
            m = acti_layer(m)
        m = Concatenate()([n, m])
        m = conv_block(m, filters, acti_layer, init, bn)
    else:
        m = conv_block(inputs, filters, acti_layer, init, bn, do)
        m = Dropout(0.2)(m)
    return m

def UNet(img_shape, num_classes=1, filters=64, depth=4, activation=lambda x: Activation('relu')(x),
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
    o = level_block(i, filters, depth, activation, init, dropout, dropout_base_only, batchnorm, maxpool, upconv)
    o = Conv2D(num_classes, (1, 1), activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)
