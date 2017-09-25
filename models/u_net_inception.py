from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Activation


'''def base_block(inputs, filters, acti_layer, init, bn, do=0):
    n = inception_block(inputs, filters, acti_layer)
    n = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(inputs)
    n = acti_layer(n)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(n)
    n = acti_layer(n)
    n = BatchNormalization()(n) if bn else n
    return n'''

def inception_block(inputs, filters, init, actv, splitted=True):
    assert filters % 16 == 0
    c1_1 = Conv2D(filters/4, (1, 1), kernel_initializer=init, padding='same')(inputs)   
    c1_1 = actv(c1_1)
    c1_1 = BatchNormalization()(c1_1)

    c2_1 = Conv2D(filters/8*3, (1, 1), kernel_initializer=init, padding='same')(inputs)
    c2_1 = actv(c2_1)
    c2_1 = BatchNormalization()(c2_1)
    if splitted:
        c2_2 = Conv2D(filters/2, (1, 3), kernel_initializer=init, padding='same')(c2_1)
        c2_2 = actv(c2_2)
        c2_2 = BatchNormalization()(c2_2)
        c2_3 = Conv2D(filters/2, (3, 1), kernel_initializer=init, padding='same')(c2_2)
        c2_3 = actv(c2_3)
        c2_3 = BatchNormalization()(c2_3)
    else:
        c2_3 = Conv2D(filters/2, (3, 3), kernel_initializer=init, padding='same')(c2_1)
        c2_3 = actv(c2_3)
        c2_3 = BatchNormalization()(c2_3)    

    c3_1 = Conv2D(filters/16, (1, 1), kernel_initializer=init, padding='same')(inputs)
    c3_1 = actv(c3_1)
    c3_1 = BatchNormalization()(c3_1)

    if splitted:
        c3_2 = Conv2D(filters/8, (1, 3), kernel_initializer=init, padding='same')(c3_1)
        c3_2 = actv(c3_2)
        c3_2 = BatchNormalization()(c3_2)
        c3_3 = Conv2D(filters/8, (3, 1), kernel_initializer=init, padding='same')(c3_2)
        c3_3 = actv(c3_3)
        c3_3 = BatchNormalization()(c3_3)
    else:
        c3_3 = Conv2D(filters/8, (3, 3), kernel_initializer=init, padding='same')(c3_1)
        c3_3 = actv(c3_3)
        c3_3 = BatchNormalization()(c3_3)
    
    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(inputs)
    c4_2 = Conv2D(filters/8, (1, 1), kernel_initializer=init, padding='same')(p4_1)
    c4_2 = actv(c4_2)
    c4_2 = BatchNormalization()(c4_2)
    
    res = Concatenate()([c1_1, c2_3, c3_3, c4_2])
    res = actv(res)
    res = BatchNormalization()(res)
    
    return res


def level_block(inputs, filters, depth, acti_layer, init, do, dbo, bn, mp, up):
    if depth > 0:
        n = inception_block(inputs, filters, init, acti_layer)
        m = MaxPooling2D()(n) if mp else Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(n)
        m = BatchNormalization()(m)
        m = acti_layer(m)
        m = Dropout(do)(m)
        m = level_block(m, 2*filters, depth-1, acti_layer, init, do, dbo, bn, mp, up)
        if up:
            m = UpSampling2D()(m)
            #m = Conv2D(filters, (2, 2), kernel_initializer=init, padding='same')(m)
            #m = acti_layer(m)
        else:
            m = Conv2DTranspose(filters, (3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(m)
            #m = acti_layer(m)
        m = Concatenate()([n, m])
        m = inception_block(m, filters, init, acti_layer)
        m = Dropout(do)(m)
    else:
        m = inception_block(inputs, filters, init, acti_layer)
        m = Dropout(do)(m)
    return m

def UNet_INCEPTION(img_shape, num_classes=1, filters=64, depth=4, activation=lambda x: Activation('relu')(x),
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
