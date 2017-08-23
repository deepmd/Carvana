from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout

def conv_block(inputs, filters, acti, init, bn, do=0):
    n = Conv2D(filters, (3, 3), activation=acti, kernel_initializer=init, padding='same')(inputs)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(filters, (3, 3), activation=acti, kernel_initializer=init, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return n

def level_block(inputs, filters, depth, acti, init, do, bn, mp, up):
    if depth > 0:
        n = conv_block(inputs, filters, acti, init, bn, do)
        m = MaxPooling2D()(n) if mp else Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(n)
        m = level_block(m, 2*filters, depth-1, acti, init, do, bn, mp, up)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(filters, (2, 2), activation=acti, kernel_initializer=init, padding='same')(m)
        else:
            m = Conv2DTranspose(filters, (3, 3), strides=(2, 2), activation=acti, kernel_initializer=init, padding='same')(m)
        m = Concatenate()([n, m])
        m = conv_block(m, filters, acti, init, bn)
    else:
        m = conv_block(inputs, filters, acti, init, bn, do)
    return m

def UNet(img_shape, num_classes=1, filters=64, depth=4, activation='relu', init='he_uniform', 
         dropout=0.2, batchnorm=True, maxpool=True, upconv=True):
    '''
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (https://arxiv.org/abs/1505.04597)
    ---
    img_shape: (height, width, channels)
    num_classes: number of output channels
    filters: number of filters of the first conv
    depth: zero indexed depth of the U-structure
    activation: activation function after convolutions
    init: kernels initialization function
    dropout: amount of dropout in the contracting part
    batchnorm: adds Batch Normalization if true
    maxpool: use strided conv instead of maxpooling if false
    upconv: use transposed conv instead of upsamping + conv if false
    '''
    i = Input(shape=img_shape)
    o = level_block(i, filters, depth, activation, init, dropout, batchnorm, maxpool, upconv)
    o = Conv2D(num_classes, (1, 1), activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)
