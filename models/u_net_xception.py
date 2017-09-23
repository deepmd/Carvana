from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Activation, add
from keras.layers.convolutional import SeparableConv2D
from keras.applications.xception import Xception

def xception_block(inputs, filters, acti_layer, init):
    n = SeparableConv2D(filters, (3, 3), depthwise_initializer=init, pointwise_initializer=init, padding='same')(inputs)
    n = BatchNormalization()(n)
    n = acti_layer(n)
    n = SeparableConv2D(filters, (3, 3), depthwise_initializer=init, pointwise_initializer=init, padding='same')(n)
    n = BatchNormalization()(n)
    return n

def xception_block_middle(inputs, filters, acti_layer, init):
    n = acti_layer(inputs)
    n = SeparableConv2D(filters, (3, 3), depthwise_initializer=init, pointwise_initializer=init, padding='same')(n)
    n = BatchNormalization()(n)
    n = acti_layer(n)
    n = SeparableConv2D(filters, (3, 3), depthwise_initializer=init, pointwise_initializer=init, padding='same')(n)
    n = BatchNormalization()(n)
    n = acti_layer(n)
    n = SeparableConv2D(filters, (3, 3), depthwise_initializer=init, pointwise_initializer=init, padding='same')(n)
    n = BatchNormalization()(n)
    return n

def level_block(inputs, filters, depth_count, depth, acti_layer, init, mp, res):
    if depth > 0:
        n = inputs if depth == depth_count else acti_layer(inputs)
        n = xception_block(n, filters, acti_layer, init)
        m = MaxPooling2D()(n) if mp else Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(n)
        if res:
            l = Conv2D(filters, (1, 1), strides=(2, 2), kernel_initializer=init, padding='same')(inputs)
            l = BatchNormalization()(l)
            m = add([l, m])
        m = level_block(m, 2*filters, depth_count, depth-1, acti_layer, init, mp, res)
        o = m if depth == depth_count else acti_layer(m)
        o = UpSampling2D()(o)
        o = Concatenate()([n, o])
        o = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(o)
        o = acti_layer(o)
        o = xception_block(o, filters, acti_layer, init)
        if res:
            l = Conv2DTranspose(filters, (1, 1), strides=(2, 2), kernel_initializer=init, padding='same')(m)
            o = add([l, o])
    else:
        filters = int(filters/2)
        for i in range(4):
            o = xception_block_middle(inputs, filters, acti_layer, init)
            #o = Dropout(0.2)(o)
            if res:
                o = add([inputs, o])
            inputs = o
    return o

def UNet_Xception(img_shape, num_classes=1, filters=64, depth=4, activation=lambda x: Activation('relu')(x),
         init='he_uniform', maxpool=True, res=True):
    
    i = Input(shape=img_shape)
    n = Conv2D(int(filters/2), (3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(i)
    n = activation(n)
    n = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(n)
    n = activation(n)
    o = level_block(n, filters*2, depth, depth, activation, init, maxpool, res)
    o = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(o)
    o = activation(o)
    o = Conv2DTranspose(int(filters/2), (3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(o)
    o = activation(o)
    o = Conv2D(num_classes, (1, 1), activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)
