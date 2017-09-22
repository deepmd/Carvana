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

def level_block(inputs, filters, depth_count, depth, acti_layer, init, do, dbo, bn, mp, up, outputs):
    if depth > 0:
        n = conv_block(inputs, filters, acti_layer, init, bn, 0 if dbo else do)
        m = MaxPooling2D()(n) if mp else Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(n)
        m = level_block(m, 2*filters, depth_count, depth-1, acti_layer, init, do, dbo, bn, mp, up, outputs)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(filters, (2, 2), kernel_initializer=init, padding='same')(m)
            m = acti_layer(m)
            outputs.append(Conv2D(1, (1, 1), activation='sigmoid', 
                           name='aux_out{}'.format(depth_count-depth))(m))
        else:
            m = Conv2DTranspose(filters, (3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(m)
            m = acti_layer(m)
        m = Concatenate()([n, m])
        m = conv_block(m, filters, acti_layer, init, bn)
    else:
        m = conv_block(inputs, filters, acti_layer, init, bn, do)
        m = Dropout(0.2)(m)
    return m

def UNet_Aux(img_shape, num_classes=1, filters=64, depth=4, activation=lambda x: Activation('relu')(x),
         init='glorot_uniform', dropout=0.2, dropout_base_only=False, batchnorm=True, maxpool=True,
         upconv=True, auxiliaries=[False, False, True, True]):
    i = Input(shape=img_shape)
    all_outputs = []
    o = level_block(i, filters, depth, depth, activation, init, dropout, dropout_base_only, batchnorm, maxpool, upconv, all_outputs)
    o = Conv2D(num_classes, (1, 1), activation='sigmoid', name='main_out')(o)
    outputs = [out for aux, out in zip(auxiliaries, reversed(all_outputs)) if aux]
    outputs.append(o)
    return Model(inputs=i, outputs=outputs)
