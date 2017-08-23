from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Activation, BatchNormalization, Dropout, Reshape
from keras.regularizers import l2

def bn_nonlin_conv(inputs, filters, sz, acti, init, reg, stride=1):
    n = BatchNormalization()(inputs)
    n = Activation(acti)(n)
    n = Conv2D(filters, (sz, sz), strides=(stride, stride), activation=acti, kernel_initializer=init, 
               kernel_regularizer=reg, padding='same')(n)
    return n

def dense_block(layers, inputs, growth_rate, acti, init, reg, do=0):
    outputs = []
    for i in range(layers):
        n = bn_nonlin_conv(inputs, growth_rate, 3, acti, init, reg)
        n = Dropout(do)(n) if do else n
        outputs.append(n)
        inputs = Concatenate()([inputs, n])
    return Concatenate()(outputs)

def transition_down(inputs, acti, init, reg, mp, do=0):
    filters = inputs.get_shape().as_list()[-1]
    if mp:
        n = bn_nonlin_conv(inputs, filters, 1, acti, init, reg)
        n = Dropout(do)(n) if do else n
        n = MaxPooling2D()(n)
    else:
        n = bn_nonlin_conv(inputs, filters, 1, acti, init, reg, stride=2)
    return n

def transition_up(inputs, acti, init, reg, up):
    filters = inputs.get_shape().as_list()[-1]
    if up:
        n = UpSampling2D()(inputs)
        n = Conv2D(filters, (2, 2), activation=acti, kernel_initializer=init, kernel_regularizer=reg, padding='same')(n)
    else:
        n = Conv2DTranspose(filters, (3, 3), strides=(2, 2), activation=acti, kernel_initializer=init, kernel_regularizer=reg, padding='same')(inputs)
    return n

def level_block(inputs, growth_rate, depth, layers_per_block, acti, init, reg, do, mp, up):
    if depth > 0:
        n = dense_block(layers_per_block[depth], inputs, growth_rate, acti, init, reg, do)
        n = Concatenate()([inputs, n])
        m = transition_down(n, acti, init, reg, mp, do)
        m = level_block(m, growth_rate, depth-1, layers_per_block, acti, init, reg, do, mp, up)[0]
        m = transition_up(m, acti, init, reg, up)
        m = Concatenate()([n, m])
        o = dense_block(layers_per_block[depth], m, growth_rate, acti, init, reg, do)
    else:
        m = None
        o = dense_block(layers_per_block[0], inputs, growth_rate, acti, init, reg, do)
    return o, m

def Tiramisu(img_shape, num_classes=1, filters=48, growth_rate=16, depth=5, layers_per_block=[4,5,7,10,12,15],
             activation='relu', init='he_uniform', dropout=0.2, regularizer=l2(1e-4), maxpool=False, upconv=False):
    '''
    The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
    (http://arxiv.org/abs/1611.09326)
    ---
    img_shape: (height, width, channels)
    num_classes: number of output channels
    filters: number of filters of the first conv
    growth_rate: growth rate used in dense blocks
    depth: zero indexed depth of the U-structure
    layers_per_block: number of convolutions in each dense blocks (its size must conform to depth)
    activation: activation function after convolutions
    init: kernels initialization function
    dropout: amount of dropout in the contracting part
    regulaizer: kernel regularizer function  default: l2(1e-4)
    maxpool: use strided conv instead of maxpooling if false
    upconv: use transposed conv instead of upsamping + conv if false
    '''
    if layers_per_block is None or len(layers_per_block) != depth+1:
        raise AssertionError("Size of the layers_per_block does not conform to depth value.")
    layers_per_block = list(reversed(layers_per_block))
    i = Input(shape=img_shape)
    o = Conv2D(filters, (3, 3), activation=activation, kernel_initializer=init, kernel_regularizer=regularizer, padding='same')(i)
    o,p = level_block(o, growth_rate, depth, layers_per_block, activation, init, regularizer, dropout, maxpool, upconv)
    o = Concatenate()([o, p]) #in original paper the second to last features are concatenated to final features
    if (num_classes == 1):
        o = Conv2D(num_classes, (1, 1), activation='sigmoid')(o)
    else :
        o = Conv2D(num_classes, (1, 1), activation=activation, kernel_initializer=init, kernel_regularizer=regularizer, padding='same')(o)
        o = Reshape((-1, num_classes))(o)
        o = Activation('softmax')(o)
        o = Reshape((img_shape[0], img_shape[1], num_classes))(o)
    return Model(inputs=i, outputs=o)
