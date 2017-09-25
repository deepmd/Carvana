from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Activation
from keras.layers.advanced_activations import PReLU, ELU

def conv_block(inputs, filters, init, n):
    m = inputs
    #acti_layer = lambda x: ELU()(x)
    acti_layer = Activation('relu')
    for i in range(n):
        #acti_layer = acti_layer if i < n-1 else lambda x: PReLU()(x)
        m = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(m)
        m = acti_layer(m)
        m = BatchNormalization()(m)
    return m

def level_block(inputs, filters, depth, init):
    if depth > 1:
        n = conv_block(inputs, filters[depth-1], init, 2)
        m = MaxPooling2D()(n)
        seg, con = level_block(m, filters, depth-1, init)

        # Segmentation expanding path
        seg = UpSampling2D()(seg)
        seg = Concatenate()([n, seg])
        seg = conv_block(seg, filters[min(depth, len(filters)-1)], init, 3)

        # Countour expanding path
        con = UpSampling2D()(con)
        con = Concatenate()([n, con])
        con = conv_block(con, filters[min(depth, len(filters)-1)], init, 3)
    else:
        seg = conv_block(inputs, filters[depth], init, 1)
        seg = Dropout(0.2)(seg)
        con = seg
    return (seg, con)

def DCAN_Heng(img_shape, num_classes=1, filters=[24,64,128,256,512,768,768], init='he_uniform'):
    filters = list(reversed(filters))
    i = Input(shape=img_shape)
    (seg, con) = level_block(i, filters, len(filters), init)
    seg_out = Conv2D(num_classes, (1, 1), activation='sigmoid', name='seg_out')(seg)
    con_out = Conv2D(num_classes, (1, 1), activation='sigmoid', name='contour_out')(con)
    return Model(inputs=i, outputs=[seg_out, con_out])
