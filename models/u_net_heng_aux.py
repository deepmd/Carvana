from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Activation
from keras.layers.advanced_activations import PReLU, ELU

def conv_block(inputs, filters, init, n):
    m = inputs
    acti_layer = lambda x: ELU()(x)
    for i in range(n):
        acti_layer = acti_layer if i < n-1 else lambda x: PReLU()(x)
        m = Conv2D(filters, (3, 3), kernel_initializer=init, padding='same')(m)
        m = BatchNormalization()(m)
        m = acti_layer(m)
    return m

def level_block(inputs, filters, depth, init, outputs):
    if depth > 1:
        n = conv_block(inputs, filters[depth-1], init, 2)
        m = MaxPooling2D()(n)
        m = level_block(m, filters, depth-1, init, outputs)
        m = UpSampling2D()(m)
        m = Concatenate()([n, m])
        m = conv_block(m, filters[min(depth, len(filters)-1)], init, 3)
        outputs.append(Conv2D(1, (1, 1), activation='sigmoid', 
                              name='aux_out{}'.format(len(filters)-depth))(m))
    else:
        m = conv_block(inputs, filters[depth-1], init, 1)
        m = Dropout(0.2)(m)
    return m

def UNet_Heng_Aux(img_shape, filters=[24,64,128,256,512,768,768], init='he_uniform',
                  auxiliaries=[False, False, False, False, True, True]):
    if (len(auxiliaries) != len(filters)-1):
        raise ValueError('The length of auxiliaries is not correct!')
    filters = list(reversed(filters))
    i = Input(shape=img_shape)
    all_outputs = []
    o = level_block(i, filters, len(filters), init, all_outputs)
    o = Conv2D(1, (1, 1), activation='sigmoid', name='main_out')(o)
    outputs = [out for aux, out in zip(auxiliaries, reversed(all_outputs)) if aux]
    outputs.append(o)
    return Model(inputs=i, outputs=outputs)
