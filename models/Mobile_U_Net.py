from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import UpSampling2D, concatenate, Conv2D, BatchNormalization, PReLU

def mobile_u_net(input_size):
    num_classes = 1

    # Build contracting path using pre-trained MobileNet
    contracting_path =  MobileNet(input_shape=(input_size, input_size, 3), include_top=False, weights=None)
    contracting_path.load_weights('weights/mobilenet_1_0_224_tf_no_top.h5')
    for layer in contracting_path.layers: layer.trainable = False

    # Build expanding path based on U-Net straucture
    center = contracting_path.output
    up4 = UpSampling2D((2, 2))(center)
    conv_pw_11_relu = contracting_path.get_layer(name='conv_pw_11_relu').output
    up4 = concatenate([up4, conv_pw_11_relu], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = PReLU()(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = PReLU()(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = PReLU()(up4)
    # 64

    up3 = UpSampling2D((2, 2))(up4)
    conv_pw_5_relu = contracting_path.get_layer(name='conv_pw_5_relu').output
    up3 = concatenate([up3, conv_pw_5_relu], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = PReLU()(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = PReLU()(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = PReLU()(up3)
    # 128

    up2 = UpSampling2D((2, 2))(up3)
    conv_pw_3_relu = contracting_path.get_layer(name='conv_pw_3_relu').output
    up2 = concatenate([up2, conv_pw_3_relu], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = PReLU()(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = PReLU()(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = PReLU()(up2)
    # 256

    up1 = UpSampling2D((2, 2))(up2)
    conv_pw_1_relu = contracting_path.get_layer(name='conv_pw_1_relu').output
    up1 = concatenate([up1, conv_pw_1_relu], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = PReLU()(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = PReLU()(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = PReLU()(up1)
    # 512

    up0 = UpSampling2D((2, 2))(up1)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = PReLU()(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = PReLU()(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = PReLU()(up0)
    # 1024

    classifier = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=contracting_path.input, outputs=classifier)

    return model
