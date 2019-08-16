from keras.models import *
from keras.layers import *
import keras.backend as K

HUBER_DELTA = 0.5


def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def l2_loss(y_true, y_pred):
    loss_a = K.square(y_true[..., 0] - y_pred[..., 0])
    loss_b = K.square(y_true[..., 1] - y_pred[..., 1])
    loss = (K.mean(loss_a) + K.mean(loss_b)) / 2
    return loss


def l1_loss(y_true, y_pred):
    loss_a = K.abs(y_true[..., 0] - y_pred[..., 0])
    loss_b = K.abs(y_true[..., 1] - y_pred[..., 1])
    loss = (K.mean(loss_a) + K.mean(loss_b)) / 2
    return loss


def predict(y_pred, anchors):
    num_anchors = anchors.shape[0]
    pred1 = y_pred[..., :num_anchors] + anchors
    scores1 = y_pred[..., num_anchors * 2:num_anchors * 3]
    a = np.sum(pred1 * scores1, axis=-1, keepdims=True)

    pred2 = y_pred[..., num_anchors:num_anchors * 2] + anchors
    scores2 = y_pred[..., num_anchors * 3:num_anchors * 4]
    b = np.sum(pred2 * scores2, axis=-1, keepdims=True)

    return np.concatenate([a, b], axis=-1)


def ColorUNet(input_size=(128, 128, 1), reg=None, init='glorot_uniform', alpha=0., large=True):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(inputs)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(conv1)
    batch1 = BatchNormalization()(conv1)
    batch1 = LeakyReLU(alpha=alpha)(batch1)
    conv2 = Conv2D(128, 3, strides=2, padding='same', kernel_initializer=init,
                   kernel_regularizer=reg)(batch1)
    conv2 = LeakyReLU(alpha=alpha)(conv2)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(conv2)
    batch2 = BatchNormalization()(conv2)
    batch2 = LeakyReLU(alpha=alpha)(batch2)
    conv3 = Conv2D(256, 3, strides=2, padding='same', kernel_initializer=init,
                   kernel_regularizer=reg)(batch2)
    conv3 = LeakyReLU(alpha=alpha)(conv3)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(conv3)
    batch3 = BatchNormalization()(conv3)
    batch3 = LeakyReLU(alpha=alpha)(batch3)
    conv4 = Conv2D(512, 3, strides=2, padding='same', kernel_initializer=init,
                   kernel_regularizer=reg)(batch3)
    conv4 = LeakyReLU(alpha=alpha)(conv4)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(conv4)
    batch4 = BatchNormalization()(conv4)
    batch4 = LeakyReLU(alpha=alpha)(batch4)
    if large:
        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(batch4)
        conv5 = LeakyReLU(alpha=alpha)(conv5)
        conv5 = Conv2D(1024, 3, strides=2, padding='same', kernel_initializer=init, kernel_regularizer=reg,
                       use_bias=False)(conv5)
        batch5 = BatchNormalization()(conv5)
        batch5 = LeakyReLU(alpha=alpha)(batch5)

        up6 = Conv2DTranspose(512, 2, strides=2, padding='same', kernel_initializer=init,
                              kernel_regularizer=reg)(batch5)
        up6 = LeakyReLU(alpha=alpha)(up6)
        merge6 = concatenate([batch4, up6], axis=3)
        conv6 = Conv2D(512, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(merge6)
        conv6 = LeakyReLU(alpha=alpha)(conv6)
        conv6 = Conv2D(512, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(conv6)
        batch6 = BatchNormalization()(conv6)
        batch6 = LeakyReLU(alpha=alpha)(batch6)
        up7 = Conv2DTranspose(256, 2, strides=2, padding='same', kernel_initializer=init,
                              kernel_regularizer=reg)(batch6)
    else:
        up7 = Conv2DTranspose(256, 2, strides=2, padding='same', kernel_initializer=init,
                              kernel_regularizer=reg)(batch4)
    up7 = LeakyReLU(alpha=alpha)(up7)
    merge7 = concatenate([batch3, up7], axis=3)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(merge7)
    conv7 = LeakyReLU(alpha=alpha)(conv7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(conv7)
    batch7 = BatchNormalization()(conv7)
    batch7 = LeakyReLU(alpha=alpha)(batch7)

    up8 = Conv2DTranspose(128, 2, strides=2, padding='same', kernel_initializer=init,
                          kernel_regularizer=reg)(batch7)
    up8 = LeakyReLU(alpha=alpha)(up8)
    merge8 = concatenate([batch2, up8], axis=3)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(merge8)
    conv8 = LeakyReLU(alpha=alpha)(conv8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(conv8)
    batch9 = BatchNormalization()(conv8)
    batch9 = LeakyReLU(alpha=alpha)(batch9)

    up9 = Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer=init,
                          kernel_regularizer=reg)(batch9)
    up9 = LeakyReLU(alpha=alpha)(up9)
    merge9 = concatenate([batch1, up9], axis=3)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg)(merge9)
    conv9 = LeakyReLU(alpha=alpha)(conv9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer=init, kernel_regularizer=reg, use_bias=False)(conv9)
    batch9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=alpha)(batch9)

    conv9 = Conv2D(2, 1, activation='tanh', kernel_initializer='glorot_normal')(conv9)

    model = Model(inputs=[inputs], outputs=[conv9])

    return model
