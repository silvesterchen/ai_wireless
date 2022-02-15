import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K



#This part realizes the quantization and dequantization operations.
#The output of the encoder must be the bitstream.

def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, 4:]).reshape(-1, Num_.shape[1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)
    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


# More details about the neural networks can be found in [1].
# [1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback,"
# in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
def Encoder(x,feedback_bits):
    B = 4
    with tf.compat.v1.variable_scope('Encoder'):
        # x = layers.Conv2D(2, 3, padding='SAME', activation='relu')(x)
        # x = layers.Conv2D(2, 3, padding='SAME', activation='relu')(x)
        x = layers.Conv2D(2, 3, padding='SAME', dilation_rate=(5,1))(x)
        # x = layers.Activation('relu')(x)
        x = layers.LeakyReLU(alpha=0.05)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv2D(2, 3, padding='SAME', dilation_rate=(5,1))(x)
        # x = layers.Activation('relu')(x)
        x = layers.LeakyReLU(alpha=0.05)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv2D(2, 3, padding='SAME', dilation_rate=(5,1))(x)
        # x = layers.Activation('relu')(x)
        x = layers.LeakyReLU(alpha=0.05)(x)
        x = layers.Dropout(rate=0.2)(x)
        # x = layers.Conv2D(2, 3, padding='SAME', dilation_rate=(5,1))(x)
        # x = layers.Activation('relu')(x)
        # x = layers.LeakyReLU(alpha=0.05)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=int(feedback_bits//B), activation='sigmoid')(x)
        encoder_output = QuantizationLayer(B)(x)
    return encoder_output

def conv_block(inputs, 
        neuron_num, 
        kernel_size,  
        use_bias, 
        padding= 'same',
        strides= (1, 1),
        with_conv_short_cut = False):
    conv1 = layers.Conv2D(
        neuron_num,
        kernel_size = kernel_size,
        activation= 'relu',
        strides= strides,
        use_bias= use_bias,
        padding= padding
    )(inputs)
    conv1 = layers.BatchNormalization(axis = 1)(conv1)

    conv2 = layers.Conv2D(
        neuron_num,
        kernel_size= kernel_size,
        activation= 'relu',
        use_bias= use_bias,
        padding= padding)(conv1)
    conv2 = layers.BatchNormalization(axis = 1)(conv2)

    if with_conv_short_cut:
        inputs = layers.Conv2D(
            neuron_num, 
            kernel_size= kernel_size,
            strides= strides,
            use_bias= use_bias,
            padding= padding
            )(inputs)
        return layers.add([inputs, conv2])

    else:
        return layers.add([inputs, conv2])


def Resnet34_Encoder(x, feedback_bits):
    B = 4

    # Define the converlutional block 1
    x = layers.Conv2D(64, kernel_size= (7, 7), strides= (1, 1), padding= 'same')(x)
    x = layers.BatchNormalization(axis= 1)(x)
    x = layers.MaxPooling2D(pool_size= (3, 3), strides= (2, 2), padding= 'same')(x)

    # Define the converlutional block 2
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)

    # Define the converlutional block 3
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)

    # Define the converlutional block 4
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)

    # Define the converltional block 5
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True, strides= (1, 1), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
    x = layers.AveragePooling2D(pool_size=(7, 7))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=int(feedback_bits // B), activation='sigmoid')(x)
    encoder_output = QuantizationLayer(B)(x)

    return encoder_output

def Decoder(x,feedback_bits):
    B = 4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.reshape(decoder_input, (-1, int(feedback_bits//B)))
    x = layers.Dense(32256)(x)
    x_ini = layers.Reshape((126, 128, 2))(x)
    for i in range(3):
        x = layers.Conv2D(8, 3, padding='SAME')(x_ini)
        # x = layers.Activation('relu')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(16, 3, padding='SAME')(x)
        # x = layers.BatchNormalization(axis=3)(x)
        # x = layers.Activation('relu')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(2, 3, padding='SAME')(x)
        # x = layers.Activation('relu')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x_ini = keras.layers.Add()([x_ini, x])
    decoder_output = layers.Conv2D(2, 3, padding='SAME',activation="sigmoid")(x_ini)
    return decoder_output


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


class ComplexNMSE(keras.losses.Loss):
    def __init__(self, name="complex_nmse"):
        super().__init__(name=name)

    def call(self, x, x_hat):
        x_real = tf.reshape(x[:, :, :, 0], (tf.shape(x)[0], -1))
        x_imag = tf.reshape(x[:, :, :, 1], (tf.shape(x)[0], -1))
        x_hat_real = tf.reshape(x_hat[:, :, :, 0], (tf.shape(x_hat)[0], -1))
        x_hat_imag = tf.reshape(x_hat[:, :, :, 1], (tf.shape(x_hat)[0], -1))
        abs_x = tf.math.sqrt((x_real - 0.5) ** 2 + (x_imag - 0.5) ** 2)
        abs_x_minus_x_hat = tf.math.sqrt((x_hat_real - x_real) ** 2 + (x_hat_imag - x_imag) ** 2)
        power = tf.reduce_sum(abs_x ** 2, axis=1)
        mse = tf.reduce_sum(abs_x_minus_x_hat ** 2, axis=1)
        nmse = tf.reduce_mean(mse / power)
        return nmse



# Return keywords of your own custom layers to ensure that model
# can be successfully loaded in test file.
def get_custom_objects():
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer}
