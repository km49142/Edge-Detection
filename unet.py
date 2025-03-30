### unet.py
import tensorflow as tf

def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(f)
    return f, p

def decoder_block(x, skip, filters):
    us = tf.keras.layers.UpSampling2D((2, 2))(x)
    concat = tf.keras.layers.Concatenate()([us, skip])
    return conv_block(concat, filters)

def build_unet(input_shape=(128, 128, 1)):
    inputs = tf.keras.Input(input_shape)

    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)

    bottleneck = conv_block(p3, 512)

    d1 = decoder_block(bottleneck, f3, 256)
    d2 = decoder_block(d1, f2, 128)
    d3 = decoder_block(d2, f1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d3)

    return tf.keras.Model(inputs, outputs)