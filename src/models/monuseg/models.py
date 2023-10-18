from functools import partial

import tensorflow as tf
import numpy as np
from src.models.layers import DCTConv2D


def get_model(model_name="Unet",
              output_channels=1,
              loss=None,
              metrics=None,
              n_freq=4,
              lbd=False,
              coefs_l1reg=None,
              cosine_decay=True,
              run_eagerly=False,
              n_feature_maps=[8, 16, 32],
              last_activation="sigmoid",
              kernel_size_first=3,
              depth_wise=False,
              padding='VALID',
              depth_multiplier=1):
    model_dict = {
        "Unet":
        Unet,
        "DCTUnet":
        partial(
            DCTUnet,
            n_freq=n_freq,
            lbd=lbd,
            coefs_l1reg=coefs_l1reg,
        ),
    }

    if cosine_decay:
        lr = tf.keras.experimental.CosineDecayRestarts(
            1e-3,
            4500,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = model_dict[model_name](
        output_channels=output_channels,
        last_activation=last_activation,
        n_feature_maps=n_feature_maps,
        kernel_size_first=kernel_size_first,
        depth_wise=depth_wise,
        padding=padding,
        depth_multiplier=depth_multiplier,
    )

    model.compile(
        loss=[loss],
        optimizer=optimizer,
        metrics=metrics,
        run_eagerly=run_eagerly,
    )
    return model

class UnetBase(tf.keras.Model):

    def __init__(self,
                 *args,
                 kernel_size_first=3,
                 output_channels=3,
                 last_activation="softmax",
                 depth_wise=False,
                 n_feature_maps=None,
                 padding='VALID',
                 depth_multiplier=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size = 5  # Unet default: 5
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        if n_feature_maps is None:
            n_feature_maps = [8, 16, 32]
        self.stem = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_feature_maps[0]/4,
                                   kernel_size,
                                   activation="relu",
                                   padding='SAME'),
        ])


        self.conv_1 = self.get_conv_block(n_feature_maps[0],
                                          kernel_size_first,
                                          depth_wise=depth_wise,
                                          dct=True
                                          )
        self.conv_2 = self.get_conv_block(n_feature_maps[1], kernel_size)
        self.conv_3 = self.get_conv_block(n_feature_maps[2], kernel_size)
        self.conv_4 = self.get_conv_block(n_feature_maps[1], kernel_size)
        self.conv_5 = self.get_conv_block(n_feature_maps[0], kernel_size)

        self.down_sampling_1 = tf.keras.layers.MaxPool2D()
        self.down_sampling_2 = tf.keras.layers.MaxPool2D()
        if kernel_size == 3:
            self.crop_1 = tf.keras.layers.Cropping2D(cropping=(2, 2))
            self.crop_2 = tf.keras.layers.Cropping2D(cropping=(8, 8))
        elif kernel_size == 5:
            self.crop_1 = tf.keras.layers.Cropping2D(cropping=(4, 4)) 
            self.crop_2 = tf.keras.layers.Cropping2D(cropping=(16, 16)) 

        self.upsampling_1 = tf.keras.Sequential([
            #tf.keras.layers.Conv2D(n_feature_maps[1], 1, padding='SAME'),
            tf.keras.layers.UpSampling2D(interpolation="bilinear")
        ], )
        self.upsampling_2 = tf.keras.Sequential([
            #tf.keras.layers.Conv2D(n_feature_maps[0], 1, padding='SAME'),
            tf.keras.layers.UpSampling2D(interpolation="bilinear")
        ], )
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation=last_activation,
                                   padding='SAME'),
        ])

    def get_conv_block(self, filters, ksize, depth_wise, dct=False):
        raise NotImplementedError()

    def call(self, inputs, training=None):
        x0 = self.stem(inputs, training=training)
        x1 = self.conv_1(x0, training=training)
        x2 = self.conv_2(self.down_sampling_1(x1), training=training)
        x3 = self.conv_3(self.down_sampling_2(x2), training=training)
        x4 = self.conv_4(
            tf.concat([
                self.upsampling_1(x3),
                self.crop_1(x2),
            ], axis=-1),
            training=training,
        )
        x5 = self.conv_5(
            tf.concat([
                self.upsampling_2(x4),
                self.crop_2(x1),
            ], axis=-1),
            training=training,
        )
        return self.last(x5, training=training)


class Unet(UnetBase):
    def get_conv_block(self,
                       filters,
                       ksize,
                       depth_wise=False,
                       dct=False):
        block = tf.keras.Sequential()
        if depth_wise:
            block.add(
                tf.keras.layers.DepthwiseConv2D(
                    ksize,
                    padding=self.padding,
                    depth_multiplier=self.depth_multiplier,
                    activation="relu"))
            block.add(tf.keras.layers.Conv2D(
                filters, 1))  # project to number of filters.
        else:
            block.add(
                tf.keras.layers.Conv2D(filters, ksize, padding=self.padding))
        block.add(tf.keras.layers.BatchNormalization())
        block.add(tf.keras.layers.Activation("relu"))
        return block

class DCTUnet(UnetBase):

    def __init__(self,
                 *args,
                 output_channels=1,
                 n_freq=4,
                 lbd=False,
                 coefs_l1reg=None,
                 **kwargs):
        self.n_freq = n_freq
        self.lbd = lbd
        self.coefs_l1reg = coefs_l1reg
        super().__init__(*args, output_channels=output_channels, **kwargs)

    def get_conv_block(self,
                       filters,
                       ksize,
                       depth_wise=False,
                       dct=False):
        block = tf.keras.Sequential()
        if dct:
            block.add(
                DCTConv2D(
                    filters,
                    ksize,
                    self.n_freq,
                    self.lbd,
                    self.coefs_l1reg,
                    padding="VALID",
                ))
        else:
            block.add(
                tf.keras.layers.Conv2D(filters, ksize, padding=self.padding))
        block.add(tf.keras.layers.BatchNormalization())
        block.add(tf.keras.layers.Activation("relu"))
        return block

