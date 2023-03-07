import tensorflow as tf
from typing import Tuple, Union

DROPOUT_PROB = 0.5
EPSILON = 1e-12


# -------------------------------------------------------------------------


class InstanceNorm(tf.keras.layers.Layer):
    """Instance normalisation layer for Pix2Pix generator"""

    def __init__(self, name: Union[str, None] = None):
        super().__init__(name=name)
        self.epsilon = EPSILON

    def build(self, input_shape):
        self.beta = self.add_weight(
            "beta",
            shape=[1, 1, 1, 1, input_shape[-1]],
            initializer="zeros",
            trainable=True,
        )
        self.gamma = self.add_weight(
            "gamma",
            shape=[1, 1, 1, 1, input_shape[-1]],
            initializer="ones",
            trainable=True,
        )

    def call(self, x, training: Union[bool, None] = None):
        mu = tf.math.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        sigma = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)

        return (x - mu) / (sigma + self.epsilon) * self.gamma + self.beta


# -------------------------------------------------------------------------


class DownBlock(tf.keras.layers.Layer):

    """Down-sampling convolutional block for Pix2pix discriminator
    and generator
    :param nc: number of feature maps
    :param strides: tuple of strides e.g. (2, 2, 1)
    :param initialiser: e.g. keras.initializers.RandomNormal
    :param model: model type "generator"/"discriminator"
    :param batch_norm: True/False
    """

    def __init__(
        self,
        nc: int,
        weights: Tuple[int],
        strides: Tuple[int],
        initialiser: tf.keras.initializers.Initializer,
        model: str,
        batch_norm: bool = True,
        name: Union[str, None] = None,
    ):
        super().__init__(name=name)
        self.batch_norm = batch_norm
        bias = not batch_norm

        self.conv = tf.keras.layers.Conv3D(
            nc,
            weights,
            strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            use_bias=bias,
            name="conv",
        )

        # Normalisation
        if batch_norm and model == "generator":
            self.bn = InstanceNorm(name="instancenorm")
        elif batch_norm and model == "discriminator":
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")

        # Discriminator requires x2 tiled phase information
        # as discriminator minibatch is twice that of generator
        if model == "generator":
            self.phase_reps = 1
        else:
            self.phase_reps = 2

    def call(self, x: tf.Tensor, t: Union[tf.Tensor, None] = None):
        """Layer call method. NB: training argument not used as
        discriminator not used in inference mode
        :param x: feature map from previous layer
        :param t: phase information
        """

        # Tile phase information and concatenate to feature map
        if t is not None:
            tiled_phase = tf.tile(
                tf.reshape(t, [-1, 1, 1, 1, 1]),
                [self.phase_reps] + x.shape[1:4] + [1],
                name="time_tile",
            )
            x = tf.concat([x, tiled_phase], axis=4, name="time_concat")

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x, training=True)

        return tf.nn.leaky_relu(x, alpha=0.2, name="l_relu")


# -------------------------------------------------------------------------


class UpBlock(tf.keras.layers.Layer):

    """Up-sampling convolutional block for Pix2pix generator
    :param nc: number of feature maps
    :param strides: tuple of strides e.g. (2, 2, 1)
    :param initialiser: e.g. keras.initializers.RandomNormal
    :param instance_norm: Use instance norm True/False
    :param dropout: Use dropout True/False
    """

    def __init__(
        self,
        nc: int,
        weights: Tuple[int],
        strides: Tuple[int],
        initialiser: tf.keras.initializers.Initializer,
        instance_norm: bool = True,
        dropout: bool = False,
        name: Union[str, None] = None,
    ):
        super().__init__(name=name)
        self.instance_norm = instance_norm
        bias = not instance_norm
        self.dropout = dropout

        self.tconv = tf.keras.layers.Conv3DTranspose(
            nc,
            weights,
            strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            use_bias=bias,
            name="tconv",
        )
        self.conv = tf.keras.layers.Conv3D(
            nc,
            weights,
            strides=(1, 1, 1),
            padding="same",
            kernel_initializer=initialiser,
            use_bias=bias,
            name="conv",
        )

        # Instance normalisation
        if instance_norm:
            self.bn1 = InstanceNorm(name="instancenorm1")
            self.bn2 = InstanceNorm(name="instancenorm2")

        if dropout:
            self.dropout1 = tf.keras.layers.Dropout(DROPOUT_PROB, name="dropout1")
            self.dropout2 = tf.keras.layers.Dropout(DROPOUT_PROB, name="dropout2")

        self.concat = tf.keras.layers.Concatenate(name="concat")

    def call(self, x: tf.Tensor, skip: tf.Tensor, t: Union[tf.Tensor, None] = None):
        """Layer call method. NB: training argument not used with
        instance norm and dropout is used during inference
        :param x: feature map from previous layer
        :param skip: feature map from U-Net skip layer
        :param t: phase information
        """

        # Tile phase information and concatenate to feature map
        if t is not None:
            tiled_phase = tf.tile(
                tf.reshape(t, [-1, 1, 1, 1, 1]), [1] + x.shape[1:4] + [1], "time_tile"
            )
            x = tf.concat([x, tiled_phase], axis=4, name="time_concat")

        x = self.tconv(x)

        if self.instance_norm:
            x = self.bn1(x)

        if self.dropout:
            x = self.dropout1(x, training=True)

        x = tf.nn.relu(x)
        x = self.concat([x, skip])
        x = self.conv(x)

        if self.instance_norm:
            x = self.bn2(x)

        if self.dropout:
            x = self.dropout2(x, training=True)

        return tf.nn.relu(x)
