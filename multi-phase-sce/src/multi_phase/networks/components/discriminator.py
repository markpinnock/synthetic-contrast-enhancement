import numpy as np
import tensorflow as tf
from typing import Union

from .layer.layers import DownBlock


class Discriminator(tf.keras.Model):

    """PatchGAN discriminator for Pix2pix and CycleGAN
    :param initialiser: e.g. keras.initializers.RandomNormal
    :param config: configuration dict
    """

    def __init__(
        self,
        initialiser: tf.keras.initializers.Initializer,
        config: dict,
        name: Union[str, None] = None,
    ):
        super().__init__(name=name)

        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]]) / 4))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        nc = config["ndf"]  # Number first layer channels
        num_layers = config["d_layers"]

        # Get layers incorporating phase information
        self.phase_layers = config["d_phase_layers"]

        assert (
            num_layers <= max_num_layers and num_layers > 0
        ), f"Maximum number of discriminator layers: {max_num_layers}"
        self.encoder = []

        # PatchGAN i.e. NxN receptive field for N > 1
        batch_norm = False

        for i in range(0, num_layers):
            if i > 0:
                batch_norm = True
            channels = tf.minimum(nc * 2**i, 512)

            if i > max_z_downsample:
                strides = (2, 2, 1)
                kernel = (4, 4, 2)
            else:
                strides = (2, 2, 2)
                kernel = (4, 4, 4)

            self.encoder.append(
                DownBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    model="discriminator",
                    batch_norm=batch_norm,
                    name=f"down_{i}",
                )
            )

        self.encoder.append(
            tf.keras.layers.Conv3D(
                1,
                (4, 4, 1),
                (1, 1, 1),
                padding="valid",
                kernel_initializer=initialiser,
                name="output",
            )
        )

        layer_names = [layer.name for layer in self.encoder]

        for phase_input in self.phase_layers:
            assert phase_input in layer_names, (phase_input, layer_names)

    def call(self, x: tf.Tensor, phase: tf.Tensor = None) -> tf.Tensor:
        """Discriminator call method
        :param x: input image volume
        :param phase: phase information
        """

        for conv in self.encoder:
            if conv.name in self.phase_layers:
                x = conv(x, phase, training=True)
            else:
                x = conv(x, training=True)

        return x
