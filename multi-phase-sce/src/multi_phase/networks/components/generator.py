import numpy as np
import tensorflow as tf

from .layer.layers import DownBlock, UpBlock


class Generator(tf.keras.Model):

    """Generator for Pix2pix and CycleGAN. Can also be used as plain
    U-Net with mode=="UNet"
    :param initialiser: e.g. keras.initializers.RandomNormal
    :param config: configuration dict
    :param mode: "GAN" or "UNet"
    """

    def __init__(
        self,
        initialiser: tf.keras.initializers.Initializer,
        config: dict,
        mode: str = "GAN",
        name: str | None = None,
    ):
        super().__init__(name=name)

        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]])))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        nc = config["ngf"]  # Number first layer channels
        num_layers = config["g_layers"]

        # Get layers incorporating phase information
        if config["g_phase_layers"] is not None:
            self.phase_layers = config["g_phase_layers"]
        else:
            self.phase_layers = []

        assert (
            num_layers <= max_num_layers and num_layers > 1
        ), f"Maximum number of generator layers: {max_num_layers}"
        self.encoder = []

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        for i in range(0, num_layers - 1):
            channels = np.min([nc * 2**i, 512])

            if i >= max_z_downsample - 1:
                strides = (2, 2, 1)
                kernel = (4, 4, 2)
            else:
                strides = (2, 2, 2)
                kernel = (4, 4, 4)

            cache["channels"].append(channels)
            cache["strides"].append(strides)
            cache["kernels"].append(kernel)

            self.encoder.append(
                DownBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    model="generator",
                    batch_norm=True,
                    name=f"down_{i}",
                )
            )

        self.bottom_layer = DownBlock(
            channels,
            kernel,
            strides,
            initialiser=initialiser,
            model="generator",
            batch_norm=True,
            name="bottom",
        )

        cache["strides"].append(strides)
        cache["kernels"].append(kernel)

        cache["channels"].reverse()
        cache["kernels"].reverse()
        cache["strides"].reverse()

        self.decoder = []

        # If mode == UNet, dropout is switched off, else dropout used as in Pix2Pix
        if mode == "GAN":
            dropout = True
        elif mode == "UNet":
            dropout = False
        else:
            raise ValueError

        for i in range(0, num_layers - 1):
            if i > 2:
                dropout = False
            channels = cache["channels"][i]
            strides = cache["strides"][i]
            kernel = cache["kernels"][i]

            self.decoder.append(
                UpBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    instance_norm=True,
                    dropout=dropout,
                    name=f"up_{i}",
                )
            )

        self.final_layer = tf.keras.layers.Conv3DTranspose(
            1,
            (4, 4, 4),
            (2, 2, 2),
            padding="same",
            activation="linear",
            kernel_initializer=initialiser,
            name="output",
        )

        layer_names = (
            [layer.name for layer in self.encoder]
            + ["bottom"]
            + [layer.name for layer in self.decoder]
        )

        for phase_input in self.phase_layers:
            assert phase_input in layer_names, (phase_input, layer_names)

    def call(self, x: tf.Tensor, phase: tf.Tensor = None) -> tf.Tensor:
        """Generator call method
        :param x: input image volume
        :param phase: phase information
        """

        skip_layers = []

        for conv in self.encoder:
            if conv.name in self.phase_layers:
                x = conv(x, phase, training=True)
            else:
                x = conv(x, training=True)

            skip_layers.append(x)

        if self.bottom_layer.name in self.phase_layers:
            x = self.bottom_layer(x, phase, training=True)
        else:
            x = self.bottom_layer(x, training=True)

        x = tf.nn.relu(x)
        skip_layers.reverse()

        for skip, tconv in zip(skip_layers, self.decoder):
            if tconv.name in self.phase_layers:
                x = tconv(x, skip, phase, training=True)
            else:
                x = tconv(x, skip, training=True)

        if self.final_layer.name in self.phase_layers:
            x = self.final_layer(x, phase, training=True)
        else:
            x = self.final_layer(x, training=True)

        return x
