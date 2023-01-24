import pytest
import tensorflow as tf

from multi_phase.networks.components.layer.layers import (
    DownBlock,
    UpBlock
)


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims",
    [
        ((2, 2, 4), [2, 32, 32, 3, 4]),
        ((2, 2, 2), [4, 32, 32, 6, 4]),
        ((2, 2, 1), [4, 32, 32, 12, 4])
    ]
)
def test_DownBlockNoPhase(
    strides: tuple[int],
    out_dims: list[int]
) -> None:
    """ Test DownBlock with no phase information """

    init = tf.keras.initializers.HeNormal()

    down = DownBlock(
        nc=4,
        weights=(4, 4, 4),
        strides=strides,
        initialiser=init,
        model="generator"
    )

    in_dims = [out_dims[0], 64, 64, 12, 4]
    img = tf.zeros(in_dims)
    x = down(x=img, t=None)
    assert x.shape == out_dims


#-------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["generator", "discriminator"])
def test_DownBlockPhase(model: str) -> None:
    """ Test DownBlock with phase information """

    init = tf.keras.initializers.HeNormal()

    down = DownBlock(
        nc=4,
        weights=(4, 4, 4),
        strides=(2, 2, 2),
        initialiser=init,
        model=model
    )

    if model == "generator":
        phase_info = tf.constant([1.0, 1.0, 2.0, 2.0])
    else:
        phase_info = tf.constant([1.0, 2.0])

    in_dims = [4, 64, 64, 12, 4]
    img = tf.zeros(in_dims)
    x = down(x=img, t=phase_info)
    assert x.shape == [4, 32, 32, 6, 4]


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims",
    [
        ((4, 2, 2), [2, 24, 64, 64, 4]),
        ((2, 2, 2), [4, 12, 64, 64, 8]),
        ((1, 2, 2), [4, 6, 64, 64, 8])
    ]
)
def test_UpBlockNoPhase(
    strides: list[int],
    out_dims: list[int]
) -> None:
    """ Test UpBlock without phase information """

    init = tf.keras.initializers.HeNormal()

    up = UpBlock(
        nc=out_dims[-1],
        weights=(4, 4, 4),
        strides=strides,
        initialiser=init
    )

    in_dims = [out_dims[0], 6, 32, 32, 4]
    img = tf.zeros(in_dims)
    skip = tf.zeros(out_dims)
    x = up(x=img, skip=skip, t=None)
    assert x.shape == out_dims


#-------------------------------------------------------------------------

def UpBlockPhase() -> None:
    """ Test UpBlock with phase information """

    init = tf.keras.initializers.HeNormal()

    up = UpBlock(
        nc=4,
        weights=(4, 4, 4),
        strides=(2, 2, 2),
        initialiser=init
    )

    in_dims = [4, 6, 32, 32, 4]
    out_dims = [4, 12, 64, 64, 4]
    img = tf.zeros(in_dims)
    skip = tf.zeros(out_dims)

    x = up(x=img, skip=skip, t=tf.constant([1.0, 1.0, 2.0, 2.0]))
    assert x.shape == out_dims
