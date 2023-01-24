import pytest
import tensorflow as tf

from multi_phase.networks.components.discriminator import Discriminator


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "depth,img_dims",
    [
        (3, [2, 16, 16, 8, 1]),
        (4, [4, 32, 32, 16, 1]),
        (5, [4, 64, 64, 16, 1])
    ]
)
def test_ImgDimsAssert(depth: int, img_dims: list[int]) -> None:
    """ Test assertion raised if too many layers """

    config = {
        "ndf": 4,
        "d_layers": depth,
        "img_dims": img_dims[1:-1],
        "d_phase_layers": []
    }

    init = tf.keras.initializers.Zeros()

    with pytest.raises(AssertionError):
        _ = Discriminator(init, config)


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "depth,img_dims,out_dims",
    [
        (1, [1, 16, 16, 16, 1], (1, 5, 5, 8, 1)),
        (2, [2, 16, 16, 8, 1], (2, 1, 1, 2, 1)),
        (2, [4, 32, 32, 16, 1], (4, 5, 5, 4, 1)),
        (2, [4, 64, 64, 16, 1], (4, 13, 13, 4, 1)),
        (3, [4, 64, 64, 16, 1], (4, 5, 5, 2, 1))
    ]
)
def test_NoPhaseOutput(
    depth: int,
    img_dims: list[int],
    out_dims: tuple[int]
) -> None:
    """ Test discriminator output is correct size """

    config = {
        "ndf": 4,
        "d_layers": depth,
        "img_dims": img_dims[1:-1],
        "d_phase_layers": []
    }

    init = tf.keras.initializers.Zeros()

    model = Discriminator(init, config)
    img = tf.zeros(img_dims)
    out = model(img)

    assert out.shape == out_dims


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phase_layers", [(["down_3"]), (["up_0", "down_1"])]
)
def test_PhaseLayerAssert(phase_layers: list[int]) -> None:
    """ Test phase layers checked for correctly """

    img_dims = [4, 64, 64, 16, 1]
    config = {
        "ndf": 4,
        "d_layers": 3,
        "img_dims": img_dims[1:-1],
        "d_phase_layers": phase_layers
    }

    init = tf.keras.initializers.Zeros()

    with pytest.raises(AssertionError):
        _ = Discriminator(init, config)


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phase_layers",
    [
        (["down_0"]),
        (["down_0", "down_1"]),
        (["down_0", "down_2"]),
        (["down_0", "down_1", "down_2"]),
    ]
)
def test_PhaseOutput(phase_layers: list[int]) -> None:
    """ Test discriminator output is correct size """

    img_dims = [4, 64, 64, 16, 1]
    out_dims = (4, 5, 5, 2, 1)
    config = {
        "ndf": 4,
        "d_layers": 3,
        "img_dims": img_dims[1:-1],
        "d_phase_layers": phase_layers
    }

    init = tf.keras.initializers.Zeros()

    model = Discriminator(init, config)
    img = tf.zeros(img_dims)
    out = model(img, tf.constant([1.0, 2.0]))

    assert out.shape == out_dims
