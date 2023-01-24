import pytest
import tensorflow as tf

from multi_phase.networks.components.generator import Generator


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "depth,img_dims",
    [
        (5, [2, 16, 16, 8, 1]),
        (6, [4, 32, 32, 16, 1]),
        (7, [4, 64, 64, 16, 1])
    ]
)
def test_ImgDimsAssert(depth: int, img_dims: list[int]) -> None:
    """ Test assertion raised if too many layers """

    config = {
        "ngf": 4,
        "g_layers": depth,
        "img_dims": img_dims[1:-1],
        "g_phase_layers": []
    }

    init = tf.keras.initializers.Zeros()

    with pytest.raises(AssertionError):
        _ = Generator(init, config)


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "depth,img_dims",
    [
        (2, [2, 16, 16, 8, 1]),
        (2, [4, 32, 32, 16, 1]),
        (2, [4, 64, 64, 16, 1]),
        (3, [4, 64, 64, 16, 1])
    ]
)
def test_NoPhaseOutput(depth: int, img_dims: list[int]) -> None:
    """ Test generator output is correct size """

    config = {
        "ngf": 4,
        "g_layers": depth,
        "img_dims": img_dims[1:-1],
        "g_phase_layers": []
    }

    init = tf.keras.initializers.Zeros()

    model = Generator(init, config)
    img = tf.zeros(img_dims)
    out = model(img)

    assert out.shape == img.shape


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phase_layers", [(["down_3"]), (["up_4", "down_1"])]
)
def test_PhaseLayerAssert(phase_layers: list[int]) -> None:
    """ Test phase layers checked for correctly """

    img_dims = [4, 64, 64, 16, 1]
    config = {
        "ngf": 4,
        "g_layers": 3,
        "img_dims": img_dims[1:-1],
        "g_phase_layers": phase_layers
    }

    init = tf.keras.initializers.Zeros()

    with pytest.raises(AssertionError):
        _ = Generator(init, config)


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phase_layers",
    [
        (["down_0"]),
        (["down_0", "down_1"]),
        (["up_0", "up_1"]),
        (["down_0", "down_1", "up_1"]),
    ]
)
def test_PhaseOutput(phase_layers: list[int]) -> None:
    """ Test discriminator output is correct size """

    img_dims = [4, 64, 64, 16, 1]
    config = {
        "ngf": 4,
        "g_layers": 3,
        "img_dims": img_dims[1:-1],
        "g_phase_layers": phase_layers
    }

    init = tf.keras.initializers.Zeros()

    model = Generator(init, config)
    img = tf.zeros(img_dims)
    out = model(img, tf.constant([1.0, 1.0, 2.0, 2.0]))

    assert out.shape == img.shape
