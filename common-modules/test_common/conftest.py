import numpy as np
import pytest
import tensorflow as tf


#-------------------------------------------------------------------------

@pytest.fixture
def setup_flip_img(img_dims: list[int]) -> None:
    """ Create fixtures for flipping tests """

    vertical = np.zeros(img_dims[1:])
    vertical[0:img_dims[1] // 2, :, :] = 1
    horizontal = np.zeros(img_dims[1:])
    horizontal[:, 0:img_dims[2] // 2, :] = 1

    if img_dims[0] == 2:
        thetas = [
            np.array([-1, 0, 0, 0, 1, 0]),
            np.array([1, 0, 0, 0, -1, 0])
        ]

        mb = tf.convert_to_tensor(
            np.stack([horizontal, vertical], axis=0).astype("float32")
        )
        gt_mb = 1 - mb

    else:
        thetas = [
            np.array([-1, 0, 0, 0, 1, 0]),
            np.array([-1, 0, 0, 0, 1, 0]),
            np.array([1, 0, 0, 0, -1, 0]),
            np.array([1, 0, 0, 0, -1, 0])
        ]

        mb = tf.convert_to_tensor(
            np.stack(
                [
                    horizontal, vertical, horizontal, vertical
                ], axis=0
            ).astype("float32")
        )

        gt_mb = tf.convert_to_tensor(
            np.stack(
                [
                    1 - horizontal, vertical, horizontal, 1 - vertical
                ], axis=0
            ).astype("float32")
        )

    thetas = tf.convert_to_tensor(np.stack(thetas, axis=0).astype("float32"))

    return thetas, mb, gt_mb, img_dims


#-------------------------------------------------------------------------

@pytest.fixture
def setup_rotation_img(img_dims: list[int]) -> None:
    """ Create fixtures for rotation tests """

    base_img = np.zeros(img_dims[1:])
    base_img[0:img_dims[1] // 2, 0:img_dims[1] // 2, :] = 1
    base_img[-img_dims[1] // 2:, -img_dims[1] // 2:, :] = 1

    if img_dims[0] == 2:
        thetas = [
            np.array([0, 1, 0, -1, 0, 0]),
            np.array([-1, 0, 0, 0, -1, 0])
        ]

        mb = tf.convert_to_tensor(
            np.stack([base_img, base_img], axis=0).astype("float32")
        )
        gt_mb = tf.convert_to_tensor(
            np.stack([1 - base_img, base_img], axis=0).astype("float32")
        )

    else:
        thetas = [
            np.array([1, 0, 0, 0, 1, 0]),
            np.array([0, 1, 0, -1, 0, 0]),
            np.array([-1, 0, 0, 0, -1, 0]),
            np.array([0, -1, 0, 1, 0, 0])
        ]

        mb = tf.convert_to_tensor(
            np.stack([
                    base_img, base_img, base_img, base_img
                ], axis=0
            ).astype("float32")
        )

        gt_mb = tf.convert_to_tensor(
            np.stack(
                [
                    base_img, 1 - base_img, base_img, 1 - base_img
                ], axis=0
            ).astype("float32")
        )

    thetas = tf.convert_to_tensor(np.stack(thetas, axis=0).astype("float32"))

    return thetas, mb, gt_mb, img_dims


#-------------------------------------------------------------------------

@pytest.fixture
def setup_scaling_img(img_dims: list[int]) -> None:
    """ Create fixtures for scaling tests """

    H = img_dims[1]
    W = img_dims[2]

    mid_h = H // 2
    mid_w = W // 2
    img_small = np.zeros(img_dims[1:])

    if len(img_dims) == 4:
        img_small[
            mid_h - H // 4:mid_h + H // 4,
            mid_w - W // 4:mid_w + W // 4, :
        ] = 1
    else:
        img_small[
            mid_h - H // 4:mid_h + H // 4,
            mid_w - W // 4:mid_w + W // 4, :, :
        ] = 1

    img_large = np.ones(img_dims[1:])

    if img_dims[0] == 2:
        thetas = [
            np.array([2.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
        ]

        mb = tf.convert_to_tensor(
            np.stack([img_large, img_small], axis=0).astype("float32")
        )
        gt_mb = tf.convert_to_tensor(
            np.stack([img_small, img_large], axis=0).astype("float32")
        )

    else:
        thetas = [
            np.array([2.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0]),
            np.array([2.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
        ]

        mb = tf.convert_to_tensor(
            np.stack([
                    img_large, img_small, img_large, img_small
                ], axis=0
            ).astype("float32")
        )

        gt_mb = tf.convert_to_tensor(
            np.stack(
                [
                    img_small, img_large, img_small, img_large
                ], axis=0
            ).astype("float32")
        )

    thetas = tf.convert_to_tensor(np.stack(thetas, axis=0).astype("float32"))

    return thetas, mb, gt_mb, img_dims
