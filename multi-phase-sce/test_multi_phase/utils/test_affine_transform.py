import numpy as np
import pytest
import tensorflow as tf
from typing import Callable

from common import RANDOM_SEED
from common.utils.affine_transformation import AffineTransform

PCT_CORRECT_THRESHOLD = 0.6
DEFAULT_IMG_SIZE = [4, 6, 8]
NUM_HOMOGENOUS_DIMS = 3

TEST_IMG_DIMS_2D = (
    [2, 64, 64, 1],
    [4, 64, 64, 1],
    [4, 128, 128, 1],
    [4, 128, 128, 3]
)

TEST_IMG_DIMS_3D = [
    [2, 32, 64, 64, 1],
    [4, 32, 64, 64, 1],
    [4, 64, 128, 128, 1],
    [4, 64, 128, 128, 3]
]


#-------------------------------------------------------------------------

@pytest.mark.parametrize("img_dims", [(2, 3), (5, 6), (512, 512)])
def test_coord_gen_2D(img_dims: tuple[int]) -> None:
    """ Test flat 2D image coordinates """

    # 2D case
    affine = AffineTransform(img_dims)
    exp_flat_coords_shape = (NUM_HOMOGENOUS_DIMS, np.prod(img_dims))
    assert affine.flat_coords.shape == exp_flat_coords_shape


#-------------------------------------------------------------------------

@pytest.mark.parametrize("img_dims", [(2, 3, 1), (5, 6, 4), (512, 512, 64)])
def test_coord_gen_3D(img_dims: tuple[int]) -> None:
    """ Test flat 3D image coordinates """

    affine = AffineTransform(img_dims)
    exp_flat_coords_shape = (NUM_HOMOGENOUS_DIMS, np.prod(img_dims[:-1]))
    assert affine.flat_coords.shape == exp_flat_coords_shape


#-------------------------------------------------------------------------
@pytest.mark.parametrize(
    "thetas",
    [
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, -1], [0, 1, 2]],
        [[1, 5, -1], [3, 1, 2]]
    ]
)
def test_transform_coords(thetas: list[list[int]]) -> None:
    """ Test transforming flat image coordinates """

    thetas = np.array(thetas).astype("float32")
    affine = AffineTransform(DEFAULT_IMG_SIZE)

    flat_coords = affine.flat_coords.numpy()
    new_coords = thetas @ flat_coords

    affine.transform_coords(1, tf.constant(thetas.reshape([1, -1])))
    X, Y = affine.mesh_coords

    assert np.isclose(X, new_coords[0, :]).all()
    assert np.isclose(Y, new_coords[1, :]).all()


#-------------------------------------------------------------------------

@pytest.mark.parametrize("mb_size", [2, 4])
def test_transform_coords_mb(mb_size: int) -> None:
    """ Test transforming minibatch of flat image coordinates """

    affine = AffineTransform(DEFAULT_IMG_SIZE)

    np.random.seed(RANDOM_SEED)
    thetas = np.random.randint(-5, 5, size=[mb_size, 2, 3]).astype("float32")
    np.random.seed()

    flat_coords = affine.flat_coords.numpy()
    new_coords = []

    for i in range(mb_size):
        new_coords.append(thetas[i, :, :] @ flat_coords)
    
    affine.transform_coords(mb_size, tf.constant(thetas.reshape(mb_size, -1)))

    X, Y = affine.mesh_coords
    ground_truth_X = np.hstack([c[0, :] for c in new_coords])
    ground_truth_Y = np.hstack([c[1, :] for c in new_coords])
    assert np.isclose(X, ground_truth_X).all()
    assert np.isclose(Y, ground_truth_Y).all()


#-------------------------------------------------------------------------

@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS_2D)
def test_flipping_2D(setup_flip_img: Callable) -> None:
    """ Test horizontal/vertical flipping for 2D images """

    # Create test images divided vertically and horizontally
    thetas, mb, gt_mb, img_dims = setup_flip_img
    affine = AffineTransform(img_dims[1:-1])
    new_mb = affine(mb, thetas)

    # Because of rounding errors at e.g. boundaries, we can't compare directly
    pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) \
                         / np.prod(new_mb.shape)
    assert pct_correct_voxels > PCT_CORRECT_THRESHOLD


#-------------------------------------------------------------------------

@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS_3D)
def test_flipping_3D(setup_flip_img: Callable) -> None:
    """ Test 2D horizontal/vertical flipping for 3D images """

    # Create test images divided vertically and horizontally
    thetas, mb, gt_mb, img_dims = setup_flip_img
    affine = AffineTransform(img_dims[1:-1])
    new_mb = affine(mb, thetas)

    # Because of rounding errors at e.g. boundaries, we can't compare directly
    pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) \
                         / np.prod(new_mb.shape)
    assert pct_correct_voxels > PCT_CORRECT_THRESHOLD


#-------------------------------------------------------------------------

@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS_2D)
def test_rotation_2D(setup_rotation_img: Callable) -> None:
    """ Test rotation for 2D images """

    # Create test images divided into quadrants
    thetas, mb, gt_mb, img_dims = setup_rotation_img
    affine = AffineTransform(img_dims[1:-1])
    new_mb = affine(mb, thetas)

    # Because of rounding errors at e.g. boundaries, we can't compare directly
    pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) \
                         / np.prod(new_mb.shape)
    assert pct_correct_voxels > PCT_CORRECT_THRESHOLD


#-------------------------------------------------------------------------

@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS_3D)
def test_rotation_3D(setup_rotation_img: Callable) -> None:
    """ Test 2D rotation for 3D images """

    # Create test images divided into quadrants
    thetas, mb, gt_mb, img_dims = setup_rotation_img
    affine = AffineTransform(img_dims[1:-1])
    new_mb = affine(mb, thetas)

    # Because of rounding errors at e.g. boundaries, we can't compare directly
    pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) \
                         / np.prod(new_mb.shape)
    assert pct_correct_voxels > PCT_CORRECT_THRESHOLD


#-------------------------------------------------------------------------

@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS_2D)
def test_scaling_2D(setup_scaling_img: Callable) -> None:
    """ Test scaling for 2D images """

    thetas, mb, gt_mb, img_dims = setup_scaling_img
    affine = AffineTransform(img_dims[1:-1])
    new_mb = affine(mb, thetas)

    # Because of rounding errors at e.g. boundaries, we can't compare directly
    pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) \
                            / np.prod(new_mb.shape)
    assert pct_correct_voxels > PCT_CORRECT_THRESHOLD


#-------------------------------------------------------------------------

@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS_3D)
def test_scaling_3D(setup_scaling_img: Callable) -> None:
    """ Test 2D scaling for 3D images """

    thetas, mb, gt_mb, img_dims = setup_scaling_img
    affine = AffineTransform(img_dims[1:-1])
    new_mb = affine(mb, thetas)

    # Because of rounding errors at e.g. boundaries, we can't compare directly
    pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) \
                         / np.prod(new_mb.shape)
    assert pct_correct_voxels > PCT_CORRECT_THRESHOLD
