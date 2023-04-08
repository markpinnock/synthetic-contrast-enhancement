from typing import Callable, List, Tuple

import numpy as np
import pytest
import tensorflow as tf
from multi_phase import RANDOM_SEED
from multi_phase.utils.augmentation import DiffAug, StdAug

tf.random.set_seed(RANDOM_SEED)

PCT_CORRECT_THRESHOLD = 0.6
DEFAULT_IMG_SIZE = [4, 6, 8]
NUM_HOMOGENOUS_DIMS = 3

TEST_IMG_DIMS = [
    [2, 32, 32, 8, 1],
    [4, 32, 32, 8, 1],
    [4, 64, 64, 16, 1],
]

STD_AUG_CONFIG = {
    "augmentation": {
        "flip_prob": 1.0,
        "rotation": 45.0,
        "scale": [0.8, 1.6],
        "shear": 15.0,
        "translate": [0.25, 0.25],
    },
}

DIFF_AUG_CONFIG = {
    "augmentation": {
        "colour": True,
        "translation": True,
        "cutout": True,
    },
}


# -------------------------------------------------------------------------


@pytest.fixture
def create_test_img(img_dims: List[int]) -> Tuple[tf.Tensor, List[int]]:
    """Create test images"""

    img = np.zeros(img_dims, dtype="float32")
    img[:, 0 : img.shape[2] // 2, 0 : img.shape[3] // 2, :, :] = 1
    img[:, -img.shape[2] // 2 :, -img.shape[3] // 2 :, :, :] = 1

    return tf.convert_to_tensor(img), img_dims


# -------------------------------------------------------------------------


@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS)
def test_source_target_same_StdAug(create_test_img: Callable) -> None:
    """Test that source, target and segmentations
    are augmented identically for standard augmentation
    """

    source, _ = create_test_img
    target, _ = create_test_img
    seg, img_dims = create_test_img

    config = STD_AUG_CONFIG
    config["hyperparameters"] = {"img_dims": img_dims[1:-1]}
    aug = StdAug(config)

    (aug_source, aug_target), aug_seg = aug([source, target], seg)

    assert np.isclose(aug_source, aug_target).all()
    assert np.isclose(aug_source, aug_seg).all()


# -------------------------------------------------------------------------


@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS)
def test_mb_different_StdAug(create_test_img: Callable) -> None:
    """Test augmentations within minibatch are different
    for standard augmentation
    """

    source, _ = create_test_img
    target, _ = create_test_img
    seg, img_dims = create_test_img

    config = dict(STD_AUG_CONFIG)
    config["hyperparameters"] = {"img_dims": img_dims[1:-1]}
    aug = StdAug(config)

    (aug_source, aug_target), aug_seg = aug([source, target], seg)

    for i in range(1, img_dims[0]):
        assert not np.isclose(
            aug_source[i - 1, ...],
            aug_source[i, ...],
        ).all()
        assert not np.isclose(
            aug_target[i - 1, ...],
            aug_target[i, ...],
        ).all()
        assert not np.isclose(
            aug_seg[i - 1, ...],
            aug_seg[i, ...],
        ).all()


# -------------------------------------------------------------------------


@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS)
def test_rpt_different_StdAug(create_test_img: Callable) -> None:
    """Test sequential augmentations are different
    for standard augmentation
    """

    source, _ = create_test_img
    target, _ = create_test_img
    seg, img_dims = create_test_img

    config = dict(STD_AUG_CONFIG)
    config["hyperparameters"] = {"img_dims": img_dims[1:-1]}
    aug = StdAug(config)

    (aug_source, aug_target), aug_seg = aug([source, target], seg)
    (new_source, new_target), new_seg = aug([source, target], seg)

    assert not np.isclose(aug_source, new_source).all()
    assert not np.isclose(aug_target, new_target).all()
    assert not np.isclose(aug_seg, new_seg).all()


# -------------------------------------------------------------------------


@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS)
def test_source_target_same_DiffAug(create_test_img: Callable) -> None:
    """Test that source, target and segmentations
    are augmented identically for differentiable augmentation
    """

    source, _ = create_test_img
    target, _ = create_test_img
    seg, img_dims = create_test_img

    # Check without segmentations first as they don't get colour augmented
    config = dict(DIFF_AUG_CONFIG)
    config["augmentation"]["depth"] = img_dims[3]
    aug = DiffAug(config)

    (aug_source, aug_target), _ = aug([source, target], None)
    assert np.isclose(aug_source, aug_target).all()

    config["augmentation"]["colour"] = False
    (aug_source), aug_seg = aug([source], seg)
    assert np.isclose(aug_source, aug_seg).all()


# -------------------------------------------------------------------------


@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS)
def test_mb_different_DiffAug(create_test_img: Callable) -> None:
    """Test augmentations within minibatch are different
    for differentiable augmentation
    """

    source, _ = create_test_img
    target, _ = create_test_img
    seg, img_dims = create_test_img

    config = dict(DIFF_AUG_CONFIG)
    config["augmentation"]["depth"] = img_dims[3]
    aug = DiffAug(config)

    (aug_source, aug_target), aug_seg = aug([source, target], seg)

    for i in range(1, img_dims[0]):
        assert not np.isclose(
            aug_source[i - 1, ...],
            aug_source[i, ...],
        ).all()
        assert not np.isclose(
            aug_target[i - 1, ...],
            aug_target[i, ...],
        ).all()
        assert not np.isclose(
            aug_seg[i - 1, ...],
            aug_seg[i, ...],
        ).all()


# -------------------------------------------------------------------------


@pytest.mark.parametrize("img_dims", TEST_IMG_DIMS)
def test_rpt_different_DiffAug(create_test_img: Callable) -> None:
    """Test sequential augmentations are different
    for differentiable augmentation
    """

    source, _ = create_test_img
    target, _ = create_test_img
    seg, img_dims = create_test_img

    config = dict(DIFF_AUG_CONFIG)
    config["augmentation"]["depth"] = img_dims[3]
    aug = DiffAug(config)

    (aug_source, aug_target), aug_seg = aug([source, target], seg)
    (new_source, new_target), new_seg = aug([source, target], seg)

    assert not np.isclose(aug_source, new_source).all()
    assert not np.isclose(aug_target, new_target).all()
    assert not np.isclose(aug_seg, new_seg).all()
