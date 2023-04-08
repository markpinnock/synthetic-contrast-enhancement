import numpy as np
import tensorflow as tf


def extract_patches(img, xy_patch, stride_length, patch_size, downsample):
    height, width, depth = img.shape
    patches = []
    indices = []

    if xy_patch is False:
        for k in range(0, depth - patch_size[2] + stride_length, stride_length):
            if (k + patch_size[2]) > depth:
                patches.append(
                    img[::downsample, ::downsample, -patch_size[2] :, np.newaxis],
                )
                indices.append([0, 0, -patch_size[2]])
                return patches, indices

            patches.append(
                img[::downsample, ::downsample, k : (k + patch_size[2]), np.newaxis],
            )
            indices.append([0, 0, k])

        return patches, indices

    else:
        for k in range(0, depth - patch_size[2] + stride_length, stride_length):
            if (k + patch_size[2]) > depth:
                for j in range(0, width - patch_size[1] + stride_length, stride_length):
                    for i in range(
                        0,
                        height - patch_size[0] + stride_length,
                        stride_length,
                    ):
                        patches.append(
                            img[
                                i : (i + patch_size[0]) : downsample,
                                j : (j + patch_size[1]) : downsample,
                                -patch_size[2] :,
                                np.newaxis,
                            ],
                        )
                        indices.append([i, j, -patch_size[2]])

                return patches, indices

            for j in range(0, width - patch_size[1] + stride_length, stride_length):
                for i in range(
                    0,
                    height - patch_size[0] + stride_length,
                    stride_length,
                ):
                    patches.append(
                        img[
                            i : (i + patch_size[0]) : downsample,
                            j : (j + patch_size[1]) : downsample,
                            k : (k + patch_size[2]),
                            np.newaxis,
                        ],
                    )
                    indices.append([i, j, k])

        return patches, indices


def generate_indices(img, stride_length, patch_size, downsample):
    height, width, depth = img.shape
    idx_map = tf.reshape(tf.range(tf.reduce_prod(img.shape)), img.shape)
    indices = []

    for i in range(0, height - patch_size[0] + stride_length, stride_length):
        for j in range(0, width - patch_size[1] + stride_length, stride_length):
            for k in range(0, depth - patch_size[2] + stride_length, stride_length):

                if (k + patch_size[2]) > depth:
                    linear_indices = idx_map[
                        i : (i + patch_size[0]) : downsample,
                        j : (j + patch_size[1]) : downsample,
                        -patch_size[2] :,
                    ]
                    indices.append(linear_indices)

                else:
                    linear_indices = idx_map[
                        i : (i + patch_size[0]) : downsample,
                        j : (j + patch_size[1]) : downsample,
                        k : (k + patch_size[2]),
                    ]
                    indices.append(linear_indices)

    return indices
