import tensorflow as tf

from .affine_transformation import AffineTransform

PI = 3.14159265359


#-------------------------------------------------------------------------

class DiffAug(tf.keras.layers.Layer):
    """ DiffAug class for differentiable augmentation
        Paper: https://arxiv.org/abs/2006.10738
        Adapted from: https://github.com/mit-han-lab/data-efficient-gans
        :param config: configuration dict
    """
    
    def __init__(self, config, name="diff_aug"):
        super().__init__(name=name)
        self.aug_config = config["augmentation"]
        self.depth = self.aug_config["depth"]

    def brightness(self, x: tf.Tensor):
        """ Random brightness in range [-0.5, 0.5] """

        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) - 0.5
        return x + factor
    
    def saturation(self, x: tf.Tensor):
        """ Random saturation in range [0, 2] """

        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) * 2
        x_mean = tf.reduce_mean(x, axis=4, keepdims=True)
        return (x - x_mean) * factor + x_mean

    def contrast(self, x: tf.Tensor):
        """ Random contrast in range [0.5, 1.5] """

        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) + 0.5
        x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        return (x - x_mean) * factor + x_mean
    
    def translation(
        self,
        imgs: list[tf.Tensor],
        seg: tf.Tensor | None = None,
        ratio: float = 0.125
    ) -> tuple[list[tf.Tensor], tf.Tensor]:
        """ Random translation by ratio 0.125 """

        # NB: This assumes NHWDC format and does not (yet) act in z direction
        num_imgs = len(imgs)
        batch_size = tf.shape(imgs[0])[0]
        image_size = tf.shape(imgs[0])[1:3]

        if seg is not None:
            x = tf.concat(imgs + [seg], axis=3)
        else:
            x = tf.concat(imgs, axis=3)

        shift = tf.cast(tf.cast(image_size, tf.float32) \
                * ratio + 0.5, tf.int32)
        translation_x = tf.random.uniform([batch_size, 1],
                                           -shift[0],
                                           shift[0] + 1,
                                           dtype=tf.int32)
        translation_y = tf.random.uniform([batch_size, 1],
                                           -shift[1],
                                           shift[1] + 1,
                                           dtype=tf.int32)
        grid_x = tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) \
                 + translation_x + 1
        grid_x = tf.clip_by_value(grid_x, 0, image_size[0] + 1)
        grid_y = tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) \
                 + translation_y + 1
        grid_y = tf.clip_by_value(grid_y, 0, image_size[1] + 1)

        x = tf.gather_nd(
            tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]]),
            tf.expand_dims(grid_x, -1),
            batch_dims=1
        )
        x = tf.transpose(
            tf.gather_nd(
                tf.pad(
                    tf.transpose(x, [0, 2, 1, 3, 4]),
                    [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]]
                ),
                tf.expand_dims(grid_y, -1),
                batch_dims=1
            ), [0, 2, 1, 3, 4]
        )
        
        imgs = [x[:, :, :, i * self.depth:(i + 1) * self.depth, :] \
                for i in range(num_imgs)]

        if seg is not None:
            seg = x[:, :, :, -self.depth:, :]

        return imgs, seg

    def cutout(
        self,
        imgs: list[tf.Tensor],
        seg: tf.Tensor | None = None,
        ratio: float = 0.5
    ) -> tuple[list[tf.Tensor], tf.Tensor]:
        """ Random cutout by ratio 0.5 """
        # NB: This assumes NHWDC format and does not (yet) act in z direction

        num_imgs = len(imgs)
        batch_size = tf.shape(imgs[0])[0]
        image_size = tf.shape(imgs[0])[1:3]

        if seg is not None:
            x = tf.concat(imgs + [seg], axis=3)
        else:
            x = tf.concat(imgs, axis=3)

        cutout_size = tf.cast(tf.cast(
            image_size, tf.float32) * ratio + 0.5, tf.int32)
        offset_x = tf.random.uniform(
            [tf.shape(x)[0], 1, 1],
            maxval=image_size[0] + (1 - cutout_size[0] % 2),
            dtype=tf.int32
        )
        offset_y = tf.random.uniform(
            [tf.shape(x)[0], 1, 1],
            maxval=image_size[1] + (1 - cutout_size[1] % 2),
            dtype=tf.int32
        )

        grid_batch, grid_x, grid_y = tf.meshgrid(
            tf.range(batch_size, dtype=tf.int32),
            tf.range(cutout_size[0], dtype=tf.int32),
            tf.range(cutout_size[1], dtype=tf.int32),
            indexing='ij'
        )
        cutout_grid = tf.stack(
            [
                grid_batch, grid_x + offset_x - cutout_size[0] // 2,
                grid_y + offset_y - cutout_size[1] // 2
            ], axis=-1
        )

        mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
        cutout_grid = tf.maximum(cutout_grid, 0)

        cutout_grid = tf.minimum(
            cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))

        mask = tf.maximum(1 - tf.scatter_nd(
            cutout_grid,
            tf.ones([batch_size, cutout_size[0], cutout_size[1]],
            dtype=tf.float32
            ), mask_shape), 0)
        x = x * tf.expand_dims(tf.expand_dims(mask, axis=3), axis=4)
       
        imgs = [x[:, :, :, i * self.depth:(i + 1) * self.depth, :] \
                for i in range(num_imgs)]

        if seg is not None:
            seg = x[:, :, :, -self.depth:, :]

        return imgs, seg

    def call(
        self,
        imgs: list[tf.Tensor],
        seg: tf.Tensor | None = None
    ) -> tuple[list[tf.Tensor], tf.Tensor]:
        """ Augmentation call method
            :param imgs: list of images to be augmented
            :param seg: (optional) segmentation to be augmented
        """
        num_imgs = len(imgs)

        imgs = tf.concat(imgs, axis=4)

        # NB: no saturation as monochrome images
        if self.aug_config["colour"]:
            imgs = self.contrast(self.brightness(imgs))
        imgs = [tf.expand_dims(imgs[:, :, :, :, i], 4) \
                for i in range(num_imgs)]

        if self.aug_config["translation"]:
            imgs, seg = self.translation(imgs, seg)
        if self.aug_config["cutout"]:
            imgs, seg = self.cutout(imgs, seg)

        return imgs, seg


#-------------------------------------------------------------------------

class StdAug(tf.keras.layers.Layer):
    """ Standard augmentation performing flipping,
        rotating, scale and shear
        :param config: configuration dict
    """

    def __init__(self, config, name="std_aug"):
        super().__init__(name=name)

        self.transform = AffineTransform(
            config["hyperparameters"]["img_dims"]
        )

        self.flip_probs = tf.math.log(
            [[config["augmentation"]["flip_prob"],
              1 - config["augmentation"]["flip_prob"]]])
        self.rot_angle = config["augmentation"]["rotation"] / 180 * PI
        self.scale_factor = config["augmentation"]["scale"]
        self.shear_angle = config["augmentation"]["shear"] / 180 * PI
        self.x_shift = [-config["augmentation"]["translate"][0],
                        config["augmentation"]["translate"][0]]
        self.y_shift = [-config["augmentation"]["translate"][1],
                        config["augmentation"]["translate"][1]]

    def flip_matrix(self, mb_size: int):
        """ Create flip transformation matrix """

        updates = tf.reshape(
            tf.cast(
                tf.random.categorical(
                    logits=self.flip_probs,
                    num_samples=mb_size * 2), "float32"),
                [mb_size * 2]
            )
        updates = 2.0 * updates - 1.0
        indices = tf.concat(
            [tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis],
             tf.tile(tf.constant([[0, 0], [1, 1]]), [mb_size, 1])],
            axis=1)
        flip_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])

        return flip_mat

    def rotation_matrix(self, mb_size: int):
        """ Create rotation transformation matrix """

        thetas = tf.random.uniform([mb_size], -self.rot_angle, self.rot_angle)
        rot_mat = tf.stack(
            [
                [tf.math.cos(thetas), -tf.math.sin(thetas)],
                [tf.math.sin(thetas), tf.math.cos(thetas)]
            ]
        )

        rot_mat = tf.transpose(rot_mat, [2, 0, 1])

        return rot_mat

    def scale_matrix(self, mb_size: int):
        """ Create scaling transformation matrix """

        updates = tf.repeat(
            tf.random.uniform([mb_size], * self.scale_factor), 2)
        indices = tf.concat(
            [tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis],
             tf.tile(tf.constant([[0, 0], [1, 1]]), [mb_size, 1])],
            axis=1)
        scale_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])

        return scale_mat

    def shear_matrix(self, mb_size: int):
        """ Create shearing transformation matrix """

        mask = tf.cast(
            tf.random.categorical(
                logits=[[0.5, 0.5]],
                num_samples=mb_size), "float32")
        mask = tf.reshape(
            tf.transpose(
                tf.concat([mask, 1 - mask], axis=0),
                [1, 0]
            ), [1, -1])
        updates = tf.repeat(
            tf.random.uniform(
                [mb_size], -self.shear_angle, self.shear_angle),
            2)
        updates = tf.reshape(updates * mask, [-1])
        indices = tf.concat(
            [tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis],
             tf.tile(tf.constant([[0, 1], [1, 0]]), [mb_size, 1])], axis=1)
        shear_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])
        shear_mat += tf.tile(tf.eye(2)[tf.newaxis, :, :], [mb_size, 1, 1])

        return shear_mat

    def translation_matrix(self, m, mb_size: int):
        """ Create translation transformation matrix """

        xs = tf.random.uniform([mb_size], *self.x_shift)
        ys = tf.random.uniform([mb_size], *self.y_shift)
        xys = tf.stack([xs, ys], axis=0)
        xys = tf.transpose(xys, [1, 0])[:, :, tf.newaxis]
        m = tf.concat([m, xys], axis=2)

        return m

    def transformation(self, mb_size: int):
        """ Compose transformation matrices """

        trans_mat = tf.matmul(
            self.shear_matrix(mb_size), tf.matmul(
                self.flip_matrix(mb_size), tf.matmul(
                    self.rotation_matrix(mb_size),
                    self.scale_matrix(mb_size))))

        trans_mat = self.translation_matrix(trans_mat, mb_size)

        return trans_mat
    
    def call(
        self,
        imgs: list[tf.Tensor],
        seg: tf.Tensor | None = None
    ) -> tuple[list[tf.Tensor], tf.Tensor]:
        """ Augmentation call method
            :param imgs: list of images to be augmented
            :param seg: (optional) segmentation to be augmented
        """

        l = len(imgs)
        imgs = tf.concat(imgs, axis=4)
        mb_size = imgs.shape[0]
        thetas = tf.reshape(self.transformation(mb_size), [mb_size, -1])

        if seg is not None:
            img_seg = tf.concat([imgs, seg], axis=4)
            img_seg = self.transform(im=img_seg, thetas=thetas)
            imgs = [img_seg[:, :, :, :, i][:, :, :, :, tf.newaxis] \
                    for i in range(l)]
            seg = img_seg[:, :, :, :, -1][:, :, :, :, tf.newaxis]

            return tuple(imgs), seg
        
        else:
            imgs = self.transform(im=imgs, mb_size=mb_size, thetas=thetas)
            imgs = [imgs[:, :, :, :, i][:, :, :, :, tf.newaxis] \
                    for i in range(l)]
            return tuple(imgs), None


#-------------------------------------------------------------------------
""" Short routine for visually testing augmentations """

def plot(aug: StdAug | DiffAug) -> None:
    """ Generates and plots example augmented images
        :param aug: augmentation class
    """

    for _ in range(2):
        pred = np.zeros([1] + test_config["data"]["patch_size"] + [1])
        pred[:, 0:pred.shape[1] // 2, 0:pred.shape[1] // 2, :, :] = 1
        pred[:, pred.shape[2] // 2:, pred.shape[2] // 2:, :, :] = 1
        inv_pred = 1 - pred
        seg = np.zeros_like(pred)
        seg[
            :, seg.shape[1] // 4:seg.shape[1] * 3 // 4,
            seg.shape[2] // 4:seg.shape[2] * 3 // 4, :, :
        ] = 1

        (pred_aug, inv_pred_aug), seg_aug = aug([pred, inv_pred], seg=seg)

        plt.subplot(2, 3, 1)
        plt.imshow(pred[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(inv_pred[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        
        plt.subplot(2, 3, 3)
        plt.imshow(seg[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(pred_aug[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(inv_pred_aug[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        
        plt.subplot(2, 3, 6)
        plt.imshow(seg_aug[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import yaml

    test_config = yaml.load(
        open(Path(__file__).parent / "test_config.yml", 'r'),
        Loader=yaml.FullLoader
    )

    TestAug = StdAug(test_config)
    plot(TestAug)
    TestAug = DiffAug(test_config)
    plot(TestAug)
