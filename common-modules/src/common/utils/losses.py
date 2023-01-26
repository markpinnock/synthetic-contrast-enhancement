import tensorflow as tf

EPSILON = 1e-12


#-------------------------------------------------------------------------
""" Non-saturating minimax loss
    Goodfellow et al. Generative adversarial networks. NeurIPS, 2014
    https://arxiv.org/abs/1406.2661 """

@tf.function
def minimax_D(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    real_loss = tf.keras.losses.binary_crossentropy(
        tf.ones_like(real_output), real_output, from_logits=True)
    fake_loss = tf.keras.losses.binary_crossentropy(
        tf.zeros_like(fake_output), fake_output, from_logits=True)
    return 0.5 * tf.reduce_mean(real_loss + fake_loss)

@tf.function
def minimax_G(fake_output: tf.Tensor) -> tf.Tensor:
    fake_loss = tf.keras.losses.binary_crossentropy(
        tf.ones_like(fake_output), fake_output, from_logits=True)
    return tf.reduce_mean(fake_loss)


#-------------------------------------------------------------------------
""" Pix2pix L1 loss
    Isola et al. Image-to-image translation with conditional adversarial networks.
    CVPR, 2017.
    https://arxiv.org/abs/1406.2661 """

@tf.function
def L1(real_img: tf.Tensor, fake_img: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(real_img - fake_img), name="L1")


#-------------------------------------------------------------------------
""" Least squares loss
    Mao et al. Least squares generative adversarial networks. ICCV, 2017.
    https://arxiv.org/abs/1611.04076
    
    Zhu et al. Unpaired Image-to-image translation using cycle-consistent
    adversarial networks. ICCV 2017.
    https://arxiv.org/abs/1703.10593 """

@tf.function
def least_squares_D(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    real_loss = 0.5 * tf.reduce_mean(tf.square(real_output - 1))
    fake_loss = 0.5 * tf.reduce_mean(tf.square(fake_output))
    return fake_loss + real_loss

@tf.function
def least_squares_G(fake_output: tf.Tensor) -> tf.Tensor:
    fake_loss = tf.reduce_mean(tf.square(fake_output - 1))
    return fake_loss


#-------------------------------------------------------------------------
""" Focused L1 loss, calculates L1 inside and outside masked area """

@tf.function
def focused_mae(x, y, m):
    global_absolute_err = tf.abs(x - y)
    focal_absolute_err = tf.abs(x - y) * m
    global_mae = tf.reduce_mean(global_absolute_err)
    focal_mae = tf.reduce_sum(focal_absolute_err) / (tf.reduce_sum(m) + EPSILON)

    return global_mae, focal_mae


#-------------------------------------------------------------------------
""" Focused loss, weights loss according to foreground/background """

class FocusedLoss(tf.keras.layers.Layer):
    def __init__(self, mu, name=None):
        super().__init__(name=name)
        assert mu <= 1.0 and mu >= 0.0, "Mu must be in range [0, 1]"
        self.mu = mu
        self.loss = focused_mae

    def call(self, y, x, mask):
        global_loss, focal_loss = self.loss(x, y, mask)

        return (1 - self.mu) * global_loss + self.mu * focal_loss


#-------------------------------------------------------------------------
""" Focused metric, weights loss according to foreground/background """

class FocusedMetric(tf.keras.metrics.Metric):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.global_loss = self.add_weight(name="global", initializer="zeros")
        self.focal_loss = self.add_weight(name="focal", initializer="zeros")
        self.N = self.add_weight(name="N", initializer="zeros")
        self.loss = focused_mae

    def update_state(self, y, x, mask):
        global_loss, focal_loss = self.loss(x, y, mask)
        self.global_loss.assign_add(global_loss)
        self.focal_loss.assign_add(focal_loss)
        self.N.assign_add(x.shape[0])
    
    def result(self):
        return [self.global_loss / self.N, self.focal_loss / self.N]

    def reset_states(self):
        self.global_loss.assign(0.0)
        self.focal_loss.assign(0.0)
        self.N.assign(EPSILON) # So no nans if result called after reset_states
