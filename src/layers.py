import tensorflow as tf
import numpy as np

class DCTConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 output_channels,
                 kernel_size,
                 n_freq=4,
                 lbd=False,
                 coefs_l1reg=None,
                 strides=1,
                 padding='SAME',
                 use_bias=True,
                 bias_initializer="zeros",
                 activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_freq = n_freq
        self.lbd = lbd
        if coefs_l1reg:
            self.coefs_regularizer = tf.keras.regularizers.L1(coefs_l1reg)
        else:
            self.coefs_regularizer = None
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self._indices = None
        self.activation = tf.keras.activations.get(activation)
        self.strides = strides
        self.padding = padding
        if use_bias:
            self.bias = self.add_weight(
                shape=(self.output_channels, ),
                initializer=bias_initializer,
                trainable=True,
                name="bias_dctconv2d",
            )
        else:
            self.bias = None

    def build(self, input_shape):
        if self.n_freq > self.kernel_size:
            raise ValueError("n_freq (", self.n_freq, ") > kernel_size (",
                             self.kernel_size, ")")
        self.c_in = input_shape[-1]
        c_in = self.c_in
        c_out = self.output_channels
        basis = []
        x_grid = np.array(range(self.kernel_size))
        x, y = np.meshgrid(x_grid, x_grid)
        # for tf.nn.conv2d (needed to define manually the filters, i.e. basis): [filter_height, filter_width, in_channels, out_channels]
        for n2 in range(self.n_freq):
            for n1 in range(self.n_freq):
                if self.lbd and n1 + n2 > self.n_freq - 1:
                    continue
                atom = np.cos(
                    (np.pi * (x + 1 / 2) * n1) / self.kernel_size) * np.cos(
                        (np.pi * (y + 1 / 2) * n2) / self.kernel_size)
                atom = atom / np.linalg.norm(atom)
                basis.append(np.reshape(atom, atom.shape + (1, 1)))
        if self.lbd:
            self.n_atoms = int(self.n_freq * (self.n_freq + 1) / 2)
        else:
            self.n_atoms = self.n_freq * self.n_freq
        basis = np.concatenate(basis, -1)
        basis = np.repeat(basis, c_in, axis=-2)
        self.basis = tf.Variable(basis, dtype=tf.float32, trainable=False)
        limit = limit_dct(
            c_in, self.n_atoms
        )  # Normalized the basis (atoms) as in Weiler
        # Define weights (DCT coeffs)
        self.w = self.add_weight(
            shape=(
                1,
                1,
                self.n_atoms * c_in,
                c_out,
            ),
            regularizer=self.coefs_regularizer,
            initializer=tf.keras.initializers.RandomUniform(minval=-limit,
                                                            maxval=limit),
            trainable=True,
            name="w_dct",
        )

    def call(self, inputs):
        # Convolve with DCT basis, then recombine using the trainable DCT coefs
        x_base = tf.nn.depthwise_conv2d(inputs, self.basis,
                                        [1, self.strides, self.strides, 1],
                                        self.padding)
        # Recombine using the trainable DCT coefs with a 1x1 conv weights are 1 x 1 x (n_atoms x c_in) x c_out
        x = tf.nn.conv2d(x_base, self.w, self.strides, self.padding)
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x)
        return x

def is_approx_equal(x, y, epsilon=1e-3):
    return np.abs(x - y) / (np.sqrt(np.abs(x) * np.abs(y)) + epsilon) < epsilon


def tri(x):
    return np.where(np.abs(x) <= 1, np.where(x < 0, x + 1, 1 - x), 0)


def limit_glorot(c_in, c_out):
    return np.sqrt(6 / (c_in + c_out))


def limit_dct(c_in, n_atoms):
    return 2 / (c_in * n_atoms)

