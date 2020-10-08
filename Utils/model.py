import tensorflow as tf
import tensorflow_addons as tfa


class ScConv(tf.keras.layers.Layer):
    def __init__(self, kernel, stride, dilation, padding=None):
        tf.keras.backend.set_image_data_format("channels_first")
        super(ScConv, self).__init__()
        if padding is not None:
            self.sc_conv = tf.keras.layers.Conv2D(1, kernel, stride, dilation_rate=dilation,
                                                  activation=tf.nn.sigmoid, padding=padding)
        else:
            self.sc_conv = tf.keras.layers.Conv2D(1, kernel, stride, dilation_rate=dilation,
                                                  activation=tf.nn.sigmoid)

    def call(self, inputs, **kwargs):
        return self.sc_conv(inputs)


class DepthSeperatedConv(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding=None):
        tf.keras.backend.set_image_data_format("channels_first")
        super(DepthSeperatedConv, self).__init__()
        if padding is not None:
            self.depth_conv = tf.keras.layers.Conv2D(in_ch, kernel_size, stride,
                                                     dilation_rate=dilation, padding=padding)
            self.point_conv = tf.keras.layers.Conv2D(out_ch, 1, 1, dilation_rate=1,
                                                     activation=tf.nn.sigmoid, padding=padding)
        else:
            self.depth_conv = tf.keras.layers.Conv2D(in_ch, kernel_size, stride,
                                                     dilation_rate=dilation)
            self.point_conv = tf.keras.layers.Conv2D(out_ch, 1, 1, dilation_rate=1,
                                                     activation=tf.nn.sigmoid)

    def call(self, inputs, **kwargs):
        out = self.depth_conv(inputs)
        out = self.point_conv(out)
        return out


class MaskedConvolution(tf.keras.layers.Layer):
    def __init__(self, in_filters, filters, kernel_size, stride=1, pad_val=0, dilation=1, padding="SAME", activation=tf.nn.relu,
                 sc=False):
        tf.keras.backend.set_image_data_format("channels_first")
        super(MaskedConvolution, self).__init__()
        assert padding in ["SAME", "SYMMETRIC", "REFLECT", None]
        self.pad_val = pad_val
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.c1 = tf.keras.layers.Conv2D(filters, self.kernel_size, stride, dilation_rate=self.dilation,
                                         activation=activation)
        # self.c2 doesn't use padding
        if sc:
            self.c2 = ScConv(self.kernel_size, stride, self.dilation)
        else:
            self.c2 = DepthSeperatedConv(in_filters, filters, self.kernel_size, stride, self.dilation)

    def call(self, inputs, **kwargs):
        if self.padding == "SYMMETRIC" or self.padding == "REFLECT":
            inputs = tf.pad(inputs, [[0, 0], [0, 0], [self.pad_val, self.pad_val], [self.pad_val, self.pad_val]],
                            mode=self.padding)
        tmp_input = inputs
        inputs = self.c1(tmp_input)
        gated_mask = self.c2(tmp_input)
        return inputs * gated_mask


class TransposedMaskedConvolution(tf.keras.layers.Layer):
    def __init__(self, in_filters, filters, kernel_size, stride=1, pad_val=0, dilation=1, padding=None,
                 activation=tf.nn.leaky_relu, s_factor=2, sc=False):
        tf.keras.backend.set_image_data_format("channels_first")
        super(TransposedMaskedConvolution, self).__init__()
        self.s_factor = s_factor
        self.c1 = MaskedConvolution(in_filters, filters, kernel_size, stride, pad_val, dilation, padding, activation,
                                    sc=sc)

    def call(self, inputs, training=None, mask=None):
        x = tf.transpose(inputs, perm=[0, 2, 3, 1])  # Roll to channels last for below op
        x = tf.image.resize(x, [int(x.shape[1] * self.s_factor), int(x.shape[2] * self.s_factor)],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        return self.c1(x)


class CRAModel:
    def __init__(self):
        tf.keras.backend.set_image_data_format("channels_first")

    @staticmethod
    def custom_ssim(y_pred: float, y_true: float) -> float:
        return tf.image.ssim_multiscale(y_pred, y_true, 1, [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    @staticmethod
    def cal_patch(patch_num, mask, raw_size):
        ksize = raw_size // patch_num
        pool = tf.nn.max_pool2d(mask, ksize, ksize, padding="SAME")  # Or valid?
        return pool

    @staticmethod
    def cosine_matrix(matrix_a, matrix_b):
        _matrixA_matrixB = tf.matmul(matrix_a, tf.transpose(matrix_b, perm=[0, 2, 1]))
        _matrixA_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum((matrix_a * matrix_a), axis=2)), axis=2)
        _matrixB_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum((matrix_b * matrix_b), axis=2)), axis=2)
        return _matrixA_matrixB / tf.matmul(_matrixA_norm, tf.transpose(_matrixB_norm, perm=[0, 2, 1]))

    @staticmethod
    def extract_image_patches(img, patch_num):
        b, c, h, w = img.shape
        img = tf.reshape(img, [b, c, patch_num, h // patch_num, patch_num, w // patch_num])
        img = tf.transpose(img, perm=[0, 2, 4, 3, 5, 1])
        return img

    @staticmethod
    def compute_attention(feature, patch_fb):
        b = feature.shape[0]
        feature = tf.transpose(feature, perm=[0, 2, 3, 1])
        feature = tf.image.resize(feature, [int(feature.shape[1] // 2), int(feature.shape[2] // 2)],
                                  method=tf.image.ResizeMethod.BILINEAR)

        p_fb = tf.reshape(patch_fb, [b, 32 * 32, 1])
        p_matrix = tf.matmul(p_fb, tf.transpose(1 - p_fb, perm=[0, 2, 1]))
        f = tf.reshape(feature, [b, 32 * 32, 128])
        c = CRAModel.cosine_matrix(f, f) * p_matrix
        s = tf.nn.softmax(c, axis=2) * p_matrix
        return s

    @staticmethod
    def attention_transfer(feature, attention):
        b_num, c, h, w = feature.shape
        f = CRAModel.extract_image_patches(feature, 32)
        f = tf.reshape(f, [b_num, f.shape[1] * f.shape[2], -1])
        f = tf.matmul(attention, f)
        f = tf.reshape(f, [b_num, 32, 32, h // 32, w // 32, c])
        f = tf.transpose(f, perm=[0, 5, 1, 3, 2, 4])
        f = tf.reshape(f, [b_num, c, h, w])
        return f

    def dilated_block(self):
        pass

    class CoarseNet(tf.keras.Model):
        def __init__(self, activation):
            tf.keras.backend.set_image_data_format("channels_first")
            super(CRAModel.CoarseNet, self).__init__()
            self.activation = activation
            self.coarse_blocks = []
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                MaskedConvolution(4, 32, 5, 2, 2, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(32, 32, 3, 1, 1, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(32, 64, 3, 2, 1, activation=activation, padding="REFLECT", sc=True)
            ]))
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True)
            ]))
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True)
            ]))
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                MaskedConvolution(64, 64, 3, 1, 2, 2, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 2, 2, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 2, 2, activation=activation, padding="REFLECT", sc=True)
            ]))
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                MaskedConvolution(64, 64, 3, 1, 4, 4, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 4, 4, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 4, 4, activation=activation, padding="REFLECT", sc=True)
            ]))
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                MaskedConvolution(64, 64, 3, 1, 8, 8, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 8, 8, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 8, 8, activation=activation, padding="REFLECT", sc=True)
            ]))
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                MaskedConvolution(64, 64, 3, 1, 16, 16, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 16, 16, activation=activation, padding="REFLECT", sc=True),
            ]))
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True),
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT", sc=True)
            ]))
            self.coarse_blocks.append(tf.keras.Sequential(layers=[
                TransposedMaskedConvolution(64, 64, 1, 1, 1, sc=True),
                TransposedMaskedConvolution(64, 32, 1, 1, 1, sc=True),
                MaskedConvolution(32, 3, 3, 1, 1, activation=None, padding="REFLECT")
            ]))

        def get_config(self):
            pass

        def call(self, inputs, training=False, mask=None):
            x = self.coarse_blocks[0](inputs)
            for block in self.coarse_blocks[1:-1]:
                x = block(x) + x
            x = self.coarse_blocks[-1](x)
            x = tf.nn.tanh(x)
            return x

    class RefinedNet(tf.keras.Model):
        def __init__(self, activation):
            tf.keras.backend.set_image_data_format("channels_first")
            super(CRAModel.RefinedNet, self).__init__()

            self.coarse_net = CRAModel.CoarseNet(activation)

            self.rb1 = tf.keras.Sequential(layers=[
                MaskedConvolution(3, 32, 5, 2, 2, activation=activation, padding="REFLECT"),
                MaskedConvolution(32, 32, 3, 1, 1, activation=activation, padding="REFLECT")
            ])
            self.rb2 = tf.keras.Sequential(layers=[
                MaskedConvolution(32, 64, 3, 2, 1, activation=activation, padding="REFLECT"),
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT")
            ])
            self.rb3 = MaskedConvolution(64, 128, 3, 2, 1, activation=activation, padding="REFLECT")

            self.rb4 = tf.keras.Sequential(layers=[
                MaskedConvolution(128, 128, 3, 1, 1, activation=activation, padding="REFLECT"),
                MaskedConvolution(128, 128, 3, 1, 1, activation=activation, padding="REFLECT")
            ])
            self.rb5 = tf.keras.Sequential(layers=[
                MaskedConvolution(128, 128, 3, 1, 2, 2, activation=activation, padding="REFLECT"),
                MaskedConvolution(128, 128, 3, 1, 4, 4, activation=activation, padding="REFLECT")
            ])
            self.rb6 = tf.keras.Sequential(layers=[
                MaskedConvolution(128, 128, 3, 1, 8, 8, activation=activation, padding="REFLECT"),
                MaskedConvolution(128, 128, 3, 1, 16, 16, activation=activation, padding="REFLECT")
            ])

            self.cp13 = MaskedConvolution(128, 128, 3, 1, 1, activation=activation, padding="REFLECT")
            self.rb7 = tf.keras.Sequential(layers=[
                MaskedConvolution(128, 128, 3, 1, 1, activation=activation, padding="REFLECT"),
                TransposedMaskedConvolution(128, 64, 3, 1, 1, activation=activation, padding="REFLECT"),
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT"),
            ])
            self.cp12 = tf.keras.Sequential(layers=[
                MaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT"),
                MaskedConvolution(64, 64, 3, 1, 2, 2, activation=activation, padding="REFLECT")
            ])

            self.rb8 = tf.keras.Sequential(layers=[
                TransposedMaskedConvolution(64, 64, 3, 1, 1, activation=activation, padding="REFLECT"),
                MaskedConvolution(64, 32, 3, 1, 1, activation=activation, padding="REFLECT")
            ])
            self.cp11 = tf.keras.Sequential(layers=[
                MaskedConvolution(32, 32, 3, 1, 1, activation=activation, padding="REFLECT"),
                MaskedConvolution(32, 32, 3, 1, 2, 2, activation=activation, padding="REFLECT")
            ])
            self.rb9 = tf.keras.Sequential(layers=[
                TransposedMaskedConvolution(32, 32, 3, 1, 1, activation=activation, padding="REFLECT"),
                MaskedConvolution(32, 3, 3, 1, 1, activation=None, padding="REFLECT")
            ])

        def get_config(self):
            pass

        def call(self, inputs, training=None, mask=None):
            inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
            r_, g_, b_, mask = tf.split(inputs, 4, axis=-1)
            img = tf.concat((r_, g_, b_), axis=-1)
            # assert img.shape == mask.shape, "Incompatible Mask/Input Found. See Utils.Loader"
            img_bilin = tf.image.resize(img, [int(img.shape[1] // 2), int(img.shape[2] // 2)],
                                        method=tf.image.ResizeMethod.BILINEAR)
            mask_near = tf.image.resize(mask, [int(img.shape[1] // 2), int(img.shape[2] // 2)],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img_bilin = tf.transpose(img_bilin, perm=[0, 3, 1, 2])
            mask_near = tf.transpose(mask_near, perm=[0, 3, 1, 2])

            first_masked_img = img_bilin * (1 - mask_near) + mask_near  # Applies mask to first input :)
            first_input = tf.concat((first_masked_img, mask_near), axis=1)
            first_output = self.coarse_net(first_input)
            first_output = tf.transpose(first_output, perm=[0, 2, 3, 1])
            first_output = tf.image.resize(first_output, [img.shape[1], img.shape[2]],
                                           method=tf.image.ResizeMethod.BILINEAR)
            second_input = img * (1 - mask) + first_output * mask  # Applies mask to second input :)
            first_output = tf.transpose(first_output, perm=[0, 3, 1, 2])
            second_input = tf.transpose(second_input, perm=[0, 3, 1, 2])
            x_1 = self.rb1(second_input)
            x_2 = self.rb2(x_1)
            x_3 = self.rb3(x_2)
            x_4 = self.rb4(x_3) + x_3
            x_5 = self.rb5(x_4) + x_4
            x_6 = self.rb6(x_5) + x_5
            patch_fb = CRAModel.cal_patch(32, mask, 512)
            attention = CRAModel.compute_attention(x_6, patch_fb)
            cp_13 = self.cp13(CRAModel.attention_transfer(x_6, attention))
            x_7 = tf.concat((x_6, cp_13), axis=1)
            x_7 = self.rb7(x_7)
            cp12 = self.cp12(CRAModel.attention_transfer(x_2, attention))
            x_7 = tf.concat((x_7, cp12), axis=1)
            x_8 = self.rb8(x_7)
            cp11 = self.cp11(CRAModel.attention_transfer(x_1, attention))
            x_8 = tf.concat((x_8, cp11), axis=1)
            x_9 = self.rb9(x_8)
            second_output = tf.nn.tanh(x_9)
            return first_output, second_output


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        tf.keras.backend.set_image_data_format("channels_first")
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)
        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)
        super(SpectralNormalization, self).build()

    def call(self, inputs, **kwargs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u_hat = self.u
        v_hat = self.v  # init v vector
        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)
        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


class SpectralInstanceNormalisedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride, pad_val, activation, dilation=1, normalisation="in",
                 spectral=True, padding="SYMMETRIC"):
        tf.keras.backend.set_image_data_format("channels_first")
        super(SpectralInstanceNormalisedConv2D, self).__init__()
        if pad_val is None:
            self.pad_val = [[0, 0], [0, 0], [0, 0], [0, 0]]
        else:
            self.pad_val = [[0, 0], [0, 0], [pad_val, pad_val], [pad_val, pad_val]]  # If channels first.
        self.padding = padding
        if spectral:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel, stride, dilation_rate=dilation))
        else:
            self.conv = tf.keras.layers.Conv2D(filters, kernel, stride, dilation_rate=dilation)
        if normalisation == 'bn':
            self.norm = tf.keras.layers.BatchNormalization()
        elif normalisation == 'in':
            self.norm = tfa.layers.InstanceNormalization()
        else:
            self.norm = None
        if activation:
            self.activation = tf.keras.layers.Activation(activation)
        else:
            self.activation = None

    def call(self, inputs, **kwargs):
        x = tf.pad(inputs, self.pad_val, self.padding)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class PatchDiscriminator(tf.keras.Model):
    def __init__(self):
        tf.keras.backend.set_image_data_format("channels_first")
        super(PatchDiscriminator, self).__init__()
        self.sn = True
        self.norm = 'in'
        self.features = tf.keras.Sequential(layers=[
            SpectralInstanceNormalisedConv2D(64, 3, 2, 1, activation='relu',
                                             normalisation=self.norm, spectral=self.sn),
            SpectralInstanceNormalisedConv2D(128, 3, 2, 1, activation='relu',
                                             normalisation=self.norm, spectral=self.sn),
            SpectralInstanceNormalisedConv2D(256, 3, 2, 1, activation='relu',
                                             normalisation=self.norm, spectral=self.sn),
            SpectralInstanceNormalisedConv2D(256, 3, 2, 1, activation='relu',
                                             normalisation=self.norm, spectral=self.sn),
            SpectralInstanceNormalisedConv2D(256, 3, 2, 1, activation='relu',
                                             normalisation=self.norm, spectral=self.sn),
            SpectralInstanceNormalisedConv2D(16, 3, 2, 1, activation='relu',
                                             normalisation=self.norm, spectral=self.sn)
        ])
        self.block_out = tf.keras.layers.Dense(1, input_dim=1024)

    def call(self, inputs, training=None, mask=None):
        x = self.features(inputs)
        x = tf.reshape(x, shape=[x.shape[0], 1024])
        x = self.block_out(x)
        return x

    def get_config(self):
        pass


class PerceptualNet(tf.keras.Model):
    def __init__(self):
        tf.keras.backend.set_image_data_format("channels_first")
        super(PerceptualNet, self).__init__()
        self.features = tf.keras.Sequential(layers=[
            tf.keras.layers.Conv2D(64, 3, 1), tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3, 1), tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(128, 3, 1), tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 1), tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(256, 3, 1), tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, 3, 1), tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, 3, 1),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(512, 3, 1), tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, 3, 1, ), tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, 3, 1)
        ])

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        x = self.features(inputs)
        return x


if __name__ == "__main__":

    maskedNet = CRAModel.RefinedNet(activation=tf.nn.relu)
    maskedNet.build(input_shape=(1, 4, 512, 512))
    maskedNet.summary()

    patchNet = PatchDiscriminator()
    patchNet.build(input_shape=(1, 4, 512, 512))
    patchNet.summary()

    perpNet = PerceptualNet()
    perpNet.build(input_shape=(None, 3, None, None))
    perpNet.summary()
