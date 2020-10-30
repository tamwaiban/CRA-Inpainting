import glob

import tensorflow as tf

AUTO_TUNE = tf.data.experimental.AUTOTUNE
ALL_MASK_TYPES = ['single_bbox', 'bbox', 'free_form']
GENERATOR = tf.random.Generator.from_seed(1)


class RandomMaskLoader:
    def __init__(self):
        pass

    @staticmethod
    def trapez(y, y0, w):
        return tf.clip_by_value(tf.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)

    @staticmethod
    def apply_output(img, yy, xx, val):
        stack = tf.stack([yy, xx], axis=1)
        stack = tf.cast(stack, tf.int64)
        values = tf.ones(len(stack), tf.float32)
        mask = tf.sparse.SparseTensor(indices=stack, values=values, dense_shape=img.shape)
        mask = tf.sparse.reorder(mask)
        mask = tf.sparse.to_dense(mask)
        mask = tf.cast(mask, tf.float32)
        new_values = tf.sparse.SparseTensor(indices=stack, values=val, dense_shape=img.shape)
        new_values = tf.sparse.reorder(new_values)
        new_values = tf.sparse.to_dense(new_values)
        return img * (1 - mask) + new_values * mask

    @staticmethod
    def weighted_line(img, r0, c0, r1, c1, w, shape):
        output = img
        slope = (r1 - r0) / (c1 - c0)
        w *= tf.sqrt(1 + tf.abs(slope)) / 2
        if c1 < c0:
            cold = c0
            c0 = c1
            c1 = cold
        x = tf.range(c0, c1 + 1, dtype=tf.float32)
        y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)
        thickness = tf.math.ceil(w / 2)
        yy = (tf.reshape(tf.math.floor(y), [-1, 1]) + tf.reshape(tf.range(-thickness - 1, thickness + 2), [1, -1]))
        # Can't simply do tf.repeat(x, yy.shape[1]) because there is no shape in the dataloaders view of it.
        xx = tf.repeat(x, tf.cast(2 * thickness + 3, tf.uint16))
        values = tf.reshape(RandomMaskLoader.trapez(yy, tf.reshape(y, [-1, 1]), w), [-1])
        yy = tf.reshape(yy, [-1])

        limits_y = tf.math.logical_and(yy >= 0, yy < shape)
        limits_x = tf.math.logical_and(xx >= 0, xx < shape)
        limits = tf.math.logical_and(limits_y, limits_x)
        limits = tf.math.logical_and(limits, values > 0)
        yy = tf.cast(yy[limits], tf.float32)
        xx = tf.cast(xx[limits], tf.float32)
        return RandomMaskLoader.apply_output(output, yy, xx, values[limits])

    @staticmethod
    def random_ff_mask(shape, max_angle=4, max_len=40, max_width=10, times=15):
        height = shape
        width = shape
        mask = tf.zeros((height, width), tf.float32)
        times = GENERATOR.uniform([], 1, times)
        for i in range(times):
            start_x = GENERATOR.uniform([], 1, width)
            start_x = tf.cast(start_x, tf.float32)
            start_y = GENERATOR.uniform([], 1, height)
            start_y = tf.cast(start_y, tf.float32)
            for j in range(1 + GENERATOR.uniform([], 0, 5)):
                angle = 0.01 + GENERATOR.uniform([], 0, max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + GENERATOR.uniform([], 0, max_len)
                brush_w = tf.cast(5 + GENERATOR.uniform([], 0, max_width), tf.float32)
                end_x = tf.cast((start_x + length * tf.sin(angle)), tf.float32)
                end_y = tf.cast((start_y + length * tf.cos(angle)), tf.float32)
                mask = RandomMaskLoader.weighted_line(mask, start_y, start_x, end_y, end_x, brush_w,
                                                      tf.cast(shape, tf.float32))
                start_x, start_y = end_x, end_y
        return tf.expand_dims((1 - mask) * 255, axis=-1)

    @staticmethod
    def random_bbox(shape, margin, bbox_shape):
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        max_t = img_height - ver_margin - height
        max_l = img_width - hor_margin - width
        top = GENERATOR.uniform([], ver_margin, max_t)
        left = GENERATOR.uniform([], hor_margin, max_l)
        return top / img_height, left / img_width, (height + top) / img_height, (width + left) / img_width

    @staticmethod
    def fill_bbox(img, top, left, height, width, shape):
        top = top * shape
        left = left * shape
        height = height * shape
        width = width * shape
        x_s = tf.range(left, width)
        y_s = tf.range(top, height)
        x_s, y_s = tf.meshgrid(x_s, y_s)
        x_s = tf.reshape(tf.cast(x_s, tf.float32), [-1])
        y_s = tf.reshape(tf.cast(y_s, tf.float32), [-1])
        values = tf.ones(len(x_s), tf.float32)
        img = tf.cast(img, tf.float32)
        img = RandomMaskLoader.apply_output(img, y_s, x_s, tf.cast(values, tf.float32))
        return img

    @staticmethod
    def bbox2mask(shape, margin, bbox_shape, times):
        mask = tf.zeros((1, shape, shape, 1), tf.float32)
        for i in range(times):
            bbox_base = RandomMaskLoader.random_bbox(shape, margin, bbox_shape)
            mask = RandomMaskLoader.fill_bbox(tf.squeeze(tf.squeeze(mask, axis=0), axis=-1), bbox_base[0],
                                              bbox_base[1], bbox_base[2], bbox_base[3], shape)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=-1)
        return tf.squeeze((1 - mask) * 255, axis=0)


class ImageDataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.file_list = glob.glob(opt.__flist__ + "/*.jpg") + glob.glob(opt.__flist__ + "/*.png")
        self.file_list = tf.random.shuffle(self.file_list)
        assert self.opt.__mask_type__ in ALL_MASK_TYPES, "Invalid Mask Type, Valid Types: " + ' '.join(ALL_MASK_TYPES)
        assert (len(self.file_list) != 0), "Invalid Data Path"
        if self.opt.__gen_masks__:
            if self.opt.__mask_type__ == "single_bbox":
                self.opt.__box_count__ = 1
            self.data = self.prepare_training_gen()
        else:
            self.mask_list = glob.glob(opt.__mlist__ + "/*.jpg") + glob.glob(opt.__mlist__ + "/*.png")
            assert (len(self.mask_list) != 0), "Invalid Mask Path"
            self.data = self.prepare_training()

    @staticmethod
    def load_image(image_path, channels=3):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, expand_animations=False, channels=channels)
        img.set_shape([None, None, channels])
        return img

    @staticmethod
    def lambda_image(img_a, img_b):
        img_a = img_a / 255.
        img_b = img_b / 255.
        img_b = 1 - img_b
        img_a = tf.transpose(img_a, perm=[2, 0, 1])
        img_b = tf.transpose(img_b, perm=[2, 0, 1])
        input_ = tf.concat((img_a, img_b), axis=0)
        return input_

    def balance_input_ds(self, val_a, val_b):
        val_a = tf.image.resize(val_a, [self.opt.__height__, self.opt.__width__])
        val_b = tf.image.resize(val_b, [self.opt.__height__, self.opt.__width__])
        return val_a, val_b

    def prepare_images(self, data):
        data = data.map(self.lambda_image, num_parallel_calls=AUTO_TUNE)
        return data.batch(1)

    def return_data(self, ds_a, ds_b):
        balanced_data = tf.data.Dataset.zip((ds_a, ds_b)).map(self.balance_input_ds, num_parallel_calls=AUTO_TUNE)
        assert tf.data.experimental.cardinality(balanced_data).numpy() > 0, "No Data Available. Aborting"
        balanced_data = balanced_data.shuffle(buffer_size=256).repeat()
        return self.prepare_images(balanced_data)

    def prepare_training(self):
        ds_a = tf.data.Dataset.from_tensor_slices(self.file_list)
        ds_a = ds_a.map(lambda x: self.load_image(x, 3), num_parallel_calls=AUTO_TUNE)
        ds_b = tf.data.Dataset.from_tensor_slices(self.mask_list)
        ds_b = ds_b.map(lambda x: self.load_image(x, 1), num_parallel_calls=AUTO_TUNE)
        return self.return_data(ds_a, ds_b)

    def prepare_training_gen(self):
        ds_a = tf.data.Dataset.from_tensor_slices(self.file_list)
        ds_a = ds_a.map(lambda x: self.load_image(x, 3), num_parallel_calls=AUTO_TUNE)
        ds_b = tf.data.Dataset.from_tensor_slices(tf.zeros(shape=len(self.file_list)))
        masker = RandomMaskLoader()
        if self.opt.__mask_type__ == 'free_form':
            kwargs = dict(shape=self.opt.__height__, max_angle=4, max_len=40, max_width=30,
                          times=self.opt.__box_count__)
            ds_b = ds_b.map(lambda x: masker.random_ff_mask(**kwargs), num_parallel_calls=AUTO_TUNE)
        else:
            kwargs = dict(shape=self.opt.__height__, margin=20, bbox_shape=40, times=self.opt.__box_count__)
            ds_b = ds_b.map(lambda x: masker.bbox2mask(**kwargs), num_parallel_calls=AUTO_TUNE)
        return self.return_data(ds_a, ds_b)


class BalancedAudioLoader:
    def __init__(self, opt):
        self.opt = opt
        self.file_list = glob.glob(self.opt.__flist__ + "/*.mp3") + glob.glob(self.opt.__flist__ + "/*.wav")
        self.file_list = tf.random.shuffle(self.file_list)
        assert self.opt.__mask_type__ in ALL_MASK_TYPES, "Invalid Mask Type, Valid Types: " + ' '.join(ALL_MASK_TYPES)
        assert (len(self.file_list) != 0), "Invalid Data Path"
        if self.opt.__gen_masks__:
            if self.opt.__mask_type__ == "single_bbox":
                self.opt.__box_count__ = 1
            self.data = self.prepare_training_gen()
        else:
            self.mask_list = glob.glob(self.opt.__mlist__ + "/*.jpg") + glob.glob(self.opt.__mlist__ + "/*.png")
            assert (len(self.mask_list) != 0), "Invalid Mask Path"
            self.data = self.prepare_training()

    @staticmethod
    def plot_transform(real, imag):
        complex_array = tf.complex(real, imag)
        db_scaled = BalancedAudioLoader.amplitude_to_db(complex_array)
        return db_scaled

    @staticmethod
    def revert_complex_transform(real, imag, filepath):
        complex_ = tf.complex(real, imag)
        inverse_transform = tf.signal.inverse_stft(tf.squeeze(complex_, axis=-1), frame_length=1024, frame_step=512)
        string = tf.audio.encode_wav(tf.expand_dims(inverse_transform, axis=-1), 32000)
        tf.io.write_file(filepath, string)

    @staticmethod
    def tf_log10(input_):
        numerator = tf.math.log(input_)
        return numerator / tf.math.log(tf.constant(10, dtype=numerator.dtype))

    @staticmethod
    def power_to_db(power, amin=1e-16, top_db=80.0):
        ref_value = tf.reduce_max(power)
        log_spec = 10.0 * BalancedAudioLoader.tf_log10(tf.maximum(amin, power))
        log_spec -= 10.0 * BalancedAudioLoader.tf_log10(tf.maximum(amin, ref_value))
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
        return log_spec

    @staticmethod
    def load_audio(audio_path):
        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
        return audio

    @staticmethod
    def amplitude_to_db(transform, amin=1e-5, top_db=80.0):
        power = tf.square(tf.abs(transform))
        return BalancedAudioLoader.power_to_db(power, amin=amin ** 2, top_db=top_db)

    @classmethod
    def convert_complex_transform(cls, audio_a, mask):
        audio_a = tf.signal.stft(tf.squeeze(audio_a, axis=-1), frame_length=1024, frame_step=512)
        areal, a_imag = tf.math.real(audio_a), tf.math.imag(audio_a)
        audio_a = tf.stack((areal, a_imag), axis=-1)
        mask = tf.image.resize(mask, [audio_a.shape[0], audio_a.shape[1], 1])
        input_ = tf.concat((audio_a, mask), axis=-1)
        return tf.transpose(input_, perm=[2, 0, 1])

    def prepare_training(self):
        ds_a = tf.data.Dataset.from_tensor_slices(self.file_list)
        ds_a = ds_a.map(self.load_audio, num_parallel_calls=AUTO_TUNE)
        ds_b = tf.data.Dataset.from_tensor_slices(self.mask_list)
        ds_b = ds_b.map(lambda x: ImageDataLoader.load_image(x, 1), num_parallel_calls=AUTO_TUNE)
        balanced_data = tf.data.Dataset.zip((ds_a, ds_b))
        assert tf.data.experimental.cardinality(balanced_data).numpy() > 0, "No Data Available. Aborting"
        return balanced_data.map(BalancedAudioLoader.convert_complex_transform, num_parallel_calls=AUTO_TUNE).batch(1)

    def prepare_training_gen(self):
        ds_a = tf.data.Dataset.from_tensor_slices(self.file_list)
        ds_a = ds_a.map(self.load_audio, num_parallel_calls=AUTO_TUNE)
        ds_b = tf.data.Dataset.from_tensor_slices(tf.zeros(shape=len(self.file_list)))
        if self.opt.__mask_type__ == 'free_form':
            kwargs = dict(shape=self.opt.__height__, max_angle=4, max_len=40, max_width=10,
                          times=self.opt.__box_count__)
            ds_b = ds_b.map(lambda x: RandomMaskLoader.random_ff_mask(**kwargs), num_parallel_calls=AUTO_TUNE)
        else:
            kwargs = dict(shape=self.opt.__height__, margin=20, bbox_shape=40, times=self.opt.__box_count__)
            ds_b = ds_b.map(lambda x: RandomMaskLoader.bbox2mask(**kwargs), num_parallel_calls=AUTO_TUNE)
        balanced_data = tf.data.Dataset.zip((ds_a, ds_b))
        assert tf.data.experimental.cardinality(balanced_data).numpy() > 0, "No Data Available. Aborting"
        return balanced_data.map(BalancedAudioLoader.convert_complex_transform, num_parallel_calls=AUTO_TUNE).batch(1)
