import os
import time
import datetime

import tensorflow as tf
from Utils.utils import create_generator, create_discriminator, create_perceptual_net
from Utils.loader import ImageDataLoader


def check_dirs(opt):
    save_folder = opt.__savepath__
    sample_folder = opt.__samplepath__
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)


class WassersteinTrainer:
    """ Not a full wasserstein impl., but takes logic from the design """
    def __init__(self, opt):
        tf.keras.backend.set_image_data_format("channels_first")
        check_dirs(opt)

        self.generator = create_generator(opt)
        self.discriminator = create_discriminator(opt)
        self.perceptual_net = create_perceptual_net()

        g_sch = tf.keras.optimizers.schedules.ExponentialDecay(
            opt.__lrg__, decay_steps=10000, decay_rate=0.96, staircase=True)
        opt_g = tf.keras.optimizers.Adam(learning_rate=g_sch)
        d_sch = tf.keras.optimizers.schedules.ExponentialDecay(
            opt.__lrd__, decay_steps=10000, decay_rate=0.96, staircase=True)
        opt_d = tf.keras.optimizers.Adam(learning_rate=d_sch)

        gen_writer = tf.summary.create_file_writer("./Logs/Generator")

        train_loader = ImageDataLoader(opt)
        prev_time = time.time()
        all_steps = 0
        for epoch in range(opt.__epochs__):
            for step, input_ in enumerate(train_loader.data):
                all_steps += 1
                r_, g_, b_, mask = tf.split(input_, 4, axis=1)  # Our mask is never changing...
                img = tf.concat((r_, g_, b_), axis=1)
                with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
                    first_output, second_output = self.generator(input_, training=True)
                    first_out_whole_img = img * (1 - mask) + first_output * mask
                    second_out_whole_img = img * (1 - mask) + second_output * mask

                    fake_scalar = self.discriminator(tf.concat((second_out_whole_img, mask), axis=1), training=True)
                    real_scalar = self.discriminator(input_, training=True)

                    d_loss = -tf.reduce_mean(real_scalar) + tf.reduce_mean(fake_scalar)

                    d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
                    opt_d.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

                    first_mask_loss = tf.reduce_mean(tf.losses.mean_absolute_error(img, first_out_whole_img))
                    second_mask_loss = tf.reduce_mean(tf.losses.mean_absolute_error(img, second_out_whole_img))

                    fake_scalar = self.discriminator(tf.concat((second_out_whole_img, mask), axis=1), training=True)
                    gan_loss = - tf.reduce_mean(fake_scalar)

                    img_feature_maps = self.perceptual_net(img)
                    second_out_whole_img_feature_maps = self.perceptual_net(second_out_whole_img)
                    second_perceptual_loss = tf.losses.mean_absolute_error(
                        second_out_whole_img_feature_maps, img_feature_maps)
                    second_perceptual_loss = tf.reduce_mean(second_perceptual_loss)

                    loss = opt.__lambda_l1__ * first_mask_loss + opt.__lambda_l1__ * second_mask_loss \
                           + gan_loss * opt.__lambda_gan__ + second_perceptual_loss * opt.__lambda_perceptual__

                    g_grads = g_tape.gradient(loss, self.generator.trainable_weights)
                    opt_g.apply_gradients(zip(g_grads, self.generator.trainable_weights))

                if (step + 1) % 20 == 0:
                    with gen_writer.as_default():  # Will this fall over if batch size isn't 1? Need to r_mean
                        tf.summary.scalar('First Mask Loss', first_mask_loss, step=all_steps)
                        tf.summary.scalar('Second Mask Loss', second_mask_loss, step=all_steps)
                        tf.summary.scalar('Gan Loss', second_mask_loss, step=all_steps)
                        tf.summary.scalar('Perceptual Loss', second_perceptual_loss, step=all_steps)
                        gen_writer.flush()

                    batches_done = epoch * opt.__steps_per_epoch__ + step
                    batches_left = opt.__epochs__ * opt.__steps_per_epoch__ - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()
                    print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f] [D "
                          "Loss: %.5f] [Perceptual Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                          ((epoch + 1), opt.__epochs__, (step + 1), opt.__steps_per_epoch__, first_mask_loss,
                           second_mask_loss, d_loss, second_perceptual_loss, gan_loss, time_left))
                if (step + 1) % opt.__steps_per_epoch__ == 0:
                    if epoch % opt.__checkpoint_interval__ == 0:
                        print('Models Saved at Epoch:', epoch + 1)
                        self.save_model(epoch, opt, disc=True)
                        self.save_model(epoch, opt)
                        self.sample_model(img, mask, first_output, second_output, first_out_whole_img,
                                          second_out_whole_img, epoch, opt.__samplepath__)
                    break

    def sample_model(self, img, mask, first_out, second_out, fout_whole, sout_whole, epoch, sample_folder):
        masked_img = img * (1 - mask) + mask
        img_save = tf.concat((img, masked_img, first_out, second_out, fout_whole, sout_whole), axis=3)
        img_save = img_save * 255
        img_copy = tf.transpose(img_save, perm=[0, 2, 3, 1])[0, :, :, :]
        img_copy = tf.clip_by_value(img_copy, 0, 255)
        img_copy = tf.cast(img_copy, tf.uint8)
        save_img_name = 'epoch' + str(epoch + 1) + 'sample' + '.jpg'
        save_img_path = os.path.join(sample_folder, save_img_name)
        jpeg_string = tf.io.encode_jpeg(img_copy)
        tf.io.write_file(save_img_path, jpeg_string)

    def save_model(self, epoch, opt, disc=False):
        if disc is True:
            net = self.discriminator
            model_name = 'Discriminator_WGAN_epoch%d.ckpt' % (epoch + 1)
        else:
            net = self.generator
            model_name = 'Generator_WGAN_epoch%d.ckpt' % (epoch + 1)
        model_name = os.path.join(opt.__savepath__, model_name)
        net.save_weights(model_name)
