import tensorflow as tf

from Utils.model import CRAModel, PatchDiscriminator, PerceptualNet


def create_generator(opt):
    generator = CRAModel.RefinedNet(opt.__activation__)
    if opt.__gexists__:
        tf.keras.backend.set_image_data_format("channels_first")
        generator.build(input_shape=(1, 4, opt.__height__, opt.__width__))
        latest = tf.train.latest_checkpoint(opt.__gpath__)
        generator.load_weights(latest)
        print('Loaded Existing Generator %s' % opt.__gpath__)
    print("Generator Ready")
    return generator


def create_discriminator(opt):
    discriminator = PatchDiscriminator()
    if opt.__dexists__:
        discriminator.build(input_shape=(1, 4, opt.__height__, opt.__width__))
        latest = tf.train.latest_checkpoint(opt.__dpath__)
        discriminator.load_weights(latest)
        print('Loaded Existing Discriminator %s' % opt.__dpath__)
    print("Discriminator Ready")
    # The base repo has a weight initalisation process. Will use standard TF for now.
    return discriminator


def create_perceptual_net():
    perceptual_net = PerceptualNet()
    vgg_16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
                                         pooling=None, classes=1000, classifier_activation='softmax')
    perceptual_net.build(input_shape=(None, 3, 256, 256))
    weights = vgg_16.get_weights()[:20]  # Only need first 20 (Other combos will fall over)
    perceptual_net.set_weights(weights)
    perceptual_net.trainable = False
    print("PerceptualNet Ready")
    return perceptual_net


def psnr_loss(y_true, y_pred):
    return -tf.image.psnr(y_true, y_pred, max_val=255)


def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)


if __name__ == "__main__":
    tf.keras.backend.set_image_data_format("channels_first")
    create_perceptual_net()
