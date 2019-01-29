import tensorflow as tf
from models import rmc_vanilla, rmc_att, rmc_vdcnn, lstm_vanilla

generator_dict = {
    'lstm_vanilla': lstm_vanilla.generator,
    'rmc_vanilla': rmc_vanilla.generator,
    'rmc_att': rmc_att.generator,
    'rmc_vdcnn': rmc_vdcnn.generator
}

discriminator_dict = {
    'lstm_vanilla': lstm_vanilla.discriminator,
    'rmc_vanilla': rmc_vanilla.discriminator,
    'rmc_att': rmc_att.discriminator,
    'rmc_vdcnn': rmc_vdcnn.discriminator
}


def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)