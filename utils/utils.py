import os

import tensorflow as tf


def get_ms():
    if os.name == 'nt':
        cross_device_ops = tf.distribute.ReductionToOneDevice()
    else:
        cross_device_ops = tf.distribute.NcclAllReduce()
    return tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)

