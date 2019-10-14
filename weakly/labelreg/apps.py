"""
This is a tutorial example.
"""
import tensorflow as tf
import labelreg.utils as util


def warp_volumes_by_ddf(input_, ddf):
    grid_warped = util.get_reference_grid(ddf.shape[1:4]) + ddf
    warped = util.resample_linear(tf.convert_to_tensor(input_, dtype=tf.float32), grid_warped)
    with tf.Session() as sess:
        return sess.run(warped)
