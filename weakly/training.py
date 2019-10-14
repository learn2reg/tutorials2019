import tensorflow as tf
import sys
import random
import time

import labelreg.helpers as helper
import labelreg.networks as network
import labelreg.utils as util
import labelreg.losses as loss


# 0 - get configs
config = helper.ConfigParser(sys.argv, 'training')

# 1 - data
reader_moving_image, reader_fixed_image, reader_moving_label, reader_fixed_label = helper.get_data_readers(
    config['Data']['dir_moving_image'],
    config['Data']['dir_fixed_image'],
    config['Data']['dir_moving_label'],
    config['Data']['dir_fixed_label'])


# 2 - graph
ph_moving_image = tf.placeholder(tf.float32, [config['Train']['minibatch_size']]+reader_moving_image.data_shape+[1])
ph_fixed_image = tf.placeholder(tf.float32, [config['Train']['minibatch_size']]+reader_fixed_image.data_shape+[1])
ph_moving_affine = tf.placeholder(tf.float32, [config['Train']['minibatch_size']]+[1, 12])
ph_fixed_affine = tf.placeholder(tf.float32, [config['Train']['minibatch_size']]+[1, 12])
input_moving_image = util.warp_image_affine(ph_moving_image, ph_moving_affine)  # data augmentation
input_fixed_image = util.warp_image_affine(ph_fixed_image, ph_fixed_affine)  # data augmentation

# predicting ddf
reg_net = network.build_network(network_type=config['Network']['network_type'],
                                minibatch_size=config['Train']['minibatch_size'],
                                image_moving=input_moving_image,
                                image_fixed=input_fixed_image)

# loss
ph_moving_label = tf.placeholder(tf.float32, [config['Train']['minibatch_size']]+reader_moving_image.data_shape+[1])
ph_fixed_label = tf.placeholder(tf.float32, [config['Train']['minibatch_size']]+reader_fixed_image.data_shape+[1])
input_moving_label = util.warp_image_affine(ph_moving_label, ph_moving_affine)  # data augmentation
input_fixed_label = util.warp_image_affine(ph_fixed_label, ph_fixed_affine)  # data augmentation

warped_moving_label = reg_net.warp_image(input_moving_label)  # warp the moving label with the predicted ddf

loss_similarity, loss_regulariser = loss.build_loss(similarity_type=config['Loss']['similarity_type'],
                                                    similarity_scales=config['Loss']['similarity_scales'],
                                                    regulariser_type=config['Loss']['regulariser_type'],
                                                    regulariser_weight=config['Loss']['regulariser_weight'],
                                                    label_moving=warped_moving_label,
                                                    label_fixed=input_fixed_label,
                                                    network_type=config['Network']['network_type'],
                                                    ddf=reg_net.ddf)

train_op = tf.train.AdamOptimizer(config['Train']['learning_rate']).minimize(loss_similarity+loss_regulariser)

# utility nodes - for information only
dice = util.compute_binary_dice(warped_moving_label, input_fixed_label)
dist = util.compute_centroid_distance(warped_moving_label, input_fixed_label)

# 3 - training
num_minibatch = int(reader_moving_label.num_data/config['Train']['minibatch_size'])
train_indices = [i for i in range(reader_moving_label.num_data)]

saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(config['Train']['total_iterations']):

    if step in range(0, config['Train']['total_iterations'], num_minibatch):
        random.shuffle(train_indices)

    minibatch_idx = step % num_minibatch
    case_indices = train_indices[
                    minibatch_idx*config['Train']['minibatch_size']:(minibatch_idx+1)*config['Train']['minibatch_size']]
    label_indices = [random.randrange(reader_moving_label.num_labels[i]) for i in case_indices]

    trainFeed = {ph_moving_image: reader_moving_image.get_data(case_indices),
                 ph_fixed_image: reader_fixed_image.get_data(case_indices),
                 ph_moving_label: reader_moving_label.get_data(case_indices, label_indices),
                 ph_fixed_label: reader_fixed_label.get_data(case_indices, label_indices),
                 ph_moving_affine: helper.random_transform_generator(config['Train']['minibatch_size']),
                 ph_fixed_affine: helper.random_transform_generator(config['Train']['minibatch_size'])}

    sess.run(train_op, feed_dict=trainFeed)

    if step in range(0, config['Train']['total_iterations'], config['Train']['freq_info_print']):
        current_time = time.asctime(time.gmtime())
        loss_similarity_train, loss_regulariser_train, dice_train, dist_train = sess.run(
            [loss_similarity,
             loss_regulariser,
             dice,
             dist],
            feed_dict=trainFeed)

        # print('----- Training -----')
        print('Step %d [%s]: Loss=%f (similarity=%f, regulariser=%f)' %
              (step,
               current_time,
               loss_similarity_train+loss_regulariser_train,
               1-loss_similarity_train,
               loss_regulariser_train))
        print('  Dice: %s' % dice_train)
        print('  Distance: %s' % dist_train)
        print('  Image-label indices: %s - %s' % (case_indices, label_indices))

    if step in range(0, config['Train']['total_iterations'], config['Train']['freq_model_save']):
        save_path = saver.save(sess, config['Train']['file_model_save'], write_meta_graph=False)
        print("Model saved in: %s" % save_path)
