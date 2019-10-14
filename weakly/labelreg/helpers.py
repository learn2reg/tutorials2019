
import numpy as np
import nibabel as nib
import os
import configparser


def get_data_readers(dir_image0, dir_image1, dir_label0=None, dir_label1=None):

    reader_image0 = DataReader(dir_image0)
    reader_image1 = DataReader(dir_image1)

    reader_label0 = DataReader(dir_label0) if dir_label0 is not None else None
    reader_label1 = DataReader(dir_label1) if dir_label1 is not None else None

    # some checks
    if not (reader_image0.num_data == reader_image1.num_data):
        raise Exception('Unequal num_data between images0 and images1!')
    if dir_label0 is not None:
        if not (reader_image0.num_data == reader_label0.num_data):
            raise Exception('Unequal num_data between images0 and labels0!')
        if not (reader_image0.data_shape == reader_label0.data_shape):
            raise Exception('Unequal data_shape between images0 and labels0!')
    if dir_label1 is not None:
        if not (reader_image1.num_data == reader_label1.num_data):
            raise Exception('Unequal num_data between images1 and labels1!')
        if not (reader_image1.data_shape == reader_label1.data_shape):
            raise Exception('Unequal data_shape between images1 and labels1!')
        if dir_label0 is not None:
            if not (reader_label0.num_labels == reader_label1.num_labels):
                raise Exception('Unequal num_labels between labels0 and labels1!')

    return reader_image0, reader_image1, reader_label0, reader_label1


class DataReader:

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.files = os.listdir(dir_name)
        self.files.sort()
        self.num_data = len(self.files)

        self.file_objects = [nib.load(os.path.join(dir_name, self.files[i])) for i in range(self.num_data)]
        self.num_labels = [self.file_objects[i].shape[3] if len(self.file_objects[i].shape) == 4
                           else 1
                           for i in range(self.num_data)]

        self.data_shape = list(self.file_objects[0].shape[0:3])

    def get_num_labels(self, case_indices):
        return [self.num_labels[i] for i in case_indices]

    def get_data(self, case_indices=None, label_indices=None):
        if case_indices is None:
            case_indices = range(self.num_data)
        # todo: check the supplied label_indices smaller than num_labels
        if label_indices is None:  # e.g. images only
            data = [np.asarray(self.file_objects[i].dataobj) for i in case_indices]
        else:
            if len(label_indices) == 1:
                label_indices *= self.num_data
            data = [self.file_objects[i].dataobj[..., j] if self.num_labels[i] > 1
                    else np.asarray(self.file_objects[i].dataobj)
                    for (i, j) in zip(case_indices, label_indices)]
        return np.expand_dims(np.stack(data, axis=0), axis=4)


def random_transform_generator(batch_size, corner_scale=.1):
    offsets = np.tile([[[1., 1., 1.],
                        [1., 1., -1.],
                        [1., -1., 1.],
                        [-1., 1., 1.]]],
                      [batch_size, 1, 1]) * np.random.uniform(0, corner_scale, [batch_size, 4, 3])
    new_corners = np.transpose(np.concatenate((np.tile([[[-1., -1., -1.],
                                                         [-1., -1., 1.],
                                                         [-1., 1., -1.],
                                                         [1., -1., -1.]]],
                                                       [batch_size, 1, 1]) + offsets,
                                               np.ones([batch_size, 4, 1])), 2), [0, 1, 2])  # O = T I
    src_corners = np.tile(np.transpose([[[-1., -1., -1., 1.],
                                         [-1., -1., 1., 1.],
                                         [-1., 1., -1., 1.],
                                         [1., -1., -1., 1.]]], [0, 1, 2]), [batch_size, 1, 1])
    transforms = np.array([np.linalg.lstsq(src_corners[k], new_corners[k], rcond=-1)[0]
                           for k in range(src_corners.shape[0])])
    transforms = np.reshape(np.transpose(transforms[:][:, :][:, :, :3], [0, 2, 1]), [-1, 1, 12])
    return transforms


def initial_transform_generator(batch_size):
    identity = identity_transform_vector()
    transforms = np.reshape(np.tile(identity, batch_size), [batch_size, 1, 12])
    return transforms


def identity_transform_vector():
    identity = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    return identity.flatten()


def get_padded_shape(size, stride):
    return [int(np.ceil(size[i] / stride)) for i in range(len(size))]


def write_images(input_, file_path=None, file_prefix=''):
    if file_path is not None:
        batch_size = input_.shape[0]
        affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        [nib.save(nib.Nifti1Image(input_[idx, ...], affine),
                  os.path.join(file_path,
                               file_prefix + '%s.nii' % idx))
         for idx in range(batch_size)]


class ConfigParser:
    def __init__(self, argv='', config_type='all'):

        nargs_ = len(argv)
        if nargs_ == 2:
            if (argv[1] == '-h') or (argv[1] == '-help'):
                self.print_help()
                exit()
            filename_ = argv[1]
        else:
            filename_ = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config_demo.ini"))
            print('Reading default config file in: %s.' % filename_)

        self.config_file = configparser.ConfigParser()
        if os.path.isfile(filename_):
            self.config_file.read(filename_)
        else:
            print('Using defaults due to missing config file.')

        self.config_type = config_type.lower()
        self.config = self.get_defaults()
        self.check_defaults()
        self.print()

    def check_defaults(self):

        for section_key in self.config.keys():
            if section_key in self.config_file:
                for key, value in self.config[section_key].items():
                    if key in self.config_file[section_key] and self.config_file[section_key][key]:
                        if type(value) == str:
                            self.config[section_key][key] = os.path.expanduser(self.config_file[section_key][key])
                        else:
                            self.config[section_key][key] = eval(self.config_file[section_key][key])
                    # else:
                        # print('Default set in [''%s'']: %s = %s' % (section_key, key, value))
            # else:
                # print('Default section set: [''%s'']' % section_key)

    def __getitem__(self, key):
        return self.config[key]

    def print(self):
        print('')
        for section_key, section_value in self.config.items():
                for key, value in section_value.items():
                    print('[''%s'']: %s: %s' % (section_key, key, value))
        print('')

    def get_defaults(self):

        home_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        network = {'network_type': 'local'}

        data = {'dir_moving_image': os.path.join(home_dir, 'data/train/mr_images'),
                'dir_fixed_image': os.path.join(home_dir, 'data/train/us_images'),
                'dir_moving_label': os.path.join(home_dir, 'data/train/mr_labels'),
                'dir_fixed_label': os.path.join(home_dir, 'data/train/us_labels')}

        loss = {'similarity_type': 'dice',
                'similarity_scales': [0, 1, 2, 4, 8, 16],
                'regulariser_type': 'bending',
                'regulariser_weight': 0.5}

        train = {'total_iterations': int(1e5),
                 'learning_rate': 1e-5,
                 'minibatch_size': 2,
                 'freq_info_print': 100,
                 'freq_model_save': 500,
                 'file_model_save': os.path.join(home_dir, 'data/model.ckpt')}

        inference = {'file_model_saved': train['file_model_save'],
                     'dir_moving_image': os.path.join(home_dir, 'data/test/mr_images'),
                     'dir_fixed_image': os.path.join(home_dir, 'data/test/us_images'),
                     'dir_save': os.path.join(home_dir, 'data/'),
                     'dir_moving_label': '',
                     'dir_fixed_label': ''}

        if self.config_type == 'training':
            config = {'Data': data, 'Network': network, 'Loss': loss, 'Train': train}
        elif self.config_type == 'inference':
            config = {'Network': network, 'Inference': inference}
        else:
            config = {'Data': data, 'Network': network, 'Loss': loss, 'Train': train, 'Inference': inference}

        return config

    @staticmethod
    def print_help():
        print('\n'.join([
            '',
            '************************************************************',
            '  Weakly-Supervised CNNs for Multimodal Image Registration',
            '      2018 Yipeng Hu <yipeng.hu@ucl.ac.uk> ',
            '  LabelReg package is licensed under: ',
            '      http://www.apache.org/licenses/LICENSE-2.0',
            '************************************************************',
            '',
            'Training script:',
            '   python3 training.py myConfig.ini',
            '',
            'Inference script:',
            '   python3 inference.py myConfig.ini',
            '',
            'Options in config file myConfig.ini:',
            '   network_type:       {local, global, composite}',
            '   similarity_type:    {dice, cross-entropy, mean-squared, jaccard}',
            '   regulariser_type:   {bending, gradient-l2, gradient-l1}',
            'See other parameters in the template config file config_demo.ini.',
            ''
        ]))
