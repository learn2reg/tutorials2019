import os
import nibabel as nib
import numpy as np


dir_moving_image = './data/train/mr_images'
dir_fixed_image = './data/train/us_images'
dir_moving_label = './data/train/mr_labels'
dir_fixed_label = './data/train/us_labels'


moving_images = [np.asarray(nib.load(os.path.join(dir_moving_image, f)).dataobj)
                 for f in os.listdir(dir_moving_image) if os.path.isfile(os.path.join(dir_moving_image, f))]
fixed_images = [np.asarray(nib.load(os.path.join(dir_fixed_image, f)).dataobj)
                for f in os.listdir(dir_fixed_image) if os.path.isfile(os.path.join(dir_fixed_image, f))]

moving_labels = [np.asarray(nib.load(os.path.join(dir_moving_label, f)).dataobj).astype(np.uint8)
                 for f in os.listdir(dir_moving_label) if os.path.isfile(os.path.join(dir_moving_label, f))]
moving_labels = [data[:,:,:,np.newaxis] if data.ndim<4 else data for data in moving_labels]
fixed_labels = [np.asarray(nib.load(os.path.join(dir_fixed_label, f)).dataobj).astype(np.uint8)
                for f in os.listdir(dir_fixed_label) if os.path.isfile(os.path.join(dir_fixed_label, f))]
fixed_labels = [data[:,:,:,np.newaxis] if data.ndim<4 else data for data in fixed_labels]


np.savez_compressed('training_data', moving_images, fixed_images, moving_labels, fixed_labels)
