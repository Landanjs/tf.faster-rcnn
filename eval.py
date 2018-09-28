import h5py
import tensorflow as tf
import numpy as np
import argparse
from resnet import resnet
import os
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str,
                    help="Path to the dataset")
parser.add_argument("--lr", type=float, default=0.1,
                    help="Initial Learning Rate")
parser.add_argument("--num_classes", type=int, default=10,
                    help="Number of possible classifications")
parser.add_argument("--batch_size", type=int, default=128,
                    help="Training batch size")
parser.add_argument("--num_epochs", type=int, default=182,
                    help="Number of epochs to train for")
parser.add_argument("--load", type=str, default="",
                    help="Checkpoint used to load model")
parser.add_argument("--val_split", type=float, default="0.1",
                    help="Percentage of training set used for val")

args = parser.parse_args()
#TODO: subtract per pixel mean, difference?
# 128 batch size, lr of 0.1 and divide by 10 at 32k and 48k
# terminate at 64k, 45k/5k train/val split
# performance without augmentation?

# better way to input filename?


def get_batch(data_dir, batch_size, train):
    # random state to generate the train/val split
    rand = np.random.RandomState(123)
    data_file = 'test_cifar10.hdf5'
    # calculate the size of the data and the mean of the images
    with h5py.File(os.path.join(data_dir, 'train_cifar10.hdf5')) as data:
        img_mean = np.mean(data['images'], axis=0)
        
    with h5py.File(os.path.join(data_dir, data_file), 'r') as data:
        set_size = int(len(data['labels']))

    indices = np.arange(set_size)
    num_batches = (set_size // batch_size) + 1
    
    for start in range(0, set_size, batch_size):
        end = start + batch_size
        batch = indices[start:end]
        with h5py.File(os.path.join(data_dir, data_file), 'r') as data:
            images = np.zeros((batch_size, 32, 32, 3))
            labels = np.zeros(batch_size)
            for batch_ind, data_ind in enumerate(batch):
                img = data['images'][data_ind] 

                if train:
                    img_pad = np.pad(img, ((4, 4), (4, 4), (0, 0)), 'constant', constant_values=0)
                    x_start = np.random.randint(0, 8)
                    y_start = np.random.randint(0, 8)
                    x_end = x_start + 32
                    y_end = y_start + 32
                    img = img_pad[x_start:x_end, y_start:y_end]
                    if np.random.random_sample() < 0.5:
                        img = np.fliplr(img)

                images[batch_ind] = (img - img_mean) 
                labels[batch_ind] = data['labels'][data_ind].astype(int)
        yield (images, labels)

inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
pred = resnet(inputs, args.num_classes, True, init_kernel_size=3,
              block_sizes=[5]*3, init_num_filters=16, init_conv_stride=1,
              init_pool_size=0, bottleneck=False)
act = tf.placeholder(tf.float32)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, args.load)

epoch_start = time.time()
num_correct, num_samples = 0, 0
for i, batch in enumerate(get_batch(args.data_dir, args.batch_size, train=False)):
    imgs, labels = batch
    predict = sess.run(pred, feed_dict={inputs:imgs})
    num_correct += (predict.argmax(axis=1) == labels).sum()
    num_samples += predict.shape[0]

epoch_end = time.time()
print(num_correct, num_samples)
print('Accuracy {}, Time{}'.format(num_correct / num_samples, epoch_end - epoch_start))


