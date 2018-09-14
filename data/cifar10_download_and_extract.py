"""Downloads and extracts the binary version of the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile
import h5py
import pickle

from six.moves import urllib
import tensorflow as tf

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='./',
    help='Directory to download data and extract the tarball')

def main(_):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(FLAGS.data_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s  %.1f%%' % (
                filename, 100.0 * count *block_size / total_size))
            sys.stdout.flush()
        
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(FLAGS.data_dir)

    path = 'cifar-10-batches-py'
    batches = [os.path.join(path, 'data_batch_{}'.format(i)) for i in range(1, 6)]

    # save training set in hdf5 file
    with h5py.File('train_cifar10.hdf5', 'w') as data:
        imgs = data.create_dataset('images', (50000, 32, 32, 3), dtype='i8')
        labels = data.create_dataset('labels', (50000,))
    
        for i, batch in enumerate(batches):
            with open(batch, 'rb') as f:
                data_batch = pickle.load(f, encoding='bytes')
                img_batch = data_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                imgs[i*10000:(i+1)*10000] = img_batch
                labels[i*10000:(i+1)*10000] = data_batch[b'labels']
    
    # save test set in hdf5 file
    with h5py.File('test_cifar10.hdf5', 'w') as hdf5:
        imgs = hdf5.create_dataset('images', (10000, 32, 32, 3), dtype='i8')
        labels = hdf5.create_dataset('labels', (10000,))
        
        with open(os.path.join(path, 'test_batch'), 'rb') as data:
            dataset = pickle.load(data, encoding='bytes')
            imgs = dataset[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = dataset[b'labels']
            
            

        #import matplotlib.pyplot as plt
        #plt.imshow(imgs[30])
        #plt.show()
            

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    print(unparsed)
    tf.app.run(argv=[sys.argv[0]] + unparsed)
    
