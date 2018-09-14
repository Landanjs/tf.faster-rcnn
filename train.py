import h5py
import tensorflow as tf
import numpy as np
import argparse
from resnet import resnet
import os
import time

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
    
    # calculate the size of the data
    with h5py.File(os.path.join(data_dir, 'train_cifar10.hdf5'), 'r') as data:
        data_size = len(data['labels'])
    
    indices = rand.permutation(data_size)
    train_size = int(data_size * (1 - args.val_split))
    if train:
        indices = indices[:train_size]
        np.random.shuffle(indices)
        set_size = train_size
    else:
        indices = indices[train_size:]
        set_size = int(data_size * args.val_split)
        
    num_batches = (set_size // batch_size) + 1
    for start in range(0, set_size, batch_size):
        end = start + batch_size
        batch = indices[start:end]
        with h5py.File(os.path.join(data_dir, 'train_cifar10.hdf5'), 'r') as data:
            images = []
            labels = []
            for ind in batch:
                if np.random.random_sample() < 0.5:
                    images.append(data['images'][ind])
                else:
                    images.append(np.fliplr(data['images'][ind]))
                labels.append(data['labels'][ind])
            images = np.stack(images)
            labels = np.stack(labels).astype(int)
        one_hot_labels = np.zeros((len(labels), args.num_classes))
        one_hot_labels[np.arange(len(labels)), labels[:]] = 1
        yield (images, one_hot_labels)


optimizer = tf.train.MomentumOptimizer(learning_rate = args.lr, momentum=0.9)

# will that continuously pull new batchs or do I need to feed_dict()
# define the initial channels of input
inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
pred = resnet(inputs, args.num_classes, training=True, init_kernel_size=3,
               block_sizes=[3]*3, init_num_filters=16, init_conv_stride=1,
               init_pool_size=0, bottleneck=False)

act = tf.placeholder(tf.float32)
cross_entropy = tf.losses.softmax_cross_entropy(act, pred)
l2_loss = 0.0001 * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
                             ])
loss = l2_loss + cross_entropy
train = optimizer.minimize(loss)
saver = tf.train.Saver()
sess = tf.Session()

if args.load:
    saver.restore(sess, args.load)
else:
    # initialize Variables
    init_vars = tf.global_variables_initializer()
    sess.run(init_vars)

for epoch in range(1, args.num_epochs):        
    print('Starting Epoch {}'.format(epoch))
    num_samples, num_correct = 0, 0
    avg_loss = 0
    epoch_start = time.time()
    for i, batch in enumerate(get_batch(args.data_dir, args.batch_size, train=True)):
        imgs, labels = batch
        # at what point are gradients reset? does loss run through the network again?
        # does having multiple arguments to sess.run affect speed or does it do this efficiently?
        _, loss_value, predict = sess.run((train, loss, pred), feed_dict={inputs:imgs, act:labels})
        num_correct += (predict.argmax(axis=1) == labels.argmax(axis=1)).sum()
        num_samples += predict.shape[0]
        avg_loss = (avg_loss * i + loss_value)  / (i + 1)
    epoch_end = time.time()
    print('Epoch {}, Loss {}, Accuracy {}, Time {}'.format(epoch, avg_loss, num_correct / num_samples, epoch_end - epoch_start))

    # save Variables and run validation
    if epoch % 1 == 0:
        save_path = saver.save(sess, 'checkpoints/resnet/epoch_{}.ckpt'.format(epoch))
        num_samples, num_correct = 0, 0
        for batch in get_batch(args.data_dir, args.batch_size, train=False):
            imgs, labels = batch
            predict = sess.run(pred, feed_dict={inputs:imgs, act:labels})
            num_correct += (predict.argmax(axis=1) == labels.argmax(axis=1)).sum()
            num_samples += predict.shape[0]
        print("Validation Accuracy {}".format(num_correct / num_samples))
    



        
        
    
