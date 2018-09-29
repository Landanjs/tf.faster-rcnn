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
    
    # calculate the size of the data and the mean of the images
    with h5py.File(os.path.join(data_dir, 'train_cifar10.hdf5'), 'r') as data:
        data_size = len(data['labels'])
        img_mean = np.mean(data['images'], axis=0)
        
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
    
    for batch_start in range(0, set_size, batch_size):
        batch_end = batch_start + batch_size
        # sort and convert to list for slicing hdf5 file
        batch_ind = list(np.sort(indices[batch_start:batch_end]))
        with h5py.File(os.path.join(data_dir, 'train_cifar10.hdf5'), 'r') as data:
            # Load data
            imgs = data['images'][batch_ind]
            labels = data['labels'][batch_ind]

            # Pad images for preprocessing
            imgs_pad = np.pad(imgs, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant', constant_values=0)

            # allocate array to store cropped images
            images = np.zeros((len(labels), 32, 32, 3))
            for i in range(len(labels)):

                if train:
                    start = np.random.randint(0, 8, size = 2)
                    end = start + 32
                    images[i] = imgs_pad[i, start[0]:end[0], start[1]:end[1]]
                    if np.random.random_sample() < 0.5:
                        images[i] = np.fliplr(images[i])

            images -= img_mean

        yield (images, labels)


# will that continuously pull new batchs or do I need to feed_dict()
# define the initial channels of input
inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
is_training = tf.placeholder(tf.bool)
pred = resnet(inputs, args.num_classes, is_training, init_kernel_size=3,
               block_sizes=[5]*3, init_num_filters=16, init_conv_stride=1,
               init_pool_size=0, bottleneck=False)
pred = tf.cast(pred, tf.float32)

learning_rate = tf.placeholder(tf.float32)
# work better with nesterov?
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=0.9)

act = tf.placeholder(tf.int32)
cross_entropy = tf.losses.sparse_softmax_cross_entropy(act, pred)
l2_loss = 2e-4 * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
loss = l2_loss + cross_entropy
# for batch_norm
# group or dependency (from BN example)?
minimize_op = optimizer.minimize(loss)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train = tf.group(minimize_op, update_op)

saver = tf.train.Saver()
sess = tf.Session()

if args.load:
    saver.restore(sess, args.load)
else:
    # initialize Variables
    init_vars = tf.global_variables_initializer()
    sess.run(init_vars)

global_step = 0
for epoch in range(1, args.num_epochs):        
    print('Starting Epoch {}'.format(epoch))
    num_samples, num_correct = 0, 0
    avg_loss = 0
    epoch_start = time.time()
    batch_start = 0
    for i, batch in enumerate(get_batch(args.data_dir, args.batch_size, train=True)):

        imgs, labels = batch
        # at what point are gradients reset? does loss run through the network again?
        # does having multiple arguments to sess.run affect speed or does it do this efficiently?
        _, loss_value, predict = sess.run((train, loss, pred),
                                          feed_dict={inputs:imgs, is_training:True,
                                                     act:labels, learning_rate:args.lr})
        num_correct += (predict.argmax(axis=1) == labels).sum()
        num_samples += predict.shape[0]
        avg_loss = (avg_loss * i + loss_value)  / (i + 1)
        global_step += 1
        if global_step == 32000 or global_step == 48000:
            args.lr /= 10
            print(f'NEW LEARNING RATE: {args.lr}')
    epoch_end = time.time()
    print('Epoch {}, Loss {}, Accuracy {}, Time {}'.format(epoch, avg_loss, num_correct / num_samples, epoch_end - epoch_start))
    # save Variables and run validation
    if epoch % 10 == 0:
        save_path = saver.save(sess, 'checkpoints/resnet/epoch_{}.ckpt'.format(epoch))
        num_samples, num_correct = 0, 0
        for batch in get_batch(args.data_dir, args.batch_size, train=False):
            imgs, labels = batch
            predict = sess.run(pred, feed_dict={inputs:imgs, is_training:False, act:labels})
            num_correct += (predict.argmax(axis=1) == labels).sum()
            num_samples += predict.shape[0]
        print("Validation Accuracy {}".format(num_correct / num_samples))
    



        
        
    
