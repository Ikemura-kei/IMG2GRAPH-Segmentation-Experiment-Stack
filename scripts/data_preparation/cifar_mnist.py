import cv2 as cv
import os
from os.path import join, exists

import random
import numpy as np

import pickle
from mnist.loader import MNIST

def generate_cifar_mnist(cifar_dir, mnist_dir, save_dir, random_seed=10, threshold=170, max_nb_imgs=2000):
    mnist_imgs, mnist_labels = unpack_mnist(mnist_dir)
    num_mnist = len(mnist_imgs)

    cifar_imgs, cifar_labels = unpack_cifar(cifar_dir)

    print("mnist image shape", mnist_imgs[0].shape)
    print("cifar image shape", cifar_imgs[0].shape)

    cv.imwrite(os.path.join(mnist_dir, "trail_cifar.jpg"), cifar_imgs[1])
    cv.imwrite(os.path.join(mnist_dir, "trail_mnist.jpg"), mnist_imgs[1] * 255)

    # draw random cifar images
    random_cifar_indexes = np.random.randint(0, len(cifar_imgs), int(max_nb_imgs / 10) + 20)

    for iter, index in enumerate(random_cifar_indexes):
        if iter*10 > max_nb_imgs:
            return
        # draw 10 mnist digits at random
        rand_index = np.random.randint(0, num_mnist, 10)

        for iter_2, i in enumerate(rand_index):
            cifar_copy = cv.resize(cifar_imgs[index].copy(), (64, 64))
            cifar_copy = cifar_copy * 0.7 # surppress
            # read the mnist image
            mnist_img = mnist_imgs[i] * 255 # originally 0-1 scale
            width, height = mnist_img.shape

            # get the original segmentation map
            seg_map = mnist_img.copy()
            seg_map[seg_map<=threshold] = 0
            seg_map[seg_map>threshold] = mnist_labels[i]
            

            # generate a random position to overlay the digit
            randx = np.random.randint(0, cifar_copy.shape[0] - width - 1)
            randy = np.random.randint(0, cifar_copy.shape[1] - height - 1)

            # overlay the digit
            cifar_copy[randx:randx+width, randy:randy+height, 0] += mnist_img
            cifar_copy[randx:randx+width, randy:randy+height, 1] += mnist_img
            cifar_copy[randx:randx+width, randy:randy+height, 2] += mnist_img
            cifar_copy = np.clip(cifar_copy, 0, 255)

            # update segmentation map
            seg_map_final = np.zeros((cifar_copy.shape[0], cifar_copy.shape[1]))
            seg_map_final[randx:randx+width, randy:randy+height] = seg_map
            # print("seg_map_final max", np.max(seg_map), "seg_map_final min", np.min(seg_map))
            # print("unique", np.unique(seg_map_final))

            # save the data back
            # 1/5 into test set and 4/5 into train set
            if i % 5 == 0:
                cv.imwrite(os.path.join(save_dir, "test", "rgb", str(iter)+"_"+str(iter_2)+".jpg"), cifar_copy)
                np.save(os.path.join(save_dir, "test", "seg_map", str(iter)+"_"+str(iter_2)+".npy"), seg_map_final)
            else:
                cv.imwrite(os.path.join(save_dir, "train", "rgb", str(iter)+"_"+str(iter_2)+".jpg"), cifar_copy)
                np.save(os.path.join(save_dir, "train", "seg_map", str(iter)+"_"+str(iter_2)+".npy"), seg_map_final)

def unpickle(file):
    """
    unpickles cifar batches from the encoded files. Code from
    https://www.cs.toronto.edu/~kriz/cifar.html
    :param file:
    :return:
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpack_cifar(direc):
    """
    the data comes in batches. So this function concatenates the data from the batches
    :param direc: directory where the batches are located
    :return:
    """
    assert exists(direc), "directory does not exist"
    X,y = [], []
    for filename in os.listdir(direc):
        if filename[:5] == 'data_':
            data = unpickle(join(direc, filename))
            X.append(data[b'data'].reshape((10000,3,32,32)))
            y += data[b'labels']
    assert X, "No data was found in '%s'. Are you sure the CIFAR10 data is there?"%direc

    X = np.concatenate(X, 0)
    X = np.transpose(X, (0,2,3,1)).astype(np.float32)
    return X,y

def unpack_mnist(direc):
    """
    Unpack the MNIST data and put them in numpy arrays
    :param direc:
    :return:
    """
    assert exists(direc), "directory does not exist"
    try:
        mndata = MNIST(direc)
        images, labels = mndata.load_training()
    except FileNotFoundError as e:
        print('Make sure that you have downloaded the data and put in %s\n Also make sure that the spelling is correct. \
              the MNIST data comes in t10k-images.idx3-ubyte or t10k-images-idx3-ubyte. We expect the latter'%(direc))
        raise FileNotFoundError(e)
    X_mnist = np.array(images).reshape(60000, 28, 28)
    y_mnist = np.array(labels)

    X_mnist = X_mnist.astype(np.float32)/np.max(X_mnist)
    return X_mnist, y_mnist

if __name__ == "__main__":
    generate_cifar_mnist("/data1/kikemura/IMG2GRAPH_Segmentation_Experiment/dataset/cifar_mnist/cifar-10-batches-py", "/data1/kikemura/IMG2GRAPH_Segmentation_Experiment/dataset/cifar_mnist", "/data1/kikemura/IMG2GRAPH_Segmentation_Experiment/dataset/cifar_mnist")