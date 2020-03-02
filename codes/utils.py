import tensorflow as tf
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def one_hot_encoding(size, dim, y):
    labels = tf.fill([size, 1], y)
    return tf.reshape(tf.one_hot(labels, dim), [-1, dim])


def merge_images(sources, targets):
    """
    :param sources: svhn images shapes (batch_size, 32, 32, 3) 
    :param targets: mnist images shapes (batch_size, 32, 32, 1) 
    :return: 
    """
    batch_size, h, w, _ = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([row * h, row * w * 2, 3])

    num_to_print = row * row
    for idx, (source_img, target_img) in enumerate(zip(sources[:num_to_print], targets[:num_to_print])):
        i = idx // row
        j = idx % row
        merged[i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h, :] = source_img
        merged[i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h, :] = target_img
    return merged
