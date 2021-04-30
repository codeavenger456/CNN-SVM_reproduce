"""Utility functions module"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def load_tfds(
    name: str = "mnist"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a data set from `tfds`.

    Parameters
    ----------
    name : str
        The name of the TensorFlow data set to load.

    Returns
    -------
    train_features : np.ndarray
        The train features.
    test_features : np.ndarray
        The test features.
    train_labels : np.ndarray
        The train labels.
    test_labels : np.ndarray
        The test labels.
    """
    train_dataset = tfds.load(name=name, split=tfds.Split.TRAIN, batch_size=-1)
    train_dataset = tfds.as_numpy(train_dataset)

    train_features = train_dataset["image"]
    train_labels = train_dataset["label"]

    train_features = train_features.astype("float32")
    train_features = train_features / 255.0

    test_dataset = tfds.load(name=name, split=tfds.Split.TEST, batch_size=-1)
    test_dataset = tfds.as_numpy(test_dataset)

    test_features = test_dataset["image"]
    test_labels = test_dataset["label"]

    test_features = test_features.astype("float32")
    test_features = test_features / 255.0

    return train_features, test_features, train_labels, test_labels

def create_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    as_supervised: bool = True,
) -> tf.data.Dataset:
    """
    Returns a `tf.data.Dataset` object from a pair of
    `features` and `labels` or `features` alone.

    Parameters
    ----------
    features : np.ndarray
        The features matrix.
    labels : np.ndarray
        The labels matrix.
    batch_size : int
        The mini-batch size.
    as_supervised : bool
        Boolean whether to load the dataset as supervised or not.

    Returns
    -------
    dataset : tf.data.Dataset
        The dataset pipeline object, ready for model usage.
    """
    if as_supervised:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((features, features))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(features.shape[1])
    dataset.features = features
    dataset.labels = labels
    
    return dataset
    
class DataSet(object):

    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0 # How many epochs have passed
        self._index_in_epoch = 0 # Index in an epoch
        self._num_examples = images.shape[0]# refers to the total number of samples of training data
    
    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch 
        # Shuffle for the first epoch The first epoch needs shuffle
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)  #Generate an np.array of all sample lengths
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
            # Go to the next epoch


        if start + batch_size > self._num_examples: #epoch end and beginning of next epoch
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start  # In the end, there is not enough batch and a few remaining
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle: 
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end] 
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)    
        else:  # Except for the first epoch, and the beginning of each epoch, the remaining batch processing methods
            self._index_in_epoch += batch_size # start = index_in_epoch
            end = self._index_in_epoch #end is very simple, it is index_in_epoch plus batch_size 
            return self._images[start:end], self._labels[start:end] #In data x,y