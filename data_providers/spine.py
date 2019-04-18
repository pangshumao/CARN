import tempfile
import os
import pickle
import random
import scipy.io as sio
import math
import cv2
import numpy as np
import time
from data_providers.base_provider import ImagesDataSet, DataProvider
from skimage import exposure

def rotate(image, angle, center=None, scale=1.0):
    '''
    @author Shumao Pang, Southern Medical University, pangshumao@126.com
    :param image: a numpy array with shape of h * w
    :param angle:
    :param center:
    :param scale:
    :return:
    '''
    (h, w) = image.shape[:2]
    if center is None:
        center = (int(w / 2), int(h / 2))
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    init_shape = image.shape
    new_shape = [int(init_shape[0] + pad * 2),
                 int(init_shape[1] + pad * 2),
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape, dtype=image.dtype)
    zeros_padded[pad:int(init_shape[0] + pad), pad:int(init_shape[1] + pad), :] = image
    # zeros_padded = np.pad(image, ((pad,pad), (pad,pad), (0,0)), mode='symmetric')
    # randomly crop to original size
    init_x = np.random.randint(0, int(pad * 2))
    init_y = np.random.randint(0, int(pad * 2))
    cropped = zeros_padded[
        init_x: int(init_x + init_shape[0]),
        init_y: int(init_y + init_shape[1]),
        :]
    # rotate
    random_angle = np.random.randint(-10, 10)
    # random_angle = np.random.randint(-15, 15)
    rotated = rotate(cropped[:,:,0], random_angle)

    # gamma transformation
    # random_factor = np.random.randint(7, 14) / 10.
    # image_min = np.min(rotated)
    # image_max = np.max(rotated)
    # image_rescale = exposure.rescale_intensity(rotated, out_range=(0.0,1.0))
    # gamma = exposure.adjust_gamma(image_rescale, random_factor)
    # gamma = exposure.rescale_intensity(gamma, out_range=(image_min, image_max))
    # cropped[:,:,0] = gamma

    cropped[:,:,0] = rotated

    # cropped += np.random.normal(0, 0.03, size=init_shape)
    # flip = random.getrandbits(1)
    # if flip:
    #     cropped = cropped[:, ::-1, :]
    return cropped

# def augment_image(image, mask):
#     image_shape = image.shape
#     new_image = image + np.random.normal(0, 0.03, size=image_shape) * mask
#     return new_image

def augment_all_images(initial_images, pad=4):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=pad)
    return new_images

class SpineDataSet(ImagesDataSet):
    def __init__(self, images, labels, recon_labels, adjacentMatrix, n_classes, shuffle, normalization,
                 augmentation):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            recon_labels: 2D numpy array
            adjacentMatrix: 2D numpy array, the similarity adjacent maxtrix
            n_classes: `int`, the dimension of output
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            augmentation: `bool`
        """
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels, recon_labels, adjacentMatrix = self.shuffle_images_and_labels(images, labels, recon_labels, adjacentMatrix)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")
        self.images = images
        self.labels = labels
        self.recon_labels = recon_labels
        self.adjacentMatrix = adjacentMatrix
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization
        self.images = self.normalize_images(images, self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels, recon_labels, adjacentMatrix = self.shuffle_images_and_labels(
                self.images, self.labels, self.recon_labels, self.adjacentMatrix)
        else:
            images, labels, recon_labels, adjacentMatrix = self.images, self.labels, self.recon_labels, self.adjacentMatrix
        # start = time.time()
        if self.augmentation:
            images = augment_all_images(images, pad=4)
        # end = time.time()
        # print(end-start)
        self.epoch_images = images
        self.epoch_labels = labels
        self.epoch_recon_labels = recon_labels
        self.epoch_adjacentMatrix = adjacentMatrix

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        recon_labels_slice = self.epoch_recon_labels[start: end]
        adjacentMatrix_slice = self.epoch_adjacentMatrix[start: end][:, start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice, recon_labels_slice, adjacentMatrix_slice


class SpineDataProvider(DataProvider):
    """Abstract class for cifar readers"""

    def __init__(self, data_path=None, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None,
                  folderInd=None, knn=None, **kwargs):
        """
        Args:
            data_path: `str`
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            folderInd: int, range from 1 to 5
        """
        self._data_path = data_path
        self.folderInd = folderInd
        self.knn = knn
        self.data_augmentation = False
        data = np.load(os.sep.join([data_path, 'samples.npz']))
        x = data['x']
        y = data['y']
        self._n_classes = y.shape[-1]
        self.height = data['height']
        self.width = data['width']
        self.pixelSizes = data['pixelSizes']
        indData = sio.loadmat(os.sep.join([data_path, 'fold' + str(folderInd) + '-ind.mat']))
        self.trainInd = indData['trainInd'].flatten()
        self.valInd = indData['valInd'].flatten()

        # load reconstruction data by LAE
        recon_data = sio.loadmat(os.sep.join([data_path, 'fold' + str(self.folderInd) + '_recon_y_k_' + str(self.knn) + '.mat']))
        recon_train_y = recon_data['train_recon_y']
        recon_val_y = recon_data['val_recon_y']
        train_S = recon_data['train_S']
        val_S = recon_data['val_S']

        # calculate the sigma of adaptive loss
        allEuDist = np.sqrt(np.sum(np.square(recon_train_y - y[self.trainInd]), axis=1))
        self.sigma = np.mean(allEuDist) / 2.0

        # add train and validations datasets
        images = x[self.trainInd]
        labels = y[self.trainInd]
        recon_labels = recon_train_y

        if validation_set is not None and validation_split is not None:
            split_idx = int(images.shape[0] * (1 - validation_split))
            self.train = SpineDataSet(
                images=images[:split_idx], labels=labels[:split_idx], recon_labels=recon_labels[:split_idx], adjacentMatrix=train_S[:split_idx][:,:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
            self.validation = SpineDataSet(
                images=images[split_idx:], labels=labels[split_idx:], recon_labels=recon_labels[split_idx:], adjacentMatrix=train_S[split_idx:][:,split_idx:],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
        else:
            self.train = SpineDataSet(
                images=images, labels=labels, recon_labels=recon_labels, adjacentMatrix=train_S,
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)

        # add visualization set
        self.visualization = SpineDataSet(
            images=images[:16], labels=labels[:2], recon_labels=recon_labels[:2], adjacentMatrix=train_S[:2][:,:2],
            n_classes=self.n_classes, shuffle=None,
            normalization=normalization,
            augmentation=False)

        # add test set
        images = x[self.valInd]
        labels = y[self.valInd]
        recon_labels = recon_val_y
        self.test = SpineDataSet(
            images=images, labels=labels, recon_labels=recon_labels, adjacentMatrix=val_S,
            shuffle=None, n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test

    @property
    def data_path(self):
        return self._data_path

    @property
    def data_shape(self):
        return (512, 256, 1)

    @property
    def n_classes(self):
        return self._n_classes

class SpineAugmentedDataProvider(DataProvider):
    """Abstract class for cifar readers"""

    def __init__(self, data_path=None, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None,
                  folderInd=None, knn=None, **kwargs):
        """
        Args:
            data_path: `str`
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            folderInd: int, range from 1 to 5
        """
        self._data_path = data_path
        self.folderInd = folderInd
        self.knn = knn
        self.data_augmentation = True
        data = np.load(os.sep.join([data_path, 'samples.npz']))
        x = data['x']
        y = data['y']
        self._n_classes = y.shape[-1]
        self.height = data['height']
        self.width = data['width']
        self.pixelSizes = data['pixelSizes']
        indData = sio.loadmat(os.sep.join([data_path, 'fold' + str(folderInd) + '-ind.mat']))
        self.trainInd = indData['trainInd'].flatten()
        self.valInd = indData['valInd'].flatten()

        # load reconstruction data by LAE
        recon_data = sio.loadmat(
            os.sep.join([data_path, 'fold' + str(self.folderInd) + '_recon_y_k_' + str(self.knn) + '.mat']))
        recon_train_y = recon_data['train_recon_y']
        recon_val_y = recon_data['val_recon_y']
        train_S = recon_data['train_S']
        val_S = recon_data['val_S']

        # calculate the sigma of adaptive loss
        allEuDist = np.sqrt(np.sum(np.square(recon_train_y - y[self.trainInd]), axis=1))
        self.sigma = np.mean(allEuDist)

        # add train and validations datasets
        images = x[self.trainInd]
        labels = y[self.trainInd]
        recon_labels = recon_train_y

        if validation_set is not None and validation_split is not None:
            split_idx = int(images.shape[0] * (1 - validation_split))
            self.train = SpineDataSet(
                images=images[:split_idx], labels=labels[:split_idx], recon_labels=recon_labels[:split_idx], adjacentMatrix=train_S[:split_idx][:,:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
            self.validation = SpineDataSet(
                images=images[split_idx:], labels=labels[split_idx:], recon_labels=recon_labels[split_idx:], adjacentMatrix=train_S[split_idx:][:,split_idx:],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
        else:
            self.train = SpineDataSet(
                images=images, labels=labels, recon_labels=recon_labels, adjacentMatrix=train_S,
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)

        # add visualization set
        self.visualization = SpineDataSet(
            images=images[:16], labels=labels[:2], recon_labels=recon_labels[:2], adjacentMatrix=train_S[:2][:,:2],
            n_classes=self.n_classes, shuffle=None,
            normalization=normalization,
            augmentation=False)

        # add test set
        images = x[self.valInd]
        labels = y[self.valInd]
        recon_labels = recon_val_y
        self.test = SpineDataSet(
            images=images, labels=labels, recon_labels=recon_labels, adjacentMatrix=val_S,
            shuffle=None, n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test

    @property
    def data_path(self):
        return self._data_path

    @property
    def data_shape(self):
        return (512, 256, 1)

    @property
    def n_classes(self):
        return self._n_classes